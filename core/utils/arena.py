import os
import re
import json
import pickle
import random
import threading
import time

from core.env.functions import get_class
from core.player.ai_server import AIServer
from services.inference.hub import ServerHub
from core.utils.config import CONFIG
from core.utils.elo import Elo
from core.utils.logger import get_logger
from core.utils.types import EnvName
from concurrent.futures import ThreadPoolExecutor, as_completed

# v2（2026-11-04）：跨网络天梯。
# - 选手键从纯 int 升级为 "profile/index"（mini/50 与 mini2/50 是两个不同模型，旧键会撞车，
#   且旧 versus 不给 AIServer 传 profile，多网络根本没法同台）。
# - 修复两个并发雷：多线程同时 pickle.dump 同一个 rates 文件（加锁+原子替换）；每局全量落盘
#   （节流：5 局一存 + 批尾必存）。
# - 移除 run_gauntlet 里 `hero_idx < 56: continue` 的实验残留。
# - 新增：名册自动发现（扫 checkpoint 目录）、逐局状态回调（前端直播卡）、比分历史（sparkline 数据源）。
# 旧 rates.pkl（int 键）不再读取，文件原地保留。


def key_of(profile: str, index: int) -> str:
    return f'{profile}/{index}'


def parse_key(key: str) -> tuple[str, int]:
    p, _, i = key.rpartition('/')
    return p, int(i)


_ARENA_NAME = re.compile(r'^[\w\u4e00-\u9fff+-]{1,32}$')


class Arena:
    """一座竞技场 = 一个独立榜单（名册/Elo/历史各自成目录）。
    default 场每棋种自动存在；新建场按名字落 data/<Game>/rates/<name>/。"""

    def __init__(self, env_name: EnvName, arena: str = 'default') -> None:
        arena = (arena or 'default').strip()
        if not _ARENA_NAME.match(arena):
            raise ValueError('竞技场名只许中英文数字 + -，≤32 字')
        self.arena = arena
        self.env_name = env_name
        self.logger = get_logger('arena')
        self.rates: dict[str, Elo] = {}
        self.history: list[dict] = []          # [{t, scores:{key:score}}] 最近 200
        self._save_lock = threading.Lock()     # pkl 串行写（旧版 10 桌并发 dump 同一文件必竞态）
        self._delta_lock = threading.Lock()    # Elo 更新串行
        self._games_since_save = 0
        self.stop_event = threading.Event()
        # 进程内推理引擎管理中心，多个 AIServer 共享引擎（同网同代自动复用）
        self.hub = ServerHub()
        self.load()

    # ---------- 名册 ----------
    def discover(self) -> list[str]:
        """全部已知档位的检查点 → 候选键。

        必须走 list_all_indices（与资产页同一目录口径）：m 档住 data/<Game>/ 根目录
        （历史零迁移布局），扫子目录的野路子会把整个 m 网络漏掉——2026-11-04 实测。"""
        from core.utils.config import available_profiles
        from core.network.functions import list_all_indices
        out = []
        for prof in sorted(available_profiles(self.env_name)):
            out += [key_of(prof, i) for i in list_all_indices(self.env_name, prof)]
        # 网络名升序、代数降序（数字序，mini/2 排在 mini/12 前会乱，这里大代在前）
        return sorted(out, key=lambda k: (k.rsplit('/', 1)[0], -int(k.rsplit('/', 1)[1])))

    def sync_roster(self, keys: list[str] | None) -> list[str]:
        """设定参赛名册并落 candidates.txt；keys=None 表示沿用现有名册（空则全发现）。"""
        if keys is None:
            if os.path.exists(self.list_path):
                with open(self.list_path) as f:
                    keys = [x.strip() for ln in f for x in ln.split(',') if x.strip()]
            if not keys:
                keys = self.discover()
        os.makedirs(os.path.dirname(self.list_path), exist_ok=True)
        with open(self.list_path, 'w') as f:
            f.write(','.join(keys) + '\n')
        for k in keys:
            self.rates.setdefault(k, Elo(k))
        for stale in set(self.rates) - set(keys):
            self.rates.pop(stale)
        return sorted(self.rates)

    # ---------- 对弈 ----------
    def pick_pair(self) -> tuple[str, str] | None:
        """智能约架：出场少者优先 × 分近者优先（沿用旧权重策略，键升级版）。"""
        keys = list(self.rates)
        if len(keys) < 2:
            return None
        w1 = [1 / (self.rates[k].n_games_played + 1) for k in keys]
        a = random.choices(keys, weights=w1, k=1)[0]
        rest = [k for k in keys if k != a]
        w2 = [1 / (abs(self.rates[k].scores - self.rates[a].scores) + 40) for k in rest]
        b = random.choices(rest, weights=w2, k=1)[0]
        return a, b

    def versus(self, k1: str, k2: str, n_sim: int = 300) -> str:
        """一局对弈，返回 'w'（k1 胜）/'l'/'d'，并按结果更新 Elo。"""
        prof1, i1 = parse_key(k1)
        prof2, i2 = parse_key(k2)
        env = get_class(self.env_name)()
        with AIServer(self.env_name, i1, n_simulation=n_sim, verbose=False,
                      hub=self.hub, profile=prof1) as p1, \
             AIServer(self.env_name, i2, n_simulation=n_sim, verbose=False,
                      hub=self.hub, profile=prof2) as p2:
            outcome = env.random_order_play((p1, p2), silent=True)
        e1, e2 = self.rates[k1], self.rates[k2]
        with self._delta_lock:
            if outcome == 1:
                e1.defeat(e2); result = 'w'
            elif outcome == 0:
                e2.defeat(e1); result = 'l'
            else:
                e1.draw(e2); result = 'd'
            self._games_since_save += 1
            if self._games_since_save >= 5:
                self.save()
        self.logger.info(f'{k1} vs {k2}: {result}  [{e1.scores:.0f}/{e2.scores:.0f}]')
        return result

    def _hub_stats(self) -> dict:
        """取本 arena 名下引擎里“批最大”的一台的实时统计（两选手两台引擎时看最忙者）。"""
        best: dict = {}
        for eng in list(self.hub.engines.values()):
            try:
                st = eng.infer_stats()
            except Exception:                             # noqa: BLE001
                continue
            if not best or st.get('batch_mean', 0) >= best.get('batch_mean', 0):
                best = st
        return best

    def _auto_tune_tables(self, t: int, st: dict) -> int:
        """桌数自适应：与 selfplay 的 K 同一信号学（桌数×每桌在途=批的供给量）。
        m 档前向 15ms+ 时 5 桌就顶满 GPU，mini 档 10 桌也喂不饱——固定值必有一边错。"""
        fwd = float(st.get('fwd_ms_mean') or 0)
        batch = float(st.get('batch_recent_mean') or st.get('batch_mean') or 0)
        if not st or fwd <= 0:
            return t
        if fwd > 15 or batch > 2400:
            return max(2, t - 2 if t > 4 else t - 1)
        if batch < 700 and fwd < 8:
            return min(12, t + 2)
        return t

    def run_batch(self, games: int, n_sim: int = 300, tables: int | None = None,
                  mode: str = 'smart', pair: tuple[str, str] | None = None,
                  on_state=None) -> dict:
        """跑一批天梯赛。mode: smart=自动约架 / duel=指定 pair 连打。
        tables=None 自动（初始 4，按每波实测增减 2）；on_state 每局回调（直播心跳）。"""
        self.stop_event.clear()
        tally = {'w': 0, 'd': 0, 'l': 0}
        feed: list[str] = []
        done = 0
        t0 = time.time()
        t = int(tables) if tables else 4
        auto = not tables
        dev0 = str(CONFIG.get('device', ''))
        while done < games and not self.stop_event.is_set():
            # 显存不足 -> Network 会把全局 device 降成 cpu（m 档在 CPU 上一局要几小时），
            # 批次就此中止并明确报错，绝不静默爬行（2026-11-04 实测坑）。
            if dev0.startswith('cuda') and not str(CONFIG.get('device', '')).startswith('cuda'):
                err = '显存不足被降级 CPU，批次已中止（重网络常驻显存不够，先停常驻大脑或移出名册）'
                self.logger.error(err)
                if on_state:
                    on_state({'running': False, 'mode': mode, 'pair': None, 'done': done,
                              'total': games, 'tally': [tally['w'], tally['d'], tally['l']],
                              'feed': feed[-40:], 'gpm': 0, 'error': err,
                              'infer': self._hub_stats(), 'device': str(CONFIG.get('device'))})
                self.save()
                return tally
            wave = min(t, games - done)
            prs, seen = [], set()
            for _ in range(wave):
                if mode == 'duel' and pair:
                    pr = pair
                else:
                    pr = None
                    for _try in range(8):    # 一波内不排同一对阵（4 桌里出现两桌一样的纯属浪费）
                        cand = self.pick_pair()
                        if cand is None:
                            break
                        key = tuple(sorted(cand))
                        if key not in seen:
                            pr = cand
                            break
                        pr = cand
                if pr is None:
                    break
                seen.add(tuple(sorted(pr)))
                prs.append(pr)
            if not prs:
                break
            if on_state:     # 波次开打即回报：前端据此立刻画当前对阵与桌数
                on_state({'running': True, 'mode': mode, 'pair': None, 'done': done,
                          'total': games, 'tally': [tally['w'], tally['d'], tally['l']],
                          'feed': feed[-40:], 'tables': t if not auto else f'自动{t}',
                          'pairs': [list(x) for x in prs], 'infer': self._hub_stats(),
                          'device': str(CONFIG.get('device', '')),
                          'gpm': round(done / max(time.time() - t0, 1e-9) * 60, 1)})
            with ThreadPoolExecutor(max_workers=len(prs)) as pool:
                futs = [pool.submit(self._safe_versus, a, b, n_sim) for a, b in prs]
                for fut in as_completed(futs):
                    r = fut.result()
                    if r is None:
                        continue
                    tally[r] += 1
                    feed.append(r)
                    done += 1
                    if on_state:
                        el = time.time() - t0
                        on_state({'running': True, 'mode': mode,
                                  'pair': list(pair) if pair else None,
                                  'done': done, 'total': games,
                                  'tally': [tally['w'], tally['d'], tally['l']],
                                  'feed': feed[-40:],
                                  'tables': t if not auto else f'自动{t}',
                                  'pairs': [list(x) for x in prs],
                                  'infer': self._hub_stats(),
                                  'device': str(CONFIG.get('device', '')),
                                  'gpm': round(done / max(el, 1e-9) * 60, 1)})
            if auto:
                new_t = self._auto_tune_tables(t, self._hub_stats())
                if new_t != t:
                    self.logger.info(f'[auto] 均批{self._hub_stats().get("batch_recent_mean", 0):.0f} '
                                     f'前向{self._hub_stats().get("fwd_ms_mean", 0):.1f}ms → 桌数 {t}→{new_t}')
                    t = new_t
        self.save()
        if on_state:
            on_state({'running': False, 'mode': mode, 'pair': None, 'done': done,
                      'total': games, 'tally': [tally['w'], tally['d'], tally['l']],
                      'feed': feed[-40:], 'gpm': 0})
        return tally

    def _safe_versus(self, k1: str, k2: str, n_sim: int) -> str | None:
        try:
            if self.stop_event.is_set():
                return None
            return self.versus(k1, k2, n_sim)
        except Exception as exc:                            # noqa: BLE001 单局失败不折整批
            self.logger.error(f'对局失败 {k1} vs {k2}: {exc}')
            return None

    # CLI 兼容（scripts/arena.py）
    def run(self, n_games: int) -> None:
        self.run_batch(n_games)

    # ---------- 持久化 ----------
    def snapshot_history(self) -> None:
        self.history.append({'t': int(time.time()),
                             'scores': {k: round(e.scores, 1) for k, e in self.rates.items()}})
        self.history = self.history[-200:]
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False)
        except OSError:
            pass

    def save(self) -> None:
        with self._save_lock:                  # 串行 + 原子替换：杀在半路不毁档
            self._games_since_save = 0
            os.makedirs(os.path.dirname(self.list_path), exist_ok=True)
            tmp = self.rates_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(self.rates, f, protocol=4)
            os.replace(tmp, self.rates_path)
            self.snapshot_history()

    def load(self) -> None:
        self.rates = {}
        if os.path.exists(self.rates_path):
            try:
                with open(self.rates_path, 'rb') as f:
                    r = pickle.load(f)
                if r and all(isinstance(k, str) for k in r):
                    self.rates = r             # v2 键（str）；旧 int 键档忽略重开
            except Exception as exc:                        # noqa: BLE001
                self.logger.warning(f'rates 读取失败，重开空榜：{exc}')
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, encoding='utf-8') as f:
                    self.history = json.load(f)
            except (OSError, ValueError):
                self.history = []
        if os.path.exists(self.list_path):
            with open(self.list_path) as f:
                for k in [x.strip() for ln in f for x in ln.split(',') if x.strip()]:
                    self.rates.setdefault(k, Elo(k))

    @property
    def _dir(self) -> str:
        return os.path.join(CONFIG['data_dir'], self.env_name, 'rates', self.arena)

    @property
    def rates_path(self) -> str:
        return os.path.join(self._dir, 'rates_v2.pkl')

    @property
    def list_path(self) -> str:
        return os.path.join(self._dir, 'candidates.txt')

    @property
    def history_path(self) -> str:
        return os.path.join(self._dir, 'history.json')

    # ---------- 展示 ----------
    def ranks(self) -> list[dict]:
        """榜单序列化（前端直接吃）。hist 从 history.json 逐存快照抽各自轨迹。"""
        trails: dict[str, list[float]] = {}
        for snap in self.history[-30:]:
            for k, v in snap['scores'].items():
                t = trails.setdefault(k, [])
                if not t or t[-1] != v:
                    t.append(v)
                    trails[k] = t[-14:]
        ranked = sorted(self.rates.values(), key=lambda e: e.scores, reverse=True)
        return [{'id': e.index, 'profile': parse_key(e.index)[0], 'index': parse_key(e.index)[1],
                 'score': round(e.scores, 1), 'games': e.n_games_played,
                 'records': {k: v for k, v in list(e.records.items())[:60]},
                 'hist': trails.get(e.index, [])}
                for e in ranked]

    def show_rank(self, limit: int = 10, ascend=True) -> None:
        if not self.rates:
            print('No candidate rates to show.')
            return
        for idx, (_, v) in enumerate(sorted(self.rates.items(), key=lambda x: x[1].scores, reverse=ascend)):
            if idx >= limit:
                break
            print(f'{idx + 1:>3}  {v.index:<14} score={v.scores:7.1f}  games={v.n_games_played}')
