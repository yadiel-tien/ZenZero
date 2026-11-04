"""竞技场执行引擎：admin 服务内的进程级管理器（多场次、线程跑批、内存心跳）。

场次模型：棋种 → N 个竞技场（default 恒在，可新建/删除）；每场独立名册、Elo 榜、历史。
同一时刻全局只跑一场战役（GPU 单卡），心跳带 (game, arena) 标识，前端按场过滤。
arena 线程复用服务进程的推理 hub（同代模型自动共享引擎），不再另起子进程。
"""
import json
import os
import re
import shutil
import threading
import time

from core.utils.config import CONFIG
from core.utils.arena import Arena

_STATE_FILE = os.path.join(CONFIG.get('log_dir', 'logs'), '_state', 'arena.json')
_NAME_OK = re.compile(r'^[\w\u4e00-\u9fff+-]{1,32}$')


class ArenaEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[tuple[str, str], threading.Thread] = {}
        self._cache: dict[tuple[str, str], Arena] = {}
        self._state: dict = {'running': False}

    # ---------- 场次缓存 ----------
    def get(self, game: str, arena: str = 'default') -> Arena:
        key = (game, arena or 'default')
        with self._lock:
            if key not in self._cache:
                self._cache[key] = Arena(game, key[1])
            return self._cache[key]

    # ---------- 场次 CRUD ----------
    def list_arenas(self, game: str) -> list[dict]:
        base = os.path.join(CONFIG['data_dir'], game, 'rates')
        names = {'default'}
        try:
            names |= {d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))}
        except OSError:
            pass
        out = []
        for n in sorted(names, key=lambda x: (x != 'default', x)):
            try:
                a = self.get(game, n)
                out.append({'name': n, 'players': len(a.rates),
                            'total_games': sum(e.n_games_played for e in a.rates.values()) // 2})
            except Exception:                             # noqa: BLE001 目录残缺不炸列表
                out.append({'name': n, 'players': 0, 'total_games': 0})
        return out

    def create(self, game: str, name: str) -> dict:
        name = (name or '').strip()
        if not _NAME_OK.match(name):
            return {'ok': False, 'error': '名字只许中英文数字 + - ，≤32 字'}
        d = os.path.join(CONFIG['data_dir'], game, 'rates', name)
        if os.path.exists(d):
            return {'ok': False, 'error': f'「{name}」已存在'}
        os.makedirs(d)
        with open(os.path.join(d, 'candidates.txt'), 'w') as f:
            f.write('')                                    # 新建竞技场默认空名册，手动点将
        self.get(game, name)
        return {'ok': True, 'name': name}

    def delete(self, game: str, name: str) -> dict:
        if name in ('', 'default'):
            return {'ok': False, 'error': 'default 场不可删（清空名册即可）'}
        key = (game, name)
        if key in self._threads and self._threads[key].is_alive():
            return {'ok': False, 'error': '这座竞技场正在打仗，先停'}
        d = os.path.join(CONFIG['data_dir'], game, 'rates', name)
        if not os.path.isdir(d):
            return {'ok': False, 'error': '不存在'}
        shutil.rmtree(d)
        self._cache.pop(key, None)
        return {'ok': True}

    # ---------- 名册 ----------
    def roster(self, game: str, arena: str, add: list[str] | None = None,
               remove: list[str] | None = None) -> dict:
        a = self.get(game, arena)
        cur = set(a.rates)
        cur |= set(add or [])
        cur -= set(remove or [])
        keys = sorted(cur)
        a.sync_roster(keys)
        return {'ok': True, 'roster': keys, 'candidates': a.discover()}

    def _preflight(self, game: str, keys: list[str]) -> str | None:
        """开赛前显存预检：名册模型常驻需求 > 空闲显存就明确拒绝。

        Network 载不动会静默降级 CPU——m 档在 CPU 上一局要几小时，用户只会看到
        “点了开始没反应”。宁可开赛前一次说清楚（2026-11-04 实测坑）。"""
        try:
            import torch
            from core.utils.config import net_dir
        except Exception:                                  # noqa: BLE001
            return None
        if not str(CONFIG.get('device', '')).startswith('cuda') or not torch.cuda.is_available():
            return None
        try:
            free, _ = torch.cuda.mem_get_info(torch.device(CONFIG['device']).index or 0)
        except Exception:                                  # noqa: BLE001
            return None
        need, detail = 0, []
        for k in keys:
            prof, _, idx = k.rpartition('/')
            d = net_dir(game, prof)
            path = next((os.path.join(d, f) for f in (f'checkpoint_{idx}.pt', f'model_{idx}.pt')
                         if os.path.exists(os.path.join(d, f))), None)
            if not path:
                continue
            n = os.path.getsize(path) * 1.35 + 600 * 1024 * 1024
            need += n
            detail.append((k, n))
        if need > free * 0.95:
            top = '、'.join(f'{k}≈{n / 2 ** 30:.1f}G' for k, n in sorted(detail, key=lambda x: -x[1])[:3])
            return (f'显存不够开赛：名册 {len(detail)} 个模型需常驻 ≈{need / 2 ** 30:.1f}G，'
                    f'当前空闲 {free / 2 ** 30:.1f}G（大块头：{top}）。'
                    '先停常驻大脑，或把重网络移出这场名册。')
        return None

    # ---------- 战役 ----------
    def _busy(self) -> tuple[str, str] | None:
        for k, t in self._threads.items():
            if t.is_alive():
                return k
        return None

    def start(self, game: str, arena: str = 'default', mode: str = 'smart',
              games: int = 30, sims: int = 300, tables: int | None = None,
              pair: list[str] | None = None) -> dict:
        busy = self._busy()
        if busy:
            return {'ok': False, 'error': f'「{busy[0]}·{busy[1]}」竞技场正在打仗，先停止'}
        a = self.get(game, arena)
        if not a.rates:
            return {'ok': False, 'error': '名册为空 — 先到「名册管理」添加选手'}
        bad = self._preflight(game, sorted(a.rates))
        if bad:
            return {'ok': False, 'error': bad}
        pr = None
        if mode == 'duel':
            if not pair or len(pair) != 2 or any(x not in a.rates for x in pair):
                return {'ok': False, 'error': '自定义对决需从本场名册中选两名选手'}
            pr = (pair[0], pair[1])
        started = time.time()

        def beat(st: dict) -> None:
            st.update({'game': game, 'arena': arena, 'started_ts': started,
                       'mode': mode, 'sims': sims, 'roster': len(a.rates)})
            st.setdefault('tables', int(tables) if tables else '自动')  # 波内回调已带实时桌数，别覆盖
            self._state = st
            try:
                os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
                tmp = _STATE_FILE + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(st, f, ensure_ascii=False)
                os.replace(tmp, _STATE_FILE)
            except OSError:
                pass

        def _run():
            try:
                a.run_batch(games=int(games), n_sim=int(sims),
                            tables=int(tables) if tables else None,
                            mode=mode, pair=pr, on_state=beat)
            except Exception as exc:                        # noqa: BLE001 线程兜底留痕
                self._state = {**self._state, 'running': False, 'error': str(exc)[:200]}

        beat({'running': True, 'mode': mode, 'pair': list(pr) if pr else None,
              'done': 0, 'total': games, 'tally': [0, 0, 0], 'feed': [], 'gpm': 0})
        t = threading.Thread(target=_run, name=f'arena-{game}-{arena}', daemon=True)
        self._threads[(game, arena)] = t
        t.start()
        return {'ok': True, 'roster': len(a.rates), 'games': games}

    def stop(self) -> dict:
        busy = self._busy()
        if busy:
            self._cache[busy].stop_event.set()
            return {'ok': True, 'note': f'已请求停止竞技场「{busy[1]}」：进行中的桌打完本批即收'}
        return {'ok': True, 'note': '没有在打的战役'}

    # ---------- 查询 ----------
    def state(self) -> dict:
        st = dict(self._state)
        st['running'] = bool(self._busy())
        if st['running'] and st.get('started_ts'):
            st['age_s'] = round(time.time() - st['started_ts'], 1)
        return st

    def ranks(self, game: str, arena: str = 'default') -> dict:
        a = self.get(game, arena)
        if not a.rates:
            a.load()
        players = a.ranks()
        rival = {p['id']: p['records'] for p in players if p['records']}
        return {'game': game, 'arena': arena, 'players': players, 'rival': rival,
                'candidates': a.discover(),
                'running': self._busy() == (game, arena)}


_engine: ArenaEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> ArenaEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ArenaEngine()
        return _engine
