"""训练监控的数据采集层：只读 + 带缓存，绝不碰训练进程。

四个数据源：
1. logs/perf/perf.jsonl —— 每轮一行结构化吞吐档案（[perf] 行的机器可读版）
2. logs/{selfplay,fit,inference}/latest.log 尾部 —— 当前轮/当前阶段的实时数
3. runs/*/events.out.tfevents.* —— TensorBoard 曲线（Loss/Val/Perf）
4. /proc + nvidia-smi —— CPU/内存/负载/GPU

设计要点：
- 全部只读，不做任何写入（除了自己的内存 ring buffer）；
- TensorBoard 事件文件解析较慢，按 mtime + TTL 缓存并增量 Reload；
- 日志只读文件尾部若干 KB，不整文件扫描（latest.log 可能几百 MB）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque

from core.utils.config import CONFIG, get_profile, net_config_for, game_name

PERF_JSONL = os.path.join(CONFIG['log_dir'], 'perf', 'perf.jsonl')

# 注意格式随版本演进了两次：`batch mean N (max M, top ...)` → 现在多了尾部均值段
# `batch mean N (recent R) (max M, top ...)`。正则必须容忍可选段，否则整个吞吐卡片变空。
_INFER_RE = re.compile(
    r'\[infer\] (?P<req_s>[\d.]+) req/s \| batch mean (?P<batch>[\d.]+)'
    r'(?: \(recent (?P<batch_recent>[\d.]+)\))?'
    r' \(max (?P<batch_max>\d+), top [^)]*\) \| window (?P<window>[\d.]+)ms'
    r' \| fwd (?P<fwd>[\d.]+)ms \| post (?P<post>[\d.]+)ms'
    r' \| pending (?P<pending>\d+) \| total (?P<total>\d+)')
_GAME_END_RE = re.compile(r'Game end\. Winner: (-?\d+), steps: (\d+)')
_STEP_RE = re.compile(r'Step (\d+):')
_LOSS_RE = re.compile(r'loss=([\d.]+), policy_loss=([\d.]+), value_loss=([\d.]+)')
# fit 结束时那一行：iteration_to_remove8:1171轮训练完成，共用时92.65秒。
_FIT_DONE_RE = re.compile(r'iteration_to_remove(\d+):(\d+)轮训练完成，共用时([\d.]+)秒')
_VAL_RE = re.compile(r'Val loss - policy ([\d.]+), value ([\d.]+)'
                     r'(?: \| KL ([\d.]+), 目标熵 ([\d.]+))?')
_EVAL_RE = re.compile(r'Model (\d+) VS (\d+)，胜:(\d+),负:(\d+),平:(\d+),总:(\d+), 胜率([\d.]+)%')

# TensorBoard 里值得画的曲线（白名单，避免把几百个标量全 dump 出去）
TB_TAGS = ('Loss/total', 'Loss/policy', 'Loss/value', 'Val/total', 'Val/policy', 'Val/value',
           'Val/policy_kl', 'Val/policy_entropy',
           'Learning Rate', 'Perf/selfplay_sims_per_s', 'Perf/fit_samples_per_s',
           'Perf/eval_games_per_min', 'Perf/infer_batch_mean', 'Perf/mcts_select_ms_per_sim',
           'Perf/mcts_wait_ms_per_sim')


def _tail(path: str, max_bytes: int = 120_000) -> list[str]:
    """读文件尾部若干字节（不整文件读），返回完整行列表。"""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fp:
            if size > max_bytes:
                fp.seek(-max_bytes, os.SEEK_END)
                fp.readline()                      # 丢掉被截断的半行
            return fp.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []


# ------------------------------------------------------------------ 硬件采样
class HardwareSampler:
    """后台线程每秒采一次 CPU/内存/GPU，维护 ~10 分钟环形历史。"""

    def __init__(self, history: int = 600):
        self._hist = deque(maxlen=history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prev = self._cpu_times()
        self._gpu_ok = None            # None=未知, False=nvidia-smi 用不了
        self._gpu_cache = (0.0, None)
        # 先同步采一次（否则服务刚起来的前几秒 hardware 是 null，面板空一片）；
        # 隔 0.2s 让 /proc/stat 有可算的增量
        time.sleep(0.2)
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name='hw-sampler', daemon=True)
        self._thread.start()

    @staticmethod
    def _cpu_times() -> tuple[int, int]:
        with open('/proc/stat') as fp:
            parts = [int(x) for x in fp.readline().split()[1:]]
        idle = parts[3] + parts[4]
        return idle, sum(parts)

    def _cpu_pct(self) -> tuple[float, list[float]]:
        """整体利用率 + 每核利用率（每核从 /proc/stat 的 cpuN 行算，需要前后两次）"""
        now = self._cpu_times()
        idle_d, total_d = now[0] - self._prev[0], now[1] - self._prev[1]
        self._prev = now
        pct = 100.0 * (1 - idle_d / total_d) if total_d > 0 else 0.0
        per_core: list[float] = []
        try:
            rows = [l for l in open('/proc/stat') if l.startswith('cpu') and l[3].isdigit()]
            for line in rows:
                f = [int(x) for x in line.split()[1:]]
                per_core.append(100.0 * (1 - (f[3] + f[4]) / max(sum(f), 1)))
        except OSError:
            pass
        return pct, per_core

    @staticmethod
    def _mem() -> dict:
        info = {}
        with open('/proc/meminfo') as fp:
            for line in fp:
                k, _, v = line.partition(':')
                info[k] = int(v.strip().split()[0])          # kB
        total = info.get('MemTotal', 1) / 1024 / 1024
        avail = info.get('MemAvailable', info.get('MemFree', 0)) / 1024 / 1024
        return {'total_gb': round(total, 1), 'used_gb': round(total - avail, 1),
                'pct': round(100 * (total - avail) / total, 1)}

    def _gpu(self) -> dict | None:
        """nvidia-smi 轮询（1s 缓存）。沙箱/无卡环境下优雅返回 None。"""
        if self._gpu_ok is False:
            return None
        ts, cached = self._gpu_cache
        if time.time() - ts < 1.0:
            return cached
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2)
            if out.returncode != 0 or not out.stdout.strip():
                raise OSError(out.stderr[:120])
            f = [x.strip() for x in out.stdout.strip().splitlines()[0].split(',')]
            gpu = {'name': f[0], 'util': float(f[1]), 'mem_used_mb': float(f[2]),
                   'mem_total_mb': float(f[3]), 'temp_c': float(f[4]), 'power_w': float(f[5])}
            self._gpu_ok = True
        except Exception:                                   # noqa: BLE001 无卡/无权限都算不可用
            gpu = None
            self._gpu_ok = False
        self._gpu_cache = (time.time(), gpu)
        return gpu

    def _sample_once(self) -> None:
        cpu, cores = self._cpu_pct()
        sample = {'t': time.time(), 'cpu': round(cpu, 1), 'mem': self._mem(),
                  'load': [round(x, 2) for x in os.getloadavg()], 'cores': len(cores),
                  'gpu': self._gpu()}
        with self._lock:
            self._hist.append(sample)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:                               # noqa: BLE001 采样失败不影响服务
                pass
            time.sleep(1.0)

    def current(self) -> dict | None:
        with self._lock:
            return dict(self._hist[-1]) if self._hist else None

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._hist)

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------ TensorBoard
class TensorBoardReader:
    """runs/ 下的标量曲线。

    痛点：训练一启动就新建 runs/<Game>_<ts>/ 并写一个 **88 字节的空事件头**，
    fit 之前没有任何标量。若按"最新目录"选，面板的 Loss 曲线会整块空白（实测踩过）。
    做法：候选按事件文件 mtime 从新到旧，逐个打开，取**第一个真的有标量**的 run；
    新 run 一旦开始出数据就自动切过去。
    """

    def __init__(self, ttl: float = 15.0, max_probe: int = 12, max_points: int = 1200):
        self._ttl = ttl
        self._max_probe = max_probe
        self._max_points = max_points
        self._lock = threading.Lock()
        self._accs: dict[str, object] = {}      # run_dir -> EventAccumulator（旧 run 只读一次）
        self._path = None
        self._loaded_at = 0.0
        self._cache: dict[str, list] = {}

    def _candidates(self) -> list[str]:
        """按事件文件 mtime 从新到旧；**过滤掉只有 88 字节表头的空 run**
        （训练一启动就会建这种目录，它们会把真正有数据的 run 挤出探测窗口）。"""
        base = CONFIG['runs_dir']
        out: list[tuple[float, str]] = []
        try:
            for name in os.listdir(base):
                d = os.path.join(base, name)
                if not os.path.isdir(d):
                    continue
                best_t, best_size = 0.0, 0
                for ev in os.listdir(d):
                    if ev.startswith('events.out.tfevents.'):
                        p = os.path.join(d, ev)
                        try:
                            st = os.stat(p)
                        except OSError:
                            continue
                        if st.st_mtime >= best_t:
                            best_t, best_size = st.st_mtime, st.st_size
                if best_t and best_size > 512:
                    out.append((best_t, d))
        except OSError:
            return []
        out.sort(reverse=True)
        return [d for _, d in out]

    def _load_from(self, run_dir: str, newest: bool) -> dict:
        """单个 run 的标量。旧 run 只 Reload 一次，只有最新 run 每次重读（它还在长）。"""
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = self._accs.get(run_dir)
        first = acc is None
        if first:
            acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
            self._accs[run_dir] = acc
        if first or newest:
            acc.Reload()
        tags = set(acc.Tags()['scalars'])
        if not tags:
            return {}
        out = {}
        for tag in TB_TAGS:
            if tag in tags:
                out[tag] = [(int(e.step), round(float(e.value), 6)) for e in acc.Scalars(tag)]
        return out

    def _downsample(self, pts: list[tuple[int, float]]) -> list[list]:
        """按步长抽稀到 max_points（保留最后一个点），避免把几十万个数塞给浏览器。"""
        if len(pts) <= self._max_points:
            return [[s, v] for s, v in pts]
        stride = max(1, len(pts) // self._max_points)
        thin = pts[::stride]
        if thin[-1][0] != pts[-1][0]:
            thin.append(pts[-1])
        return [[s, v] for s, v in thin]

    def series(self, force: bool = False) -> dict:
        """**跨 run 合并**：每重启一次训练就会新建一个 runs/ 目录，只看最新那个会让
        Loss 曲线每次重启都"从头开始"，看起来像学不动。这里按 step 合并所有有数据的 run
        （同 step 以更新的 run 为准），再抽稀到 max_points。"""
        with self._lock:
            if not force and time.time() - self._loaded_at < self._ttl:
                return self._cache
            self._loaded_at = time.time()
            cands = self._candidates()[:self._max_probe]
            merged: dict[str, dict[int, float]] = {}
            for i, run_dir in enumerate(cands):
                try:
                    data = self._load_from(run_dir, newest=(i == 0))
                except Exception:                        # noqa: BLE001 某个 run 坏了就跳过
                    continue
                for tag, pts in data.items():
                    bucket = merged.setdefault(tag, {})
                    for step, val in pts:
                        bucket[step] = val
            self._path = cands[0] if cands else None
            out = {}
            for tag, bucket in merged.items():
                if not bucket:
                    continue
                out[tag] = self._downsample(sorted(bucket.items()))
            self._cache = out
            return self._cache


# ------------------------------------------------------------------ 主采集器
def _parse_ts(text: str) -> float | None:
    """日志前缀 '2026-10-27 23:56:14,049' → epoch 秒"""
    try:
        from datetime import datetime
        return datetime.strptime(text.strip(), '%Y-%m-%d %H:%M:%S,%f').timestamp()
    except (ValueError, TypeError):
        return None


class Monitor:
    def __init__(self):
        self.hw = HardwareSampler()
        self.tb = TensorBoardReader()
        base = CONFIG['log_dir']
        self.logs = {k: os.path.join(base, k, 'latest.log')
                     for k in ('selfplay', 'fit', 'inference', 'train', 'crash')}
        self._infer_prev: tuple[float, float] | None = None   # (ts, total) 上一行，算瞬时吞吐
        self._infer_instant: float | None = None
        self._sims_cache: float | None = None                 # 模拟次数：fit/eval 阶段沿用自博弈的值

    # ---------- 各段解析 ----------
    def _rounds(self, limit: int = 400) -> list[dict]:
        rows: list[dict] = []
        try:
            with open(PERF_JSONL, encoding='utf-8') as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return rows[-limit:]

    def _live_infer(self) -> dict | None:
        """最近一条 [infer] 行 + **瞬时吞吐**。

        日志里的 req/s 是"自 reset 起的累计平均"，一轮越往后越被尾段拖低（99k→52k），
        直接显示会让人以为引擎在变慢。这里用相邻两条日志的 total 差 / 时间差算瞬时值，
        累计值另存 req_s_cum 备查。轮询比日志产出快时沿用上次算出的瞬时值。
        """
        for line in reversed(_tail(self.logs['inference'], 60_000)):
            m = _INFER_RE.search(line)
            if not m:
                continue
            d = {k: float(v) for k, v in m.groupdict().items()}
            ts_text = line[:23]
            d['ts'] = ts_text
            ts = _parse_ts(ts_text)
            prev_ts, prev_total = self._infer_prev if self._infer_prev else (None, None)
            if ts and prev_ts and prev_total is not None and d['total'] > prev_total:
                dt = ts - prev_ts
                if dt > 0.05:
                    self._infer_instant = (d['total'] - prev_total) / dt
            d['req_s_cum'] = d['req_s']
            if self._infer_instant is not None:
                d['req_s'] = self._infer_instant
            d['age_s'] = round(time.time() - ts, 1) if ts else None
            if ts:
                self._infer_prev = (ts, d['total'])
            return d
        return None

    def _phase(self) -> dict:
        sp_tail = _tail(self.logs['selfplay'], 60_000)
        sp_lines = [l for l in sp_tail if '[INFO]' in l]
        phase, progress = 'idle', None
        if sp_lines:
            last = sp_lines[-1]
            if 'Evaluation result' in last or re.search(r' VS \d+', last):
                phase = 'eval'
            elif 'Collecting data' in last:
                phase = 'fill'
            elif 'Starting selfplay' in last:
                phase = 'selfplay'
            else:
                phase = 'selfplay'
            # 当前轮已完成局数：从最近一条"Starting selfplay"/"Collecting data" 起数 Game end
            start_idx = 0
            for i in range(len(sp_tail) - 1, -1, -1):
                if 'Starting selfplay' in sp_tail[i] or 'Collecting data' in sp_tail[i]:
                    start_idx = i
                    break
            done = sum(1 for l in sp_tail[start_idx:] if _GAME_END_RE.search(l))
            try:
                total = int(CONFIG['selfplay_games_per_round'])
            except Exception:                            # noqa: BLE001
                total = None
            progress = {'done': done, 'total': total}
        # 日志静止判定：进程挂了/被 Ctrl-C 后，日志尾部还留着"上一次的样子"，
        # 面板若照旧显示 phase=selfplay + 36k req/s 会严重误导（10/27 就出现过）。
        ages = []
        for path in self.logs.values():
            try:
                ages.append(time.time() - os.path.getmtime(path))
            except OSError:
                continue
        newest_age = min(ages) if ages else None
        if newest_age is not None and newest_age > 30:
            return {'phase': 'stalled', 'progress': progress,
                    'last_game_ts': sp_lines[-1][:23] if sp_lines else None,
                    'age_s': round(newest_age, 1)}
        # fit 阶段：fit 日志比 selfplay 新 → 正在训练
        try:
            if os.path.getmtime(self.logs['fit']) > os.path.getmtime(self.logs['selfplay']) + 3:
                phase = 'fit'
        except OSError:
            pass
        return {'phase': phase, 'progress': progress,
                'last_game_ts': sp_lines[-1][:23] if sp_lines else None,
                'age_s': round(newest_age, 1) if newest_age is not None else None}

    def _live_fit(self) -> dict | None:
        """fit 阶段的行内状态。

        注意日志格式：一步是**两条行**——`Step N:` 之后跟一行
        ` loss=…, policy_loss=…, value_loss=…`（多行 message 被拆开写）。
        旧实现只看"最后 3 行"里有没有 loss，于是在 fit 刚结束/刚开始时永远拿不到 step。
        现在顺序扫一遍 tail，按记录顺序把 loss 挂到最近的 Step 上。
        """
        # fit 每步两行（约 110 字节/步 × 1171 步 ≈ 130KB/轮），窗口必须容得下一整轮，
        # 否则"上一轮末尾的 Val 行"会被挤出去（曾导致验证集卡片又变空）
        lines = [l for l in _tail(self.logs['fit'], 400_000) if '[INFO]' in l or l.lstrip().startswith('loss=')]
        out: dict = {}
        for line in lines:                                   # 顺序扫：后出现的覆盖先出现的
            m = _STEP_RE.search(line)
            if m:
                out['step'] = int(m.group(1))
                out.pop('loss', None)                        # 新的一步，旧的 loss 不再代表"当前"
                out.pop('policy_loss', None)
                out.pop('value_loss', None)
            lm = _LOSS_RE.search(line)
            if lm and 'step' in out:
                out['loss'] = float(lm.group(1))
                out['policy_loss'] = float(lm.group(2))
                out['value_loss'] = float(lm.group(3))
            vm = _VAL_RE.search(line)
            if vm:
                out['val'] = {'policy': float(vm.group(1)), 'value': float(vm.group(2))}
                if vm.group(3) is not None:                  # 日志带 KL / 目标熵
                    out['val']['kl'] = float(vm.group(3))
                    out['val']['entropy'] = float(vm.group(4))
            dm = _FIT_DONE_RE.search(line)
            if dm:
                out['fit_iteration'] = int(dm.group(1))
                out['fit_total_steps'] = int(dm.group(2))
                out['fit_seconds'] = float(dm.group(3))
        # 留出集指标优先读 sidecar（确定性全覆盖的权威值；日志行会被挤出窗口）
        side = self._val_sidecar()
        if side:
            out['val'] = {k: side.get(k) for k in ('policy', 'value', 'kl', 'entropy') if side.get(k) is not None}
            out['val']['n_samples'] = side.get('n_samples')
            out['val']['iteration'] = side.get('iteration')
            out['val']['age_s'] = side.get('age_s')
        return out or None

    def _fit_with_fallback(self) -> dict | None:
        fit = self._live_fit() or {}
        if not fit.get('val'):
            v = self._val_from_tb()
            if v:
                fit['val'] = v
        return fit or None

    def _val_sidecar(self) -> dict | None:
        path = os.path.join(CONFIG['log_dir'], 'fit', 'val_latest.json')
        try:
            with open(path, encoding='utf-8') as fp:
                data = json.load(fp)
            data['age_s'] = round(time.time() - float(data.get('ts') or 0), 1)
            return data
        except (OSError, ValueError):
            return None

    def _val_from_tb(self) -> dict | None:
        """本轮还没跑完 fit 时的兜底：取 TensorBoard 里**上一代**的留出集指标。
        没有它，每重启一次训练的第一次 fit 之前，验证集卡片就是"—"（看着像又坏了）。"""
        try:
            series = self.tb.series()
        except Exception:                                # noqa: BLE001 曲线读失败不影响其它卡
            return None
        pol = series.get('Val/policy') or []
        if not pol:
            return None
        out = {'policy': pol[-1][1], 'step': pol[-1][0], 'stale': True}
        for key, tag in (('value', 'Val/value'), ('kl', 'Val/policy_kl'),
                         ('entropy', 'Val/policy_entropy')):
            pts = series.get(tag) or []
            if pts:
                out[key] = pts[-1][1]
        return out

    def _live_eval(self) -> dict | None:
        for line in reversed(_tail(self.logs['selfplay'], 80_000)):
            m = _EVAL_RE.search(line)
            if m:
                return {'new': int(m.group(1)), 'best': int(m.group(2)),
                        'win': int(m.group(3)), 'lose': int(m.group(4)), 'draw': int(m.group(5)),
                        'games': int(m.group(6)), 'win_rate': float(m.group(7)) / 100}
        return None

    def _crashes(self) -> list[str]:
        return [l for l in _tail(self.logs['crash'], 40_000) if '未捕获' in l][-5:]

    def _running_profile(self, rt: dict | None) -> tuple[str, str, str]:
        """实际在跑的 (棋种, 档位)。优先心跳，其次训练进程的 env，最后面板自己的默认值。
        面板是独立进程、按默认档 m 启动的，直接读 settings['default_net'] 会把
        s 档（128x6）的训练显示成 m 档（256x15）——用户一眼就看出来了。"""
        game = (rt or {}).get('game') or CONFIG.get('game_name')
        profile = (rt or {}).get('profile')
        if not profile:
            try:
                from services.admin.runner import TrainingRunner
                env = TrainingRunner().status().get('env') or {}
                profile = env.get('profile') or env.get('ZZ_PROFILE')
                game = env.get('game') or game
            except Exception:                            # noqa: BLE001 面板独立性优先
                pass
        src = 'heartbeat' if (rt or {}).get('profile') else ('trainer' if profile else 'default')
        return game, (profile or get_profile()), src

    def _meta(self, rt: dict | None, game_name: str) -> dict:
        game, profile, src = self._running_profile(rt)
        try:
            net = net_config_for(profile, game)
        except Exception:                                # noqa: BLE001 档位名不认识就退回默认
            net, profile = CONFIG[game_name]['default_net'], get_profile()
        from core.utils.version import __version__, revision
        return {'version': __version__, 'revision': revision(),
                'game': game, 'profile': profile, 'profile_source': src,
                'device': CONFIG.get('device'), 'net': net,
                'games_per_round': CONFIG.get('selfplay_games_per_round'),
                'win_threshold': CONFIG.get('win_threshold'),
                'batch_leaves': CONFIG.get('mcts_batch_leaves')}

    def _heartbeat(self) -> dict | None:
        """训练进程自己写的心跳（logs/runtime.json）——面板的实时真源。"""
        try:
            from services.train import heartbeat
            return heartbeat.read()
        except Exception:                                # noqa: BLE001 训练模块缺失不该影响面板
            return None

    def _live_state(self, rt: dict | None) -> dict:
        """实时状态：心跳新鲜就用心跳，否则退回日志尾部启发式。"""
        if rt and (rt.get('age_s') or 1e9) < 12:
            gpm = None
            t0, done = rt.get('round_t0'), rt.get('games_done')
            if t0 and done:
                elapsed = max(time.time() - float(t0), 1e-9)
                gpm = round(done / elapsed * 60, 1)        # 轮内实时局/分（不再等轮末才算出来）
            if rt.get('sims_per_move'):
                self._sims_cache = float(rt['sims_per_move'])
            return {'phase': rt.get('phase') or 'unknown',
                    'gpm': gpm,
                    'sims_per_move': self._sims_cache,
                    'progress': {'done': rt.get('games_done'), 'total': rt.get('games_total')},
                    'buffer': {'size': rt.get('buffer_size'), 'capacity': rt.get('buffer_capacity'),
                               'val_size': rt.get('val_buffer_size'),
                               'val_capacity': rt.get('val_buffer_capacity')},
                    'age_s': rt.get('age_s'), 'source': 'heartbeat',
                    'iteration': rt.get('iteration'), 'round': rt.get('round')}
        base = self._phase()
        base['source'] = 'log'
        return base

    # ---------- 对外接口 ----------
    def snapshot(self) -> dict:
        rounds = self._rounds(200)
        rt = self._heartbeat()
        return {
            'ts': time.time(),
            'meta': self._meta(rt, game_name),
            'rounds': rounds,
            'live': {'infer': self._live_infer(), 'fit': self._fit_with_fallback(),
                     'eval': self._live_eval(), **(self._live_state(rt) or {})},
            'heartbeat': rt,
            'hardware': self.hw.current(),
            'hw_history': self.hw.history()[-120:],
            'crashes': self._crashes(),
        }

    def series(self) -> dict:
        rounds = self._rounds(400)
        keys = {
            'iteration': lambda r: r.get('iteration'),
            'games_per_min': lambda r: r.get('selfplay', {}).get('games_per_min'),
            'sims_per_s': lambda r: r.get('selfplay', {}).get('sims_per_s'),
            'samples_per_s': lambda r: r.get('selfplay', {}).get('samples_per_s'),
            'batch_mean': lambda r: r.get('selfplay', {}).get('infer', {}).get('batch_mean'),
            'req_per_s': lambda r: r.get('selfplay', {}).get('infer', {}).get('req_per_s'),
            'select_ms': lambda r: r.get('selfplay', {}).get('mcts', {}).get('select_ms_per_sim'),
            'wait_ms': lambda r: r.get('selfplay', {}).get('mcts', {}).get('wait_ms_per_sim'),
            'sims_per_move': lambda r: r.get('selfplay', {}).get('sims_per_move'),
            'fit_samples_per_s': lambda r: r.get('fit', {}).get('samples_per_s'),
            'eval_win_rate': lambda r: r.get('eval', {}).get('win_rate'),
            'round_wall_s': lambda r: r.get('round_wall_s'),
            'buffer_size': lambda r: r.get('buffer_size'),
        }
        series = {}
        # 横轴必须跨重启唯一：`round` 是"本次启动内第几轮"，每次重启都从 1 重数，
        # 直接拿来当 x 会把多轮叠在同一个位置（图上出现竖直多条线的"同一位置多个值"）。
        # 用 perf.jsonl 的累计序号：天然唯一且按时间单调。
        xs = list(range(1, len(rounds) + 1))
        for name, getter in keys.items():
            series[name] = [[x, getter(r)] for x, r in zip(xs, rounds) if getter(r) is not None]
        return {'rounds': series, 'tb': self.tb.series(),
                'hw': [{'t': s['t'], 'cpu': s['cpu'], 'mem': s['mem']['pct'],
                        'gpu': (s['gpu'] or {}).get('util')} for s in self.hw.history()]}

    def stop(self) -> None:
        self.hw.stop()
