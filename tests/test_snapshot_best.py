"""监控快照的「最佳」兜底：面板任何时候都得答得出"王是谁"。

背景：心跳只有自博弈的写点带 `best_index`，fit/eval 以及"心跳超时退回日志启发式"
都没有这个字段——任务卡于是显示「最佳 —」（修复前更糟：显示「尚无守擂者」）。
`Monitor._fill_best` 统一从磁盘补，本测试钉死四条路径：

  ① 心跳缺字段 + 无 best_index.pkl → 退回最新 checkpoint（read_best_index 的既有语义）
  ② 心跳缺字段 + 有 best_index.pkl → 用 best_index.pkl
  ③ 心跳写着 -1（旧进程/冷启动残留）→ 磁盘真值优先，不被 -1 盖住
  ④ 心跳自带 best_index（自博弈/fit 新写点）→ 原样透传，不查盘

运行：.venv/bin/python tests/test_snapshot_best.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils.config import CONFIG, net_dir  # noqa: E402
from core.network.functions import save_best_index  # noqa: E402

GAME, PROFILE = CONFIG['game_name'], (CONFIG.get('profile') or 'm')


def _setup():
    tmp = tempfile.mkdtemp(prefix='zz_snap_best_')
    backup = (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'])
    CONFIG['data_dir'] = os.path.join(tmp, 'data')
    CONFIG['runs_dir'] = os.path.join(tmp, 'runs')
    CONFIG['log_dir'] = os.path.join(tmp, 'logs')
    os.makedirs(os.path.join(CONFIG['log_dir'], '_state'), exist_ok=True)
    os.makedirs(CONFIG['runs_dir'], exist_ok=True)
    d = net_dir(GAME, PROFILE)
    os.makedirs(d, exist_ok=True)
    for it in (61, 62, 63):
        open(os.path.join(d, f'checkpoint_{it}.pt'), 'wb').write(b'x')
    return tmp, backup


def _teardown(tmp, backup):
    CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'] = backup
    shutil.rmtree(tmp, ignore_errors=True)


def _heartbeat(age_s: float = 0.0, **fields):
    data = {'ts': time.time() - age_s, 'game': GAME, 'profile': PROFILE,
            'phase': 'fit', 'iteration': 64, 'games_done': 21, 'games_total': 400}
    data.update(fields)
    path = os.path.join(CONFIG['log_dir'], '_state', 'runtime.json')
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)


def _selfplay_log(line: str) -> None:
    d = os.path.join(CONFIG['log_dir'], 'selfplay')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'latest.log'), 'w', encoding='utf-8') as fp:
        fp.write(line + '\n')


def _perf(rows) -> None:
    """写一份 perf.jsonl 并让采集层指向它（模块常量在 import 时就固化路径了）。"""
    d = os.path.join(CONFIG['log_dir'], '_state')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'perf.jsonl')
    with open(path, 'w', encoding='utf-8') as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + '\n')
    import services.monitor.collect as collect
    collect.PERF_JSONL = path
    collect.PERF_JSONL_OLD = os.path.join(d, 'perf_old.jsonl')


def _live():
    from services.monitor.collect import Monitor
    return Monitor().snapshot()['live']


def test_fallback_to_latest_checkpoint():
    tmp, backup = _setup()
    try:
        _heartbeat()                                  # fit 阶段心跳没有 best_index
        assert _live().get('best_index') == 63, '无 best_index.pkl 时应退回最新 checkpoint'
        print('   ok  心跳缺字段 + 无 best_index.pkl → 最新 checkpoint 63')
    finally:
        _teardown(tmp, backup)


def test_fallback_uses_best_pkl():
    tmp, backup = _setup()
    try:
        save_best_index(62, GAME, PROFILE)
        _heartbeat()
        assert _live().get('best_index') == 62, '有 best_index.pkl 时必须用它，而不是最新 checkpoint'
        print('   ok  心跳缺字段 + best_index.pkl=62 → 62')
    finally:
        _teardown(tmp, backup)


def test_stale_minus_one_overridden():
    tmp, backup = _setup()
    try:
        save_best_index(62, GAME, PROFILE)
        _heartbeat(best_index=-1)                     # 旧进程写下的冷启动残留
        assert _live().get('best_index') == 62, '-1 是"不知道"，不能让面板照抄'
        print('   ok  心跳写 -1 → 磁盘真值 62 覆盖')
    finally:
        _teardown(tmp, backup)


def test_heartbeat_value_wins():
    tmp, backup = _setup()
    try:
        save_best_index(62, GAME, PROFILE)
        _heartbeat(best_index=61)                     # 训练进程是最新真源
        assert _live().get('best_index') == 61, '心跳自带的 best_index 优先，不查盘'
        print('   ok  心跳自带 best_index=61 → 原样透传')
    finally:
        _teardown(tmp, backup)


def test_log_fallback_iteration():
    """心跳过期/缺失走日志启发式时，代数与王者也要有值（否则卡片渲染成「第 0 代」）。"""
    tmp, backup = _setup()
    try:
        _selfplay_log('2026-11-08 22:10:00 [INFO] [T] Starting selfplay, iteration: 65, '
                      'best index: 64, round 1/5 of this run.')
        live = _live()
        assert live.get('iteration') == 65, live.get('iteration')
        assert live.get('best_index') == 64, live.get('best_index')
        print('   ok  无心跳 + 日志含 iteration/best → 65 / 64')
    finally:
        _teardown(tmp, backup)


def test_stale_heartbeat_keeps_iteration():
    """心跳超过 12s 只代表"数字旧了"，不代表这一轮的代次/王者作废。"""
    tmp, backup = _setup()
    try:
        _heartbeat(age_s=120.0)                       # fit 阶段只在开头写一次心跳
        live = _live()
        assert live.get('iteration') == 64, live.get('iteration')
        assert live.get('best_index') == 63, '心跳没带王者 → 磁盘兜底'
        print('   ok  心跳过期 120s → 代数沿用 64，王者磁盘兜底 63')
    finally:
        _teardown(tmp, backup)


def test_last_iteration_from_perf():
    """「上次训练到哪代」：本轮代数读不到时靠它，别拿 0/1 代糊弄。"""
    tmp, backup = _setup()
    try:
        _perf([{'ts': '2026-11-08T21:00:00', 'profile': PROFILE, 'iteration': 64},
               {'ts': '2026-11-08T21:40:00', 'profile': PROFILE, 'iteration': 65}])
        _heartbeat()
        live = _live()
        assert live.get('last_iteration') == 65, live.get('last_iteration')
        print('   ok  perf 最近一轮 65 → last_iteration=65')
    finally:
        _teardown(tmp, backup)


def test_last_iteration_falls_back_to_checkpoint():
    tmp, backup = _setup()
    try:
        _heartbeat()                                  # 没有 perf 记录
        assert _live().get('last_iteration') == 63, '没有 perf 时应退回最新 checkpoint'
        print('   ok  无 perf → 最新 checkpoint 63')
    finally:
        _teardown(tmp, backup)


def test_last_iteration_ignores_other_profile():
    tmp, backup = _setup()
    try:
        _perf([{'ts': '2026-11-08T21:40:00', 'profile': 'other', 'iteration': 99}])
        _heartbeat()
        assert _live().get('last_iteration') == 63, '别的档位的 perf 行不能算进来'
        print('   ok  其它档位 perf 行被忽略 → 63')
    finally:
        _teardown(tmp, backup)


def main() -> None:
    for fn in (test_fallback_to_latest_checkpoint, test_fallback_uses_best_pkl,
               test_stale_minus_one_overridden, test_heartbeat_value_wins,
               test_log_fallback_iteration, test_stale_heartbeat_keeps_iteration,
               test_last_iteration_from_perf, test_last_iteration_falls_back_to_checkpoint,
               test_last_iteration_ignores_other_profile):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 9 SNAPSHOT-BEST TESTS PASSED')


if __name__ == '__main__':
    main()
