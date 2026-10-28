"""微型端到端自检：虚拟损失批量 MCTS + 新收批 + 分组唤醒 是否真的坐实。

比 tests/drills/train_cold_start_drill.py 更小更快（1 轮、2 局、32 sims/手），
目的是在改完引擎/MCTS 之后半分钟内确认：请求真的成批（2 局也能凑出 16 的批）、
棋真的下完、数据真的入池、fit/eval/checkpoint/[perf] 整条链路不断。

运行：.venv/bin/python tests/drills/selfplay_batch_smoke.py
"""
import logging
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'
os.environ['ZZ_GAME'] = 'Gomoku'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np  # noqa: E402

logging.disable(logging.CRITICAL)

from core.utils.config import CONFIG, settings  # noqa: E402

tmp = tempfile.mkdtemp(prefix='zz_smoke_')
backup_dirs = (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'])
CONFIG['data_dir'] = tmp
CONFIG['runs_dir'] = os.path.join(tmp, 'runs')
CONFIG['log_dir'] = os.path.join(tmp, 'logs')   # [perf] 写临时目录，别污染生产 logs/perf
os.makedirs(os.path.join(tmp, 'Gomoku'), exist_ok=True)

settings['default_net'].clear()
settings['default_net'].update({
    'arch': 'resnet_v1', 'in_channels': 2, 'n_filters': 8, 'n_res_blocks': 1,
    'n_cells': 225, 'n_actions': 225, 'use_se': False, 'n_policy_filters': 2,
    'n_value_filters': 1, 'n_value_hidden_channels': 16, 'feature_version': 1})

backup = (settings['max_iters'], settings['buffer_size'])
srv = mgr = None
try:
    settings['max_iters'] = 1
    # 仓容 2100 → 填仓阈值 = 一个 batch = 2048；预填 2100 条跳过 fill-loop
    settings['buffer_size'] = 2100
    from services.inference.train_server import TrainServer
    from services.train.selfplay import SelfPlayManager
    import services.train.selfplay as sp_mod
    import core.mcts.deepMcts as dm

    # 32 sims/手：只验证链路与批量，不追求棋力
    sp_mod.ramp_simulations = lambda iteration, sims=None: 32
    # 统计"驱动方式"是否走了流水线（run_pipelined 才会填 stat_select_s/stat_wait_s）
    pipelined_calls = [0]
    _orig_pipelined = dm.NeuronMCTS.run_pipelined

    def _rec_pipelined(self, *a, **kw):
        pipelined_calls[0] += 1
        return _orig_pipelined(self, *a, **kw)
    dm.NeuronMCTS.run_pipelined = _rec_pipelined

    srv = TrainServer(-1, 'Gomoku')
    srv.start()
    mgr = SelfPlayManager(n_workers=4, train_server=srv)
    rng = np.random.default_rng(0)
    for _ in range(2100):
        mgr.buffer.add((rng.random((2, 15, 15)).astype('float32') / 4,
                        np.ones(225, dtype=np.float32) / 225, 0.0))

    # 守擂评估 100 局 → 2 局：真双引擎对局链路一步不少
    _orig_eval = mgr.evaluation
    mgr.evaluation = lambda it, n_games=100: _orig_eval(it, n_games=2)

    t0 = time.time()
    mgr.run(n_games=2)
    wall = time.time() - t0
    stats = srv.infer_stats()
    print(f'[smoke] 完整一轮 {wall:.1f}s | buffer={mgr.buffer.size} '
          f'| engine {stats["req_per_s"]:.0f} req/s 均批 {stats["batch_mean"]} '
          f'(p90 {stats["batch_p90"]}, max {stats["batch_max"]}) fwd {stats["fwd_ms_mean"]}ms')

    assert stats['batch_max'] > 1, f'引擎没成批收请求: {stats}'
    assert pipelined_calls[0] > 0, '自博弈没走流水线驱动'
    mcts_stat = mgr._last_sp.get('mcts') or {}
    assert mcts_stat.get('select_ms_per_sim') is not None, f'[perf] 缺少选叶子/等引擎拆分: {mcts_stat}'
    # 在飞请求 = 局数 × K = 16，批大小不再等于并发局数（旧实现只会是 2）
    # 流水线是"完成一条补一条"，CPU 引擎单批很慢时批会贴近在飞数/2，故阈值取 > 局数
    assert stats['batch_mean'] > 2, f'2 局下批仍被局数锁死: {stats["batch_mean"]}'
    assert stats['batch_max'] >= 8, f'批上限没吃到 K: {stats}'
    vs = mgr.buffer.value_buffer[:mgr.buffer.size]
    assert np.all(np.abs(vs) <= 1.0 + 1e-5), f'样本 v 越界: {vs[:5]}'
    assert os.path.exists(os.path.join(tmp, 'Gomoku', 'checkpoint_1.pt')), 'checkpoint 未落盘'
    assert os.path.exists(os.path.join(tmp, 'logs', '_state', 'perf.jsonl')), '[perf] 未落盘'
    with open(os.path.join(tmp, 'logs', '_state', 'perf.jsonl'), encoding='utf-8') as fp:
        print('[smoke] perf: ' + fp.read().strip())
    print('[smoke] ALL OK ✓')
finally:
    settings['max_iters'], settings['buffer_size'] = backup
    for obj in (mgr, srv):
        try:
            if obj:
                obj.shutdown()
        except Exception:
            pass
    CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'] = backup_dirs
    shutil.rmtree(tmp, ignore_errors=True)
