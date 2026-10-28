"""训练线端到端微型演练：冷启动晋升(F2) → 守擂评估 → 多轮循环(F1) 的闭环验证。

用微型网络（8 filters × 1 block 五子棋）+ 临时数据目录，几分钟内跑完真实
SelfPlayManager.run() 两轮完整流程。只验证"链路通与断言成立"，不代表棋力。

运行：.venv/bin/python tests/drills/train_cold_start_drill.py   （建议后台）
"""
import logging
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'
os.environ['ZZ_GAME'] = 'Gomoku'   # 本演练是五子棋专用（buffer/网络形状都按 2x15x15）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np  # noqa: E402

logging.disable(logging.CRITICAL)

from core.utils.config import CONFIG, settings  # noqa: E402

tmp = tempfile.mkdtemp(prefix='zz_drill_')
CONFIG['data_dir'] = tmp
CONFIG['runs_dir'] = os.path.join(tmp, 'runs')   # 隔离：不碰仓库 runs/
os.makedirs(os.path.join(tmp, 'Gomoku'), exist_ok=True)

# 微型网（就地 clear/update，保持所有按引用取默认参数的绑定）
settings['default_net'].clear()
settings['default_net'].update({
    'arch': 'resnet_v1', 'in_channels': 2, 'n_filters': 8, 'n_res_blocks': 1,
    'n_cells': 225, 'n_actions': 225, 'use_se': False, 'n_policy_filters': 2,
    'n_value_filters': 1, 'n_value_hidden_channels': 16, 'feature_version': 1})

backup = (settings['max_iters'], settings['buffer_size'])
srv = mgr = None
try:
    settings['max_iters'] = 2      # 本次启动轮数（F1 新语义）
    settings['buffer_size'] = 1024
    from services.inference.train_server import TrainServer
    from services.train.selfplay import SelfPlayManager
    from core.network.functions import read_best_index, read_latest_index

    t0 = time.time()
    srv = TrainServer(-1, 'Gomoku')
    srv.start()
    mgr = SelfPlayManager(n_workers=16, train_server=srv)
    assert mgr.best_index == -1, 'drill setup: 应为无 best 的冷启动'

    # 提速两处（CPU 演练预算，不改被测链路本身）：
    # ① 预填仓越过 fill-loop 阈值（小仓配置下阈值=一个batch=2048 条），省掉整轮填仓局
    rng = np.random.default_rng(0)
    for _ in range(2100):
        pi = np.ones(225, dtype=np.float32) / 225
        mgr.buffer.add((rng.random((2, 15, 15)).astype('float32') / 4, pi, 0.0))
    # ② 守擂评估 100 局 → 6 局：真两引擎对局链路一步不少，只是快
    _orig_eval = mgr.evaluation
    mgr.evaluation = lambda it, n_games=100: _orig_eval(it, n_games=6)

    print('[drill] 开始两轮 run() ...', flush=True)
    mgr.run(n_games=6)
    wall = time.time() - t0

    gd = os.path.join(tmp, 'Gomoku')
    best, latest = read_best_index('Gomoku'), read_latest_index('Gomoku')
    files = sorted(os.listdir(gd))
    print(f'[drill] run 完成 {wall:.0f}s | best={best} latest={latest}')
    print(f'[drill] 落盘: {files}')

    # F1：两轮都真实执行（旧绝对上限语义下 resume 场景一轮不跑）
    assert os.path.exists(os.path.join(gd, 'checkpoint_1.pt')), '第 1 轮 ckpt 缺失'
    assert os.path.exists(os.path.join(gd, 'checkpoint_2.pt')), '第 2 轮 ckpt 缺失（F1 失效？）'
    # F2：冷启动第一轮必晋升（无守擂者直接晋升，不靠掷硬币）
    assert best in (1, 2), f'F2 失效：冷启动未晋升, best={best}'
    # 主引擎已经过晋升换血（不再是 model_index=-1 的随机身）
    assert srv.model_index in (1, 2), f'主引擎未换血: {srv.model_index}'
    # buffer/val 双池都落了盘
    assert os.path.exists(os.path.join(gd, 'buffer.pkl'))
    assert os.path.exists(os.path.join(gd, 'val_buffer.pkl'))
    print(f'[drill] buffer={mgr.buffer.size} 条, val={mgr.val_buffer.size} 条')
    print('[drill] ALL ASSERTIONS PASSED ✓')
finally:
    settings['max_iters'], settings['buffer_size'] = backup
    for obj in (mgr, srv):
        try:
            if obj:
                obj.shutdown()
        except Exception:
            pass
    CONFIG['data_dir'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    shutil.rmtree(tmp, ignore_errors=True)
