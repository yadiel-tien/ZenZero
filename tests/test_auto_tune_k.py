"""自适应 K 控制器：换大网络/弱设备时应收缩，换小网络/空闲 GPU 时应放大。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ZZ_DEVICE', 'cpu')
from services.train.selfplay import SelfPlayManager
from core.utils.config import CONFIG

import logging
class _Stub:
    logger = logging.getLogger('tuner-test')
cls = SelfPlayManager
tune = lambda st: cls._auto_tune_k(_Stub(), st)   # 只需 logger，不依赖真实训练状态

def with_k(k, stats):
    CONFIG['mcts_batch_leaves'] = k
    tune(stats)
    return CONFIG['mcts_batch_leaves']

cases = [
    # (说明, 起始K, 引擎实测, 期望K)
    ('小网络·批大前向快：保持', 64, {'batch_mean': 1700, 'fwd_ms_mean': 8.0}, 64),
    ('大网络·前向慢：收缩一半', 64, {'batch_mean': 1500, 'fwd_ms_mean': 22.0}, 32),
    ('批过大：收缩一半', 64, {'batch_mean': 3010, 'fwd_ms_mean': 9.0}, 32),
    ('批太小·前向有余：放大一旦', 8, {'batch_mean': 300, 'fwd_ms_mean': 3.0}, 16),
    ('批太小但前向已慢：不动', 8, {'batch_mean': 700, 'fwd_ms_mean': 18.0}, 8),
    ('收缩有下限', 16, {'batch_mean': 2600, 'fwd_ms_mean': 30.0}, 8),
    ('放大有上限', 96, {'batch_mean': 500, 'fwd_ms_mean': 5.0}, 128),
    ('无数据：不动', 32, {'batch_mean': 0, 'fwd_ms_mean': 0}, 32),
]
bad = 0
for label, k, stats, want in cases:
    got = with_k(k, stats)
    ok = got == want
    print(('PASS' if ok else 'FAIL'), f'{label}: K {k} -> {got}（期望 {want}）')
    bad += 0 if ok else 1
print('ALL AUTO-TUNE TESTS PASSED' if not bad else f'{bad} FAILED')
sys.exit(1 if bad else 0)
