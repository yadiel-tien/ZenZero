"""自适应 K 控制器：近窗口径 + 两轮迟滞降档 + 即时升档。

2026-11-09 重写口径（m8 一夜复盘）：
- 判据用 batch_recent_mean（近窗），不再读整轮累计 batch_mean——m8 常态累计批
  1300+ >900，旧版升档条件永不成立，K 变成只降不升的棘轮；
- 降档要连续两轮近窗超线（迟滞），单轮抢卡毛刺不降档；
- 升档即时（批没吃饱且引擎快）。

运行：.venv/bin/python tests/test_auto_tune_k.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ZZ_DEVICE', 'cpu')
import logging  # noqa: E402

from core.utils.config import CONFIG  # noqa: E402
from services.train.selfplay import SelfPlayManager as SPM  # noqa: E402


class _Stub:
    logger = logging.getLogger('tuner-test')


def S(batch_recent, fwd, batch_mean=None):
    """构造 infer_stats() 形状；batch_mean 故意可给"错误口径"的值验证没人再读它。"""
    return {'batch_recent_mean': batch_recent, 'fwd_ms_mean': fwd,
            'batch_mean': batch_mean if batch_mean is not None else batch_recent}


def run(label, k0, seq):
    """seq: [(stats, 期望K)]，同一 Stub 连续跑（迟滞计数跨轮延续）。"""
    CONFIG['mcts_batch_leaves'] = k0
    st = _Stub()
    for i, (stats, want) in enumerate(seq):
        SPM._auto_tune_k(st, stats)
        got = CONFIG['mcts_batch_leaves']
        if got != want:
            print(f'FAIL {label} 第{i + 1}轮: 期望 K={want} 实得 {got}')
            return 1
    print(f'PASS {label}')
    return 0


def main():
    bad = 0
    # —— 基本盘（对齐 2026-11-04 A/B 结论）——
    bad += run('稳态·批大前向快：保持', 64, [(S(1700, 8.0), 64)])
    bad += run('持续慢两轮：收缩一半（迟滞后生效）', 64,
               [(S(1500, 22.0), 64), (S(1500, 22.0), 32)])
    bad += run('近窗批顶满两轮：收缩', 64, [(S(3010, 9.0), 64), (S(3010, 9.0), 32)])
    bad += run('批太小·前向有余：立即放大', 8, [(S(300, 3.0), 16)])
    bad += run('批太小但前向已慢：不动', 8, [(S(700, 18.0), 8)])
    bad += run('收缩有下限', 16, [(S(2600, 30.0), 16), (S(2600, 30.0), 8)])
    bad += run('放大有上限', 96, [(S(500, 5.0), 128)])
    bad += run('无数据：不动', 32, [(S(0, 0), 32)])
    # —— 2026-11-08/09 m8 事故回归 ——
    bad += run('单轮抢卡毛刺（15.2ms 擦线）不再降档', 64, [(S(1273, 15.19), 64)])
    bad += run('毛刺后恢复：计数器清零，再来单轮也不降', 64,
               [(S(1273, 22.0), 64), (S(1273, 2.0), 64), (S(1273, 22.0), 64)])
    bad += run('误降后一轮即回：累计批 1300 不再挡升档（近窗小+fwd快）', 32,
               [(S(52, 2.6, batch_mean=1303), 64)])
    bad += run('完整剧本：干扰两轮降→撤→次轮升回', 64,
               [(S(1300, 18.0), 64), (S(1300, 16.0), 32),      # ComfyUI 抢卡两轮
                (S(80, 2.0), 64)])                              # 干扰撤，近窗立回
    print('\nALL AUTO-TUNE TESTS PASSED' if not bad else f'{bad} FAILED')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
