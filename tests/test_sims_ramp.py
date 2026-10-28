"""每手模拟次数（sims/move）的爬坡语义。

背景：旧实现按"本次启动内第几轮 ÷ max_iters(500)"爬坡，等于**每次重启都退回 200**，
线上 perf 行永远是 @200sims/手，爬坡从未生效。而且实测 200 sims 的目标太噪：
KL(pi200‖pi800)=0.36，占掉了 policy 交叉熵可学空间（0.57）的 64%。
现在改成按**绝对代**从 400 爬到 800（50 代到位），重启会续上。

运行：.venv/bin/python tests/test_sims_ramp.py
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils.config import CONFIG, settings  # noqa: E402
from services.train.selfplay import ramp_simulations  # noqa: E402


def test_config_has_sims_block():
    for game in ('ChineseChess', 'Gomoku'):
        cfg = CONFIG[game].get('sims')
        assert cfg and cfg['start'] >= 200, (game, cfg)
        assert cfg['max'] >= cfg['start'] and cfg['ramp_generations'] >= 1, (game, cfg)
    print(f"   ok   两个棋种都有 sims 配置：{CONFIG['ChineseChess']['sims']}")


def test_ramp_is_absolute_not_per_launch():
    """同一个绝对代 → 同一个值（重启不会重置）；且随代单调不减、到点封顶"""
    sims = {'start': 400, 'max': 800, 'ramp_generations': 50}
    vals = [ramp_simulations(it, sims) for it in range(0, 61)]
    assert vals[0] == 400, vals[:3]
    assert vals == sorted(vals), '爬坡必须单调不减'
    assert vals[50] == 800 and vals[60] == 800, (vals[49], vals[50], vals[60])
    # 重启续上：第 30 代永远是 640，不会因为"本次启动第 1 轮"退回 400
    assert ramp_simulations(30, sims) == ramp_simulations(30, sims) == 640, ramp_simulations(30, sims)
    assert ramp_simulations(1, sims) > 400, '第 1 代就该略高于 start，不能等于 200 那套'
    print(f'   ok   绝对代爬坡：0→{vals[0]} 10→{vals[10]} 25→{vals[25]} 30→{vals[30]} 50→{vals[50]}')


def test_ramp_edges():
    assert ramp_simulations(-5, {'start': 400, 'max': 800, 'ramp_generations': 50}) == 400
    assert ramp_simulations(10 ** 6, {'start': 400, 'max': 800, 'ramp_generations': 50}) == 800
    assert ramp_simulations(3, {'start': 800, 'max': 800, 'ramp_generations': 50}) == 800, 'start==max 时恒定'
    # 缺省读当前棋种配置
    assert 400 <= ramp_simulations(0) <= 800
    print('   ok   边界：负代/超大代/start==max/缺省配置 都稳')


def main() -> None:
    for fn in (test_config_has_sims_block, test_ramp_is_absolute_not_per_launch, test_ramp_edges):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 3 SIMS-RAMP TESTS PASSED')


if __name__ == '__main__':
    main()
