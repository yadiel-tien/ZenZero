"""versus 记账映射回归测试：outcome(原始元组胜者下标) 必须正确落到 Elo 与返回值。
2026-11-05 事故防复发：v2 重写时 0/1 分支抄反，全榜胜负倒置、Elo 发给输家。"""
import shutil
import sys
sys.path.insert(0, '.')
from core.utils import arena as A
from core.utils.elo import Elo


class FakeServer:
    def __init__(self, env_name, model_id, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEnv:
    outcome = 0

    def random_order_play(self, players, silent=False):
        return FakeEnv.outcome


A.AIServer = FakeServer
A.get_class = lambda name: FakeEnv


def make_arena():
    ar = A.Arena('ChineseChess', 'maptest')
    ar.rates = {'m/1': Elo('m/1'), 'mini/2': Elo('mini/2')}
    ar.save = lambda: None          # 不落盘
    return ar


def run_case(outcome, expect_ret, expect_hi):
    ar = make_arena()
    FakeEnv.outcome = outcome
    r = ar.versus('m/1', 'mini/2', n_sim=1)
    assert r == expect_ret, f'outcome={outcome} 返回 {r} 应为 {expect_ret}'
    s1, s2 = ar.rates['m/1'].scores, ar.rates['mini/2'].scores
    hi, lo = (s1, s2) if expect_hi == 'k1' else (s2, s1)
    assert hi > 1500 > lo, f'outcome={outcome} Elo 方向错误: m/1={s1:.1f} mini/2={s2:.1f}'
    print(f'outcome={outcome} -> {expect_ret}, Elo 流向 {expect_hi} ✓')


run_case(0, 'w', 'k1')       # p1(m/1) 胜：返回 w，m/1 加分
run_case(1, 'l', 'k2')       # p2(mini/2) 胜：返回 l，mini/2 加分
FakeEnv.outcome = -1
ar = make_arena()
assert ar.versus('m/1', 'mini/2', n_sim=1) == 'd'
assert ar.rates['m/1'].scores == 1500.0, '和棋不应改变分数'
print('outcome=-1 -> d, 分数不动 ✓')
try:
    shutil.rmtree(f'data/ChineseChess/rates/maptest', ignore_errors=True)
except Exception:
    pass
print('ALL 3 RESULT-MAPPING TESTS PASSED')
