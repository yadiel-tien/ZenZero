"""自博弈认输门槛的回归测试（状态机 + 配置/env 覆盖）。

要点：
- 只统计"该方自己行棋轮次"的连败计数——双方交替，对手的好局不应清掉本方的濒死计数；
- 回升即清零（防中盘一次性误判）；min_steps 前不计数（开局评估未稳）；
- threshold<=0 全关；守擂评估不走本函数（代码路径隔离，此处只测状态机）。

运行：.venv/bin/python tests/test_resign_threshold.py
"""
import os
import sys

os.environ['ZZ_DEVICE'] = 'cpu'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CFG = {'threshold': 0.95, 'consecutive': 6, 'min_steps': 20}


def _feed(seq, cfg=CFG):
    """seq: [(side, value_now, steps)] → 返回触发的第几手（1-based），未触发返回 None"""
    from services.train.selfplay import resign_update
    streak = {0: 0, 1: 0}
    for i, (side, v, steps) in enumerate(seq, 1):
        if resign_update(streak, side, v, cfg, steps):
            return i
    return None


def test_triggers_after_consecutive_own_losses():
    # 方0 连输 6 个自己轮次（12 手里第 12 手触发）
    seq = [(i % 2, -0.99 if i % 2 == 0 else 0.99, 30 + i) for i in range(12)]
    assert _feed(seq) == 11   # 方0 的第 6 个轮次 = 全程第 11 手


def test_recovery_resets_streak():
    seq = [(i % 2, -0.99 if i % 2 == 0 else 0.99, 30 + i) for i in range(10)]
    seq += [(0, -0.5, 41)]  # 方0 回升 → 清零
    seq += [(i % 2, -0.99 if i % 2 == 0 else 0.99, 42 + i) for i in range(10)]
    assert _feed(seq) is None   # 清零后 5 连不足 6，不触发


def test_min_steps_guard():
    # 20 步之前必败也不数
    seq = [(i % 2, -0.99 if i % 2 == 0 else 0.99, i) for i in range(20)]
    assert _feed(seq) is None


def test_disabled_when_threshold_zero():
    cfg = dict(CFG, threshold=0.0)
    seq = [(0, -1.0, 100)] * 50
    assert _feed(seq, cfg) is None


def test_near_threshold_does_not_count():
    seq = [(0, -0.94, 100)] * 50   # 差 0.01 不算必败
    assert _feed(seq) is None


def test_env_override_in_config():
    """ZZ_RESIGN=off / 数值 的覆盖逻辑（子进程干净求值 config 模块）"""
    import subprocess
    code = ("import os; os.environ['ZZ_RESIGN']=os.environ['VAL'];"
            "import core.utils.config as c;"
            "print(c.CONFIG['ChineseChess']['selfplay']['resign']['threshold'])")
    for val, want in [('off', 0.0), ('0.9', 0.9)]:
        env = dict(os.environ, VAL=val, PYTHONPATH='.')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
        assert abs(float(out.stdout.strip().splitlines()[-1]) - want) < 1e-9, (val, out.stdout)


def main():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, f in fns:
        print(f'== {name}')
        f()
        print('   ok')
    print(f'\nALL {len(fns)} RESIGN-THRESHOLD TESTS PASSED')


if __name__ == '__main__':
    main()
