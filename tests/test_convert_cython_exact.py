"""convert_v2（Cython）与 _convert_python 的逐位对拍。

为什么必须逐位：棋盘表示错了不会崩，只会让模型悄悄吃错输入、棋力缓慢劣化——
比崩溃毒得多。所以这条测试覆盖：

1. 真实对局轨迹的**每一步、双视角**（自动覆盖 from/to 差分面、吃子后的兵力面、
   被将面、重复窗口面、未吃子时钟面）；
2. 手工构造的边界局面：开局（层0==层1，"无历史"）、往返重复、被将、无老将、
   空盘、满编；
3. 随机拼装的棋盘（各种层历史），专打分支覆盖；
4. **并发一致性**：多线程同时调用必须与单线程结果完全一致（模块级禁止有可变状态，
   这条是防"图省事用全局缓冲区"的那类 bug——第一版真踩过）；
5. 派发正确：`convert_to_network(version=2)` 必须走 C；v1 仍走 Python 且形状正确。

运行：.venv/bin/python tests/test_convert_cython_exact.py
"""
import os
import sys
import threading

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from core.env import chess as chess_mod  # noqa: E402
from core.env.chess import ChineseChess, _cy_convert_v2  # noqa: E402

CY = _cy_convert_v2
REQUIRED = CY is not None


def _py(state, player):
    return ChineseChess._convert_python(state, player, 2)


def _assert_equal(state, player, where):
    c = CY(state, player)
    p = _py(state, player)
    assert c.shape == p.shape == (26, 10, 9), (where, c.shape, p.shape)
    assert c.dtype == p.dtype == np.float32, (where, c.dtype, p.dtype)
    if not np.array_equal(c, p):
        diff = np.argwhere(c != p)
        raise AssertionError(
            f'{where} player={player}: C 与 Python 不等，{len(diff)} 处差异，'
            f'首个 {tuple(diff[0])} C={c[tuple(diff[0])]} py={p[tuple(diff[0])]}')


def test_random_games_every_position():
    """真实对局：每走一步、双视角都比一遍"""
    n_games, n_positions = 24, 0
    for g in range(n_games):
        env = ChineseChess()
        rng = np.random.default_rng(1000 + g)
        for _ in range(60):
            for player in (0, 1):
                _assert_equal(env.state, player, f'game{g}/steps{env.steps}')
                n_positions += 1
            if env.terminated:
                break
            actions = env.get_valid_actions(env.state, env.player_to_move)
            env.step(int(rng.choice(actions)))
    print(f'   ok  {n_games} 局随机对局 × 每步双视角 = {n_positions} 次逐位相等')


def test_edge_positions():
    """边界：开局无历史、往返重复、被将、缺老将、空盘"""
    cases = []

    def _board(pieces):
        st = np.zeros((7, 10, 9), dtype=np.int32)
        layer = np.full((10, 9), -1, dtype=np.int32)
        for (r, c), pid in pieces.items():
            layer[r, c] = pid
        for k in range(6):
            st[k] = layer
        return st

    full = {(0, i): v for i, v in enumerate([7, 8, 9, 10, 11, 10, 9, 8, 7])}
    full.update({(9, i): v for i, v in enumerate([6, 5, 6, -1, 6, -1, 6, 5, 6])})
    cases.append(('空盘', np.zeros((7, 10, 9), dtype=np.int32)))
    cases.append(('满编对顶', _board(full)))
    cases.append(('只剩双帅', _board({(0, 4): 11, (9, 4): 4})))
    cases.append(('无老将', _board({(5, 5): 7, (4, 3): 0})))

    # 开局（真实 env：层0==层1，重复面必须不启动）
    cases.append(('开局', ChineseChess().state))

    # 重复窗口：直接拼历史层，把三条分支都打死（比"真下往返棋"可靠——
    # 层0==层1 是"无历史"标记，此时重复面必须保持 0）
    plane_a = np.full((10, 9), -1, dtype=np.int32)
    plane_a[9, 4] = 4; plane_a[0, 4] = 11; plane_a[7, 1] = 6; plane_a[2, 7] = 13
    plane_b = plane_a.copy(); plane_b[9, 0] = 6     # 只差一格
    plane_c = plane_a.copy(); plane_c[0, 0] = 7

    def _assemble(*layers, clock=30):
        st = np.zeros((7, 10, 9), dtype=np.int32)
        for k, plane in enumerate(layers):
            st[k] = plane
        st[6] = clock      # 未吃子时钟层（真实对局里是均匀填充的计数值）
        return st

    cases.append(('重复两次(面19+20)', _assemble(plane_a, plane_b, plane_a, plane_b, plane_a)))
    cases.append(('重复一次(面19)', _assemble(plane_a, plane_b, plane_a, plane_b, plane_c)))
    cases.append(('无历史(层0==层1)', _assemble(plane_a, plane_a, plane_b, plane_c, plane_c)))
    cases.append(('窗口内无重复', _assemble(plane_a, plane_b, plane_c, plane_b, plane_c)))

    # 被将：黑炮沉底照帅（第 0 层=黑方视角的原始 id，红帅 4 在第 9 行）
    cases.append(('被将', _board({(9, 4): 4, (0, 4): 11, (0, 0): 12})))

    for name, st in cases:
        st = np.ascontiguousarray(st, dtype=np.int32)
        for player in (0, 1):
            _assert_equal(st, player, f'{name}')
    print(f'   ok  {len(cases)} 个边界局面 × 双视角')


def test_random_assembled_boards():
    """随机拼装：专打分支（历史层差异、双方有无、各种 id 组合）"""
    rng = np.random.default_rng(7)
    for t in range(400):
        st = rng.choice(np.arange(-1, 14), size=(7, 10, 9)).astype(np.int32)
        st[6] = rng.integers(0, 120, size=(10, 9)).astype(np.int32)   # 时钟层
        _assert_equal(st, 0, f'random{t}')
        _assert_equal(st, 1, f'random{t}')
    print('   ok  400 个随机拼装局面 × 双视角')


def test_concurrent_consistency():
    """并发：8 线程同时转换，结果必须与单线程完全一致（禁模块级可变缓冲）"""
    rng = np.random.default_rng(11)
    boards = []
    for _ in range(24):
        env = ChineseChess()
        for _ in range(int(rng.integers(0, 40))):
            if env.terminated:
                break
            acts = env.get_valid_actions(env.state, env.player_to_move)
            env.step(int(rng.choice(acts)))
        boards.append(env.state.copy())
    expect = [[_py(b, p) for p in (0, 1)] for b in boards]

    bad = []
    lock = threading.Lock()

    def worker(offset: int):
        for rep in range(40):
            i = (offset + rep) % len(boards)
            for p in (0, 1):
                got = CY(boards[i], p)
                if not np.array_equal(got, expect[i][p]):
                    with lock:
                        bad.append((i, p))
                    return

    threads = [threading.Thread(target=worker, args=(k * 3,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not bad, f'并发结果与单线程不一致（疑共享缓冲）: {bad[:5]}'
    print('   ok  8 线程 × 40 轮并发转换，与单线程结果完全一致')


def test_dispatch():
    """派发：v2 必须走 C；v1 走 Python 且形状 (20,10,9)"""
    env = ChineseChess()
    for _ in range(10):
        acts = env.get_valid_actions(env.state, env.player_to_move)
        env.step(int(np.random.default_rng(3).choice(acts)))
    v2 = ChineseChess.convert_to_network(env.state, 1, 2)
    assert np.array_equal(v2, CY(env.state, 1)), 'version=2 没走 C 实现'
    v1 = ChineseChess.convert_to_network(env.state, 1, 1)
    assert v1.shape == (20, 10, 9) and v1.dtype == np.float32
    assert np.array_equal(v1, ChineseChess._convert_python(env.state, 1, 1))
    print('   ok  派发正确（v2→C，v1→Python）')


def main() -> None:
    if not REQUIRED:
        print('SKIP: 本机 .so 没有 convert_v2（未重建 Cython 扩展）——Python 路径仍可用')
        return
    for fn in (test_random_games_every_position, test_edge_positions,
               test_random_assembled_boards, test_concurrent_consistency, test_dispatch):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 5 CONVERT-EXACT TESTS PASSED')


if __name__ == '__main__':
    main()
