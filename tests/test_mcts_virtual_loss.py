"""虚拟损失批量 MCTS（吞吐手术 P0）的回归测试。

要钉住的四件事：
1. K=1 的批量路径与旧的"一手一请求"逐位等价（run(n, 1) == n 次 run(1)），
   否则这次重构就是在悄悄改搜索语义；
2. K>1 时虚拟损失必须全部撤销干净——树上任意已展开节点都满足
   sum(child_n) == n - 1（首次访问即展开、不落到孩子上），且没有任何负数计数。
   VL 泄漏的典型症状就是这里的和漂移，且是沉默的（棋力缓慢劣化，不报错）；
3. K>1 真的在批量：一次 send_many 收到 K 个 state，而不是退化成一发一发；
4. 批量下依然能搜到一步制胜（终局叶子在批里就地回传、不发请求）。

运行：.venv/bin/python tests/test_mcts_virtual_loss.py
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

import core.mcts.deepMcts as dm  # noqa: E402
from core.env.gomoku import Gomoku  # noqa: E402
from services.inference.request import QueueRequest  # noqa: E402

N = Gomoku.n_actions
_real_many = dm.send_many


def _fake_factory(value: float = 0.0, log: list | None = None):
    """假引擎：均匀 policy + 常数 value，纯 CPU、零延迟，隔离出 MCTS 自身的语义"""
    def fake(states, env_name, q, is_self_play=False):
        if log is not None:
            log.append(len(states))
        policy = np.ones(N, dtype=np.float32) / N
        return [(policy.copy(), float(value)) for _ in states]
    return fake


_real_issue, _real_harvest = dm.issue, dm.harvest


def _patch_pipelined(log=None, value=0.0):
    """流水线假引擎：issue 立刻回填（lane 放行），harvest 取出均匀 policy。

    真引擎是异步的，这里同步化以隔离 MCTS 自身的记账（VL、预算、去重）。"""
    policy = np.ones(N, dtype=np.float32) / N

    def fake_issue(state, env_name, q, is_self_play=False, lane=None):
        req = QueueRequest(state, lane=lane)
        req.fill(policy, float(value))
        if log is not None:
            log.append((state, req))
        return req, 0

    def fake_harvest(request, symmetric_idx, env_name):
        return policy, float(value)

    dm.issue, dm.harvest = fake_issue, fake_harvest
    dm.send_via_queue = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('流水线路径不应走 send_via_queue'))


def _unpatch_pipelined():
    dm.issue, dm.harvest = _real_issue, _real_harvest


def _patch(log=None, value=0.0):
    dm.send_many = _fake_factory(value, log)
    dm.send_via_queue = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('批量路径不应再走 send_via_queue'))


def _unpatch():
    dm.send_many = _real_many


def _midgame(**kw):
    """黑 (7,3)-(7,6) 活四局面：搜索必须能一步制胜，适合压"批里含终局叶子"的路径"""
    env = Gomoku()
    env.reset()
    for mv in [(7, 3), (5, 5), (7, 4), (5, 6), (7, 5), (0, 0), (7, 6), (0, 1)]:
        env.step(Gomoku.move2action(mv))
    mcts = dm.NeuronMCTS(env.state, Gomoku, env.last_action, env.player_to_move,
                         planes_version=kw.get('planes_version'))
    mcts.infer_queue = None
    return env, mcts


def _fresh():
    env = Gomoku()
    env.reset()
    mcts = dm.NeuronMCTS(env.state, Gomoku, -1, 0)
    mcts.infer_queue = None
    return mcts


def _expanded_nodes(root) -> list:
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.is_expanded:
            out.append(node)
            stack.extend(node.children.values())
    return out


def test_k1_matches_sequential():
    """K=1 走批量路径必须与逐次单发的结果逐位相同"""
    _patch()
    try:
        a = _fresh()
        a.run(80)
        b = _fresh()
        for _ in range(80):
            b.run(1)
        assert np.array_equal(a.root.child_n, b.root.child_n), 'K=1 与逐次 run(1) 不一致'
        assert np.array_equal(a.root.child_w, b.root.child_w)
    finally:
        _unpatch()
    print('   ok  K=1 与旧同步路径逐位等价')


def test_no_virtual_loss_leak():
    """K∈{1,2,4,8} 跑完后：每个已展开节点 sum(child_n) == n-1，且计数非负"""
    n_sims = 240
    for k in (1, 2, 4, 8):
        _patch()
        try:
            mcts = _fresh()
            mcts.run(n_sims, batch_leaves=k)
        finally:
            _unpatch()
        assert int(mcts.root.child_n.sum()) == n_sims - 1, (
            f'K={k}: 根孩子访问和 {mcts.root.child_n.sum()} != {n_sims - 1}（虚拟损失泄漏）')
        for node in _expanded_nodes(mcts.root):
            child_sum = float(np.sum(node.child_n))
            assert child_sum == float(node.n) - 1, (
                f'K={k}: 节点 n={node.n} 与孩子访问和 {child_sum} 不符（VL 未撤销）')
            assert np.all(node.child_n >= 0), f'K={k}: 出现负访问数（VL 撤销过量）'
        print(f'   ok  K={k}: {len(_expanded_nodes(mcts.root))} 个已展开节点计数自洽')


def test_batched_requests_really_batch():
    """一次 send_many 必须拿到 K 个 state：批大小由 K 决定，而不是退化成单发"""
    log: list = []
    _patch(log=log)
    try:
        mcts = _fresh()
        mcts.run(120, batch_leaves=8)
    finally:
        _unpatch()
    assert log, '根本没有发出请求'
    # 首批只能发 1 个：根还没展开，K 条 playout 全都落在同一个根叶子上，
    # 重复的会被跳过（撤 VL、不重复请求、不计预算）。展开之后才开始满批。
    assert log[0] == 1, f'首批应为 1 发（根未展开），实际 {log[0]}'
    assert max(log[1:]) == 8, f'展开后最大批量 {max(log[1:])} != 8'
    assert sum(log) == 120, f'请求总数 {sum(log)} != 120（重复叶子跳过会导致缺口）'
    assert len(log) == 16, f'批次数 {len(log)} != 16（1 + ceil(119/8)）'
    print(f'   ok  120 次模拟 = {len(log)} 批，展开后每批 {max(log[1:])} 发')


def test_win_found_under_batching():
    """批量路径下仍必须在根上搜到一步制胜（终局叶子在批内就地回传）"""
    win = {Gomoku.move2action((7, 2)), Gomoku.move2action((7, 7))}
    got = {}
    for k in (1, 8):
        _patch()
        try:
            env, mcts = _midgame()
            mcts.run(200, batch_leaves=k)
        finally:
            _unpatch()
        best = int(np.argmax(mcts.root.child_n))
        assert best in win, f'K={k} 没搜到胜点: {Gomoku.action2move(best)}'
        got[k] = sum(mcts.root.child_n[a] for a in win)
    # 均匀 policy + 常数 value 下搜索是确定性的：批量路径不该改变"往胜点上砸多少访问"
    assert got[8] == got[1], f'批量改了访问分布: K=8 {got[8]} vs K=1 {got[1]}'
    assert got[8] >= 60, f'胜点访问太少 {got[8]}/200'
    print(f'   ok  批量搜到胜点，访问分布与 K=1 一致（{got[8]:.0f}/200 落在活四两端）')


def test_pipelined_budget_and_no_leak():
    """流水线：预算必须精确（不许被"重复叶子"烧掉），且虚拟损失零泄漏。"""
    # 6 个模拟、K=8：树最浅时 K 次选择会撞同一个未展开根叶子，预算绝不能因此作废
    for n_sims, k in ((6, 8), (40, 8), (240, 8), (240, 2)):
        _patch_pipelined()
        try:
            mcts = _fresh()
            mcts.run_pipelined(n_sims, batch_leaves=k)
        finally:
            _unpatch_pipelined()
        assert int(mcts.root.n) == n_sims, f'n={n_sims},K={k}: 根访问 {mcts.root.n} != {n_sims}'
        assert int(mcts.root.child_n.sum()) == n_sims - 1, (
            f'n={n_sims},K={k}: 孩子访问和 {mcts.root.child_n.sum()} != {n_sims - 1}')
        for node in _expanded_nodes(mcts.root):
            assert float(np.sum(node.child_n)) == float(node.n) - 1, (
                f'n={n_sims},K={k}: 节点 n={node.n} 与孩子访问和 {float(np.sum(node.child_n))} 不符')
            assert np.all(node.child_n >= 0)
        assert abs(float(mcts.get_pi(1.0).sum()) - 1.0) < 1e-5, 'get_pi 分布不合法（空树？）'
        print(f'   ok  pipelined n={n_sims} K={k}: 预算精确、VL 无泄漏、分布合法')


def test_pipelined_finds_win():
    """流水线路径同样必须搜到一步制胜"""
    _patch_pipelined()
    try:
        env, mcts = _midgame()
        mcts.run_pipelined(200, batch_leaves=8)
    finally:
        _unpatch_pipelined()
    win = {Gomoku.move2action((7, 2)), Gomoku.move2action((7, 7))}
    best = int(np.argmax(mcts.root.child_n))
    assert best in win, f'流水线没搜到胜点: {Gomoku.action2move(best)}'
    print(f'   ok  pipelined 搜到胜点（{Gomoku.action2move(best)}，200 模拟）')


def main() -> None:
    for fn in (test_k1_matches_sequential, test_no_virtual_loss_leak,
               test_batched_requests_really_batch, test_win_found_under_batching,
               test_pipelined_budget_and_no_leak, test_pipelined_finds_win):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 6 VIRTUAL-LOSS TESTS PASSED')


if __name__ == '__main__':
    main()
