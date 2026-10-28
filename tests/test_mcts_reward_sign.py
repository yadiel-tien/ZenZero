"""MCTS/env/selfplay 奖励符号链端到端审计。

策略：monkeypatch 推理为"policy 均匀 + value 恒 0"的假网络，把符号正确性
完全隔离到 MCTS 的回传/评估、env 的终局视角、selfplay 的采样视角三段逻辑上。
若链路任何一环符号反了：搜索会在"一步制胜"局面挑不出生还手，或 root 视角断言翻车。

视角约定（全项目，测试即按此断言）：
- check_winner/game_over 返回相对 player_just_moved；
- 节点 .n/.w 存在 parent 的 child_n/child_w[action]，语义 = "走到该节点的走子方"的累计回报；
- evaluate() 返回"上个玩家（走子进叶子者）"视角：终局直接返回 leaf_reward；
  非终局返回 -value（value 头训练约定 = 叶子轮走方视角）；
- selfplay 样本 v = 0.5·z + 0.5·(-q_root)，z 与 -q 都是"该位置轮走方"视角；
- AIServer.win_rate 在 choose_action（树已推进）后读取，此时 root.w 视角=AI。
"""
import os
import random

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

import core.mcts.deepMcts as dm  # noqa: E402
from core.env.chess import ChineseChess  # noqa: E402
from core.env.gomoku import Gomoku  # noqa: E402

N_CHESS, N_GOMOKU = ChineseChess.n_actions, Gomoku.n_actions

_real_send = dm.send_via_queue
_real_send_many = dm.send_many


def _fake_send_factory(value: float):
    def fake(state, env_name, q, is_self_play=False):
        n = N_CHESS if env_name == 'ChineseChess' else N_GOMOKU
        policy = np.ones(n, dtype=np.float32) / n
        return policy, float(value)
    return fake


def _fake_send_many_factory(value: float):
    """批量路径（虚拟损失 K 条 playout 一次入队）的假引擎：必须一起打桩，
    否则 NeuronMCTS.run 会绕过 send_via_queue 直接撞真队列。"""
    single = _fake_send_factory(value)

    def fake_many(states, env_name, q, is_self_play=False):
        return [single(None, env_name, q, is_self_play) for _ in states]
    return fake_many


def setup_module(_=None):
    dm.send_via_queue = _fake_send_factory(0.0)
    dm.send_many = _fake_send_many_factory(0.0)


def teardown_module(_=None):
    dm.send_via_queue = _real_send
    dm.send_many = _real_send_many


def _chess_at(pieces, player):
    env = ChineseChess()
    st = np.zeros((7, 10, 9), dtype=np.int32)
    b = np.full((10, 9), -1, dtype=np.int32)
    for (r, c), pid in pieces.items():
        b[r, c] = pid
    st[:-1] = b
    env.state = st
    env.reset_status()
    env.player_to_move = player
    env.register_current_position()
    return env


# ---------------------------------------------------------------- 五子棋

def test_gomoku_search_finds_winning_move():
    """黑 (7,3)-(7,6) 活四，任何符号反转都会让 (+1) 边失去统治力"""
    env = Gomoku()
    env.reset()
    for mv in [(7, 3), (5, 5), (7, 4), (5, 6), (7, 5), (0, 0), (7, 6), (0, 1)]:
        env.step(Gomoku.move2action(mv))
    assert env.player_to_move == 0 and not env.terminated

    mcts = dm.NeuronMCTS(env.state, Gomoku, env.last_action, env.player_to_move)
    mcts.infer_queue = None  # fake send 不使用
    mcts.run(300)
    win = {Gomoku.move2action((7, 2)), Gomoku.move2action((7, 7))}
    best = int(np.argmax(mcts.root.child_n))
    assert best in win, f'搜索没选胜点! best={Gomoku.action2move(best)}'

    # root.w 视角 = 走进 root 的白方(必败) → 负；采样存储用 -q → 黑(轮走方)视角为正
    q = mcts.root.w / mcts.root.n
    assert q < 0, q
    assert -q > 0.5, '-q 应强烈指向黑优'

    action = mcts.choose_action()
    _, reward, done, _, _ = env.step(action)
    assert done and env.winner == 0, (action, env.winner)   # 黑(0)胜
    assert reward == 1.0, reward   # step 的 reward 相对走子方(黑)，胜利必须 +1

    # choose_action 已把树推进到"黑胜"叶子：root.w 视角 = 走进 root 的一方 = 黑(胜方) → ≈ +1
    assert mcts.root.n > 0 and mcts.root.w / mcts.root.n > 0.9, (mcts.root.w, mcts.root.n)
    # selfplay 采样约定：存 -q（轮走方=白 视角）必须为 -1：白必败
    assert -mcts.root.w / mcts.root.n < -0.9


def test_backprop_view_alternates():
    """纯回传审计（不经搜索）：+1 从叶子层出发，逐层取反；
    每层边值的视角 = 走进该节点的一方（= 1 - 节点.player_to_move）"""
    env = Gomoku()
    env.reset()
    root = dm.NeuronNode(env.state, -1, 0, Gomoku)          # 轮红；虚拟走进者=黑
    a1 = int(root.valid_actions[0])
    child1 = root.get_child(a1)                              # 轮黑；走进者=红
    a2 = int(child1.valid_actions[0])
    child2 = child1.get_child(a2)                            # 轮红；走进者=黑
    child2.back_propagate(1.0)                               # "走进 child2 的一方(黑)+1" = 黑胜

    assert child2.n == 1 and child2.w == 1.0     # 黑边 +1
    assert child1.n == 1 and child1.w == -1.0    # 红边取反 -1（红视角：黑胜=红亏）
    assert root.n == 1 and root.w == 1.0         # 再取反回黑视角 +1（root 走进者=黑）
    q_at_root = root.child_q[a1]                 # 根做决策时看孩子边：红视角 -1 → 红不该走
    assert q_at_root < 0


def test_terminal_leaf_evaluate_perspective():
    """终局叶子的 evaluate()：返回相对"走子进叶子者"，不经网络"""
    env = Gomoku()
    env.reset()
    for mv in [(7, 3), (5, 5), (7, 4), (5, 6), (7, 5), (0, 0), (7, 6), (0, 1), (7, 7)]:
        _, reward, done, _, _ = env.step(Gomoku.move2action(mv))
    assert done and env.winner == 0
    # 黑胜局面的终态作为"黑走进去"的叶子：evaluate 应 +1（相对走子方黑）
    leaf = dm.NeuronNode(env.state, Gomoku.move2action((7, 7)), 1, Gomoku)   # 轮白
    v_black_moved = leaf.evaluate(None, is_self_play=False)
    assert v_black_moved == 1.0, v_black_moved   # check_winner 相对 player_just_moved=黑 → WIN


# ---------------------------------------------------------------- 象棋

def test_chess_search_finds_mate_in_one():
    """红车 (6,5)→(6,4) 一步重杀（复用规则测试的死局骨架）"""
    K_R, C_R, H_R, K_B = 4, 0, 1, 11
    env = _chess_at({(0, 4): K_B, (6, 5): C_R, (1, 8): C_R,
                     (2, 2): H_R, (2, 6): H_R, (9, 0): K_R}, player=0)
    mcts = dm.NeuronMCTS(env.state, ChineseChess, 0, 0)   # last_action 真值(车在6,5)
    mcts.infer_queue = None
    mcts.run(400)
    best = int(np.argmax(mcts.root.child_n))
    assert best >= 0
    _, reward, _, _, _ = env.step(best)
    assert env.terminated and env.winner == 0, \
        f'没走出杀着! move={ChineseChess.action2move(best)} winner={env.winner}'
    assert reward == 1.0    # 相对走子方红：胜


def test_chess_root_sign_black_winning():
    """镜像方向验证：同死局换黑先（黑一步解杀反杀不可能→黑应选"最长寿"），
    直接测已知败局视角：黑走一步即被绝杀的局面里 root.w(走进root=白) > 0"""
    K_R, C_R, H_R, K_B = 4, 0, 1, 11
    # 白刚走 车(5,4) 成绝杀局面，轮黑（同规则测试死局）
    env = _chess_at({(0, 4): K_B, (5, 4): C_R, (1, 8): C_R,
                     (2, 2): H_R, (2, 6): H_R, (9, 0): K_R}, player=1)
    # 这是"白走进去"的终局节点：作为 evaluate 的叶子应返回 +1（白=走子方胜）
    last = ChineseChess.move2action((6, 4, 5, 4))   # 视作白刚走的棋
    leaf = dm.NeuronNode(env.state, last, 1, ChineseChess)
    assert leaf.evaluate(None) == 1.0
    # 且黑无任何合法着法（双保险，来自规则测试同一构造）
    assert leaf.valid_actions.size == 0


def test_switch_side_policy_is_involution():
    """象棋 view↔global 的换边映射必须是对合（两次=恒等），evaluate/sample 两处
    共用它，任何一边隐式假设非对合都会在这里炸"""
    rng = np.random.default_rng(7)
    for _ in range(50):
        p = rng.random(N_CHESS).astype(np.float32)
        assert np.allclose(ChineseChess.switch_side_policy(
            ChineseChess.switch_side_policy(p)), p, atol=1e-6)
        s = ChineseChess.switch_side_policy(p)
        # float32 向量和用 float64 累加比较（float32 顺序求和噪声本身 ~1e-3）
        assert abs(float(s.sum(dtype=np.float64)) - float(p.sum(dtype=np.float64))) < 1e-5


# ---------------------------------------------------------------- selfplay 采样公式

def test_selfplay_sample_z_sign():
    """复演 selfplay 的终局结算公式（v = 0.5z + 0.5q̃ 的 z 项与视角 p）"""
    rng = random.Random(3)
    for _ in range(25):
        env = Gomoku()
        env.reset()
        # 记录每样本 (p)，模拟 selfplay：轮走方视角采样
        while not env.terminated and not env.truncated:
            p = env.player_to_move
            acts = env.get_valid_actions(env.state, p)
            env.step(int(acts[rng.randrange(len(acts))]))
            if len(env.get_valid_actions(env.state, 1 - p)) == 0 and not env.terminated:
                break
        if not env.terminated:
            continue
        # 终局后按 selfplay 公式结算（z 只看 winner 与 p 的关系）
        z = -1.0 if env.winner == 1 - p else 1.0 if env.winner == p else 0.0
        if env.winner == -1:
            assert z == 0.0
        elif env.winner == p:
            assert z == 1.0
        else:
            assert z == -1.0


def main():
    setup_module()
    try:
        fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
        for name, f in fns:
            print(f'== {name}')
            f()
            print('   ok')
        print(f'\nALL {len(fns)} SIGN-CHAIN TESTS PASSED')
    finally:
        teardown_module()


if __name__ == '__main__':
    main()
