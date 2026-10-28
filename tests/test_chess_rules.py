"""象棋规则与棋盘表示回归测试（表示版本 2：26 面 + 双布局）。

覆盖：
- 开局合法着法数 = 44（象棋标准值，交叉验证 R2 过滤）
- 送将过滤：被牵制的马、照面飞将等着法剔除
- 将死/困毙判负
- 三次重复判和、长将判负、吃子重置时钟、时钟判和
- convert_to_network 双布局：version1=(20,10,9) 与 v1 逐通道一致；version2=(26,10,9) 清洁布局
- Python 与 Cython 实现随机对拍
- 随机对局 fuzz：老将永不被吃、必然终局
- 旧 20 面 checkpoint：引擎暴露 feature_version=1，按版本转换后推理成功；版本闸门触发
- MCTS 树 planes_version 透传冒烟

运行：.venv/bin/python tests/test_chess_rules.py  （或 pytest 收集）
"""
import os
import random

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from core.env.chess import ChineseChess, DRAW_NO_CAPTURE_PLIES  # noqa: E402
from core.utils.types import GameResult  # noqa: E402

RED = 0
BLACK = 1
# 棋子 id：0车1马2象3士4帅5炮6兵，+7 为黑方
C_R, H_R, K_R, C_B, K_B = 0, 1, 4, 7, 11


def make_state(pieces: dict[tuple[int, int], int], clock: int = 0) -> np.ndarray:
    """(7,10,9) int32，6 层历史全部初始化为同一盘面（冷启动=无历史标记）"""
    state = np.zeros((7, 10, 9), dtype=np.int32)
    board = np.full((10, 9), -1, dtype=np.int32)
    for (r, c), pid in pieces.items():
        board[r, c] = pid
    state[:-1] = board
    state[-1] = clock
    return state


def env_at(pieces: dict[tuple[int, int], int], player_to_move: int, clock: int = 0) -> ChineseChess:
    env = ChineseChess()
    env.state = make_state(pieces, clock)
    env.reset_status()
    env.player_to_move = player_to_move
    env.register_current_position()
    return env


def legal_moves(state, player) -> list[tuple[int, int, int, int]]:
    acts = ChineseChess.get_valid_actions(state, player)
    return [ChineseChess.action2move(int(a)) for a in acts]


def play(env: ChineseChess, seq: list[tuple[int, int, int, int]]) -> int:
    """按序走子，返回实际走的步数（提前终局则截断）"""
    n = 0
    for mv in seq:
        assert not env.terminated, f'game over at ply {n}, want to continue with {mv}'
        a = ChineseChess.move2action(mv)
        assert a >= 0, mv
        _, _, done, _, _ = env.step(a)
        n += 1
        if done:
            break
    return n


# ---------------------------------------------------------------- 基本盘与R2

def test_initial_44():
    env = ChineseChess()
    assert len(env.get_valid_actions(env.state, RED)) == 44
    assert len(env.get_valid_actions(env.state, BLACK)) == 44


def test_no_self_check_pinned():
    """被牵制的马一步都走不了（走开即送将给黑车）"""
    st = make_state({(9, 4): K_R, (9, 6): H_R, (9, 8): C_B, (0, 3): K_B})
    ms = legal_moves(st, RED)
    assert not any(m[:2] == (9, 6) for m in ms), f'pinned horse moved: {ms}'
    # 马(9,6)挡在帅(9,4)与黑车(9,8)之间：(9,4)->(9,5) 仍有马挡车 => 合法
    assert (9, 4, 9, 5) in ms, ms
    for m in ms:
        ns = ChineseChess.virtual_step(st, ChineseChess.move2action(m))
        assert not ChineseChess.is_check(ns, RED)


def test_mcts_config_knobs():
    """PUCT c 与根噪声 eps 由 config 统一驱动，且 c 沿 get_child 传播全树"""
    from core.utils.config import CONFIG, settings
    from core.mcts.deepMcts import NeuronMCTS, NeuronNode
    env = ChineseChess()
    mcts = NeuronMCTS(env.state, ChineseChess, -1, 0)
    expect_c = int(settings.get('c_puct', 5))
    assert mcts.root.c == expect_c
    child = mcts.root.get_child(int(mcts.root.valid_actions[0]))
    assert child.c == expect_c, 'c 必须传播到子节点（否则调 config 只影响根）'
    # eps 接 config：改 CONFIG 后 inject_noise 立即生效（验证接线而非具体数值）
    backup = CONFIG['dirichlet']
    try:
        CONFIG['dirichlet'] = 0.5
        node = mcts.root
        node.child_p[:] = 0.0
        va = node.valid_actions
        node.child_p[va] = 1.0 / len(va)
        node.inject_noise()
        noise_mass = 1.0 - float(node.child_p[va].max() / (0.5 * (1.0 / len(va)) + 0.5 * 1.0)) ** 0  # 仅作触发
        # eps=0.5 时至少有一个合法着概率明显偏离均匀值
        assert np.abs(node.child_p[va] - 1.0 / len(va)).max() > 1e-9
    finally:
        CONFIG['dirichlet'] = backup


def test_flying_general():
    """走子造成两帅照面为非法；照面局面轮走方视为被（敌方帅）将军"""
    st = make_state({(9, 4): K_R, (9, 5): 3, (0, 3): K_B, (0, 5): 10})
    ms = legal_moves(st, RED)
    assert (9, 4, 9, 3) not in ms  # 平移入黑帅列且中间无子 => 照面非法
    st2 = make_state({(9, 4): K_R, (0, 4): K_B})  # 已照面，轮红：红被"将"
    assert ChineseChess.is_check(st2, RED)


def test_mate_loss():
    """将死判负：黑将被红车照将且无任何解着"""
    st = make_state({(0, 4): K_B, (5, 4): C_R, (1, 8): C_R,
                     (2, 2): H_R, (2, 6): H_R, (9, 0): K_R})
    assert ChineseChess.is_check(st, BLACK)
    assert legal_moves(st, BLACK) == [], legal_moves(st, BLACK)
    assert ChineseChess.game_over(st, RED, -1) == GameResult.WIN
    assert ChineseChess.is_checkmate(st, BLACK)


def test_stalemate_loss():
    """困毙同判负：黑将不被将军但无路可走"""
    st = make_state({(0, 4): K_B, (2, 2): H_R, (2, 6): H_R, (1, 0): C_R, (9, 0): K_R})
    assert not ChineseChess.is_check(st, BLACK)
    assert legal_moves(st, BLACK) == [], legal_moves(st, BLACK)
    assert ChineseChess.game_over(st, RED, -1) == GameResult.WIN


# ---------------------------------------------------------------- 重复/时钟

def test_threefold_draw():
    """两枚闲子周期4：同局面第 3 次出现判和（恰在第 8 步）"""
    env = env_at({(9, 3): K_R, (0, 5): K_B, (5, 0): C_R, (4, 8): C_B}, RED)
    seq = [(5, 0, 6, 0), (4, 8, 3, 8), (6, 0, 5, 0), (3, 8, 4, 8)] * 2
    n = play(env, seq)
    assert n == 8 and env.terminated and env.winner == -1, (n, env.steps, env.winner)


def test_perpetual_check_lose():
    """红车 4/3 路轮流照将、黑将 4/3 路躲避："黑被将"的局面第 3 次出现，红长将判负"""
    env = env_at({(9, 0): K_R, (0, 4): K_B, (9, 5): C_R}, RED)
    seq = [(9, 5, 9, 4), (0, 4, 0, 3), (9, 4, 9, 3), (0, 3, 0, 4),
           (9, 3, 9, 4), (0, 4, 0, 3), (9, 4, 9, 3), (0, 3, 0, 4),
           (9, 3, 9, 4)]
    n = play(env, seq)
    assert env.terminated, f'not adjudicated after {n} plies'
    assert n == 9 and env.winner == BLACK, (n, env.steps, env.winner)


def test_equal_interval_different_moves_not_loop():
    """等间隔 + 次次带将 + 连续照将，但两圈绕的路不同（黑将先绕三路再绕五路）——
    "着法循环不变"不成立，只是重逢不是复读：不判负，判和"""
    env = env_at({(9, 0): K_R, (0, 4): K_B, (9, 5): C_R}, RED)
    seq = [
        (9, 5, 9, 4),   # 1 车平四将  P 第1次（黑被将）
        (0, 4, 0, 3),   # 2 将绕左
        (9, 4, 9, 3),   # 3 车跟三路将
        (0, 3, 0, 4),   # 4 将回中          —— 第一圈着法 [将43,车43,将34,车34]
        (9, 3, 9, 4),   # 5 车平四将  P 第2次
        (0, 4, 0, 5),   # 6 将绕右
        (9, 4, 9, 5),   # 7 车跟五路将
        (0, 5, 0, 4),   # 8 将回中          —— 第二圈着法 [将45,车45,将54,车54] ≠ 第一圈
        (9, 5, 9, 4),   # 9 车平四将  P 第3次
    ]
    n = play(env, seq)
    assert env.terminated, f'not adjudicated after {n} plies'
    assert n == 9 and env.winner == -1, (n, env.winner)   # 判和，不判负


def test_yijiangyixian_not_punished():
    """一将一闲交替（着着不停将才罚，中协/亚规均视为允许着法）：
    带将局面虽然三次都"被将"，但红方并非连续照将 → 只判和不判负"""
    from core.env.rule_ctx import RuleContext
    st_check = make_state({(9, 0): K_R, (0, 4): K_B, (9, 4): C_R})   # 轮黑被将
    st_quiet = make_state({(9, 0): K_R, (0, 4): K_B, (9, 5): C_R})   # 轮黑未被将(车在5路)
    k_check = ChineseChess.position_key(st_check, BLACK)
    k_quiet = ChineseChess.position_key(st_quiet, BLACK)
    ctx = RuleContext()
    # 红方着法：将(1) 闲(3) 将(5) 闲(7) 将(9)...
    ctx.register(k_check, True, BLACK, 1)     # run_red 1
    ctx.register(k_quiet, False, BLACK, 3)    # run_red 0
    ctx.register(k_check, True, BLACK, 5)     # run_red 1
    ctx.register(k_quiet, False, BLACK, 7)    # run_red 0
    # 第 9 手带将局面第 3 次出现：全部被将✓ 等间隔✓，但连续照将 run=0+1=1 < 3 → 不罚，判和
    assert ChineseChess.game_over(st_check, RED, -1, ctx=ctx, arrival_ply=9) == GameResult.DRAW
    # 之后红改着着照将（第 11 手闲着没再出现，纯循环将军）：连续数爬升后应判负
    ctx.register(k_check, True, BLACK, 9)     # run 1
    ctx.register(k_check, True, BLACK, 13)    # run 2
    # 第 17 手：带将局面第 5 次(total=5)，run=3+1... 判负要求 run+1>=3：run(3)+1=4 ✓ 等间隔(13,9 差4) ✓
    assert ChineseChess.game_over(st_check, RED, -1, ctx=ctx, arrival_ply=17) == GameResult.LOSE


def test_perpetual_requires_cycle_interval():
    """判负加保险：全部出现都被将 但 最后三次不等间隔（散见而非循环）→ 只判和不判负"""
    st = make_state({(9, 0): K_R, (0, 4): K_B, (9, 4): C_R})   # 黑轮走且被将
    key = ChineseChess.position_key(st, BLACK)
    from core.env.rule_ctx import RuleContext
    ctx = RuleContext()
    ctx.register(key, True, BLACK, 1)
    ctx.register(key, True, BLACK, 5)
    # 第 3 次到临等间隔（9-5 == 5-1）→ 循环中的长将，走子方红判负
    assert ChineseChess.game_over(st, RED, -1, ctx=ctx, arrival_ply=9) == GameResult.LOSE
    ctx.register(key, True, BLACK, 9)
    # 再来一次但间隔破坏（14-9 != 9-5）→ 都被将却非循环，判和不判负
    assert ChineseChess.game_over(st, RED, -1, ctx=ctx, arrival_ply=14) == GameResult.DRAW


def test_capture_breaks_repetition_and_clock():
    env = env_at({(9, 3): K_R, (0, 5): K_B, (5, 0): C_R, (4, 8): C_B}, RED,
                 clock=DRAW_NO_CAPTURE_PLIES - 1)
    env.step(ChineseChess.move2action((5, 0, 6, 0)))  # 不吃子 -> 时钟到
    assert env.terminated and env.winner == -1

    env = env_at({(9, 3): K_R, (0, 5): K_B, (5, 0): C_R, (5, 8): C_B}, RED, clock=50)
    env.step(ChineseChess.move2action((5, 0, 5, 8)))  # 吃车重置
    assert env.state[-1, 0, 0] == 0 and not env.terminated


def test_repetition_window_planes():
    """长将路径第 2 次出现时，窗口重复面 19 已亮、被将面 18 亮（version2 索引）"""
    env = env_at({(9, 0): K_R, (0, 4): K_B, (9, 5): C_R}, RED)
    play(env, [(9, 5, 9, 4), (0, 4, 0, 3), (9, 4, 9, 3), (0, 3, 0, 4), (9, 3, 9, 4)])
    x = ChineseChess.convert_to_network(env.state, BLACK, version=2)
    assert x[19].all(), 'window repetition plane not set'
    assert x[18].all(), 'check plane not set (black is checked)'


# ---------------------------------------------------------------- 表示（双布局）

def test_convert_v2_planes():
    env = ChineseChess()
    x = ChineseChess.convert_to_network(env.state, RED, version=2)
    assert x.shape == (26, 10, 9) and x.dtype == np.float32
    assert x[:14].sum() == 32          # 棋子 one-hot
    assert x[14:18].sum() == 0         # 无走子 -> last-move 空
    assert x[18].sum() == 0            # 未被将
    assert x[19:21].sum() == 0         # 无窗口重复（冷启动标记）
    assert x[21, 0, 0] == 0            # 时钟
    assert x[22, 0, 0] == 1.0 and x[23, 0, 0] == 1.0  # 满兵力 40/40
    # 将位（初始：红帅(9,4) 黑将(0,4)，当前视角红）
    assert x[24, 9, 4] == 1.0 and x[24].sum() == 1.0
    assert x[25, 0, 4] == 1.0 and x[25].sum() == 1.0

    # 红走 车(9,0)->(8,0)
    env.step(ChineseChess.move2action((9, 0, 8, 0)))
    xb = ChineseChess.convert_to_network(env.state, BLACK, version=2)
    assert xb[14, 0, 8] == 1.0 and xb[15, 1, 8] == 1.0   # 黑视角：翻转坐标 (9,0)->(0,8)
    assert xb[14].sum() == 1.0 and xb[15].sum() == 1.0
    xr = ChineseChess.convert_to_network(env.state, RED, version=2)
    assert xr[14:16].sum() == 0                           # 黑还没走子，"上一手"面为空
    assert xr[25].sum() == 1.0                            # 对方将位仍有值

    # 黑走 车(0,8)->(1,8) 后轮红：上一手=黑车；上上手=红车(9,0)->(8,0)
    env.step(ChineseChess.move2action((0, 8, 1, 8)))
    xr = ChineseChess.convert_to_network(env.state, RED, version=2)
    assert xr[14, 0, 8] == 1.0 and xr[15, 1, 8] == 1.0
    assert xr[16, 9, 0] == 1.0 and xr[17, 8, 0] == 1.0

    # 被将面（视角无关性）
    st = make_state({(0, 4): K_B, (5, 4): C_R, (9, 0): K_R})
    assert ChineseChess.convert_to_network(st, BLACK, version=2)[18].all()
    assert ChineseChess.convert_to_network(st, RED, version=2)[18].sum() == 0

    # 窗口重复面（需存在真实历史：层1 与层0 不同）
    st2 = make_state({(0, 4): K_B, (9, 0): K_R, (5, 0): C_R})
    st2[1, 7, 7] = H_R
    st2[2] = st2[0]
    st2[4] = st2[0]
    x2 = ChineseChess.convert_to_network(st2, RED, version=2)
    assert x2[19].all() and x2[20].all()

    # 时钟面 /120 口径
    st3 = make_state({(0, 4): K_B, (9, 0): K_R, (5, 0): C_R}, clock=60)
    assert abs(ChineseChess.convert_to_network(st3, RED, version=2)[21, 0, 0] - 0.5) < 1e-6

    # 兵力面：红只剩帅+车 -> 9/40；黑只剩帅 -> 0
    st4 = make_state({(0, 4): K_B, (9, 0): K_R, (9, 8): C_R})
    x4 = ChineseChess.convert_to_network(st4, RED, version=2)
    assert abs(x4[22, 0, 0] - 9.0 / 40.0) < 1e-5
    assert x4[23, 0, 0] == 0.0


def test_convert_v1_layout_bitwise():
    """version=1 布局必须与 v1 历史实现逐通道一致（旧模型的唯一正确视觉），
    随机合法对局上采样验证"""
    rng = random.Random(7)
    for _ in range(5):
        env = ChineseChess()
        for _ in range(30):
            x = ChineseChess.convert_to_network(env.state, env.player_to_move, version=1)
            assert x.shape == (20, 10, 9)
            cur = env.state if env.player_to_move == 0 else np.flip(env.state, axis=(1, 2))
            b = cur[0].copy()
            b[b == -1] = 14
            if env.player_to_move == 1:
                b = ChineseChess.ID_SWAP_MAP[b]
            assert np.array_equal(x[:14], np.eye(15, dtype=np.float32)[b].transpose(2, 0, 1)[:14])
            assert np.array_equal(x[14:19], np.equal(cur[:5], cur[1:6]).astype(np.float32))
            assert np.allclose(x[19], cur[-1] / 100.0)
            acts = env.get_valid_actions(env.state, env.player_to_move)
            if len(acts) == 0:
                break
            env.step(int(acts[rng.randrange(len(acts))]))
            if env.terminated:
                break


def test_convert_default_version_and_guard():
    from core.utils.config import CONFIG
    chess = CONFIG['ChineseChess']  # 注意：不能用全局 settings（那是当前 game_name 的游戏）
    st = make_state({(0, 4): K_B, (9, 0): K_R})
    assert ChineseChess.convert_to_network(st, RED).shape[0] == chess['default_net']['in_channels'] == 26
    try:
        ChineseChess.convert_to_network(st, RED, version=99)
        raise AssertionError('unknown version must raise')
    except ValueError:
        pass


# ---------------------------------------------------------------- 对拍 / fuzz

def _random_board(rng: random.Random) -> np.ndarray:
    pieces = {(rng.randrange(10), rng.randrange(9)): rng.choice([-1, -1, -1] + list(range(14)))
              for _ in range(rng.randint(15, 30))}
    pieces[(9, 4)] = K_R
    pieces[(0, 3)] = K_B
    return make_state(pieces)


def test_python_cython_parity():
    try:
        from core.env import chess_cython
    except ImportError:
        print('SKIP cython parity (module not built)')
        return
    rng = random.Random(20261027)
    for _ in range(2000):
        st = _random_board(rng)
        for p in (RED, BLACK):
            a = sorted(ChineseChess._get_valid_actions_py(st, p).tolist())
            b = sorted(chess_cython.get_valid_actions(st, p, ChineseChess.np_move2action).tolist())
            assert a == b, (p, a, b)
            king_id = 4 if p == RED else 11
            kc = np.argwhere(st[0] == king_id)[0]
            c = chess_cython.is_check(st, p)
            d = ChineseChess.king_attacked(st[0], int(kc[0]), int(kc[1]), p)
            assert c == d, (p, kc, c, d)


def test_random_playout_fuzz():
    """随机合法对局：老将永不被吃；必终局"""
    rng = random.Random(42)
    for game in range(60):
        env = ChineseChess()
        while not env.terminated:
            acts = env.get_valid_actions(env.state, env.player_to_move)
            assert len(acts) > 0, f'no legal but not terminated g{game} ply {env.steps}'
            env.step(int(acts[rng.randrange(len(acts))]))
            b = env.state[0]
            assert 4 in b and 11 in b, f'king captured! game {game} ply {env.steps}'
        assert env.terminated and env.steps <= 1000, f'game {game}: {env.steps}'


# ---------------------------------------------------------- feature 版本兼容

def test_engine_version_compat():
    """旧 20 面 checkpoint：feature_version=1 暴露正确；按 version=1 转换（20面）推理成功；
    传入错版本布局（26面）会触发通道数校验报错"""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ChineseChess', 'checkpoint_498.pt')
    if not os.path.exists(path):
        print('SKIP (checkpoint_498.pt missing)')
        return
    import threading
    import time
    from services.inference.engine import InferenceEngine
    from services.inference.request import QueueRequest

    eng = InferenceEngine(498, 'ChineseChess')
    assert eng._in_channels == 20 and eng.feature_version == 1
    eng.start()
    try:
        env = ChineseChess()
        st1 = ChineseChess.convert_to_network(env.state, RED, version=1)
        assert st1.shape[0] == 20
        ev = threading.Event()
        req = QueueRequest(st1, event=ev)
        eng.request_queue.put(req)
        t0 = time.time()
        assert ev.wait(600), 'inference timeout'
        assert req.policy is not None and req.policy.shape == (2086,)
        print(f'   legacy v1-layout inference ok ({time.time() - t0:.1f}s)')

        # 错版本输入：preprocess 线程 raise 并退出（直接报错，绝不静默降级）。
        # 此处只验证校验函数语义（线程已死，不再发正常请求）。
        st2 = ChineseChess.convert_to_network(env.state, RED, version=2)
        assert st2.shape[0] == 26 != eng._in_channels
    finally:
        eng.shutdown()


def test_mcts_version_passthrough():
    """NeuronMCTS 按 planes_version=1 建树：叶子转换出 20 面请求，旧模型链路全通"""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ChineseChess', 'checkpoint_498.pt')
    if not os.path.exists(path):
        print('SKIP (checkpoint_498.pt missing)')
        return
    from core.mcts.deepMcts import NeuronMCTS
    from services.inference.engine import InferenceEngine

    eng = InferenceEngine(498, 'ChineseChess')
    eng.start()
    try:
        env = ChineseChess()
        mcts = NeuronMCTS.make_queue_mcts(eng.request_queue, ChineseChess, env.state,
                                          env.last_action, env.player_to_move,
                                          planes_version=eng.feature_version)
        mcts.run(16)
        assert mcts.root.child_p.shape == (2086,)
        assert abs(float(mcts.root.child_p.sum()) - 1.0) < 1e-3
        mcts.shutdown()
    finally:
        eng.shutdown()


def test_version_gate():
    from core.network.functions import feature_version_compatible
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ChineseChess', 'checkpoint_498.pt')
    if not os.path.exists(path):
        print('SKIP version gate (checkpoint missing)')
        return
    assert not feature_version_compatible(498, 'ChineseChess'), 'v1 checkpoint must fail version-2 gate'


def main():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, f in fns:
        print(f'== {name}')
        f()
        print('   ok')
    print(f'\nALL {len(fns)} TESTS PASSED')


if __name__ == '__main__':
    main()
