from numpy.typing import NDArray
from core.utils.config import CONFIG
from core.utils.mirror import mirror_action_policy, apply_symmetry
from core.utils.types import ChessMove, GameResult, PieceMoveFunc

import numpy as np

from .env import BaseEnv
from .rule_ctx import RuleContext

settings = CONFIG['ChineseChess']

# ---------- 规则常量（feature version 2，单一出处） ----------
# 未吃子时钟判和：120 手（ply，任一方走一步算 1 手），即双方各 60 手不吃子（旧版注释写100、代码50，口径在此统一）
DRAW_NO_CAPTURE_PLIES = 120
# 同局面(含轮走方)第 N 次出现触发判罚（三次重复判和 / 长将判负）
REPETITION_LIMIT = 3
# 子力价值：车 马 象 士 帅 炮 兵（己方总和满编 40，归一化分母）
MATERIAL_VALUES = np.array([9.0, 4.0, 0.0, 0.0, 0.0, 4.5, 1.0], dtype=np.float32)
MATERIAL_TOTAL = 40.0
# 能跳到目标格的 8 个马位的相对偏移 (落点相对马的位移, 马腿相对马的偏移)
# 兵力存在性查表：id+1 作下标（-1 空位映射到 0），替代 np.isin 的每叶 ~8µs 全表扫描
ATTACKING_LOOKUP = np.zeros(15, dtype=bool)
ATTACKING_LOOKUP[[i + 1 for i in (0, 1, 5, 6, 7, 8, 12, 13)]] = True  # 车马炮兵(黑白)
EYE15 = np.eye(15, dtype=np.float32)  # one-hot 常量矩阵（convert 每叶使用）

# cython 热路径模块级缓存（旧版在函数内逐次 import，纯浪费）
try:
    from .chess_cython import get_valid_actions as _cy_get_valid_actions, is_check as _cy_is_check
except ImportError:
    _cy_get_valid_actions = _cy_is_check = None
try:
    # 面张量生成（v2）的 C 实现：把每叶子 35 个 numpy 调用压成一个循环。
    # 与 _convert_python 逐位等价，由 tests/test_convert_cython_exact.py 每跑必对拍；
    # 老 .so 没有这个符号时静默降级到 Python（不影响其他 Cython 热路径）。
    from .chess_cython import convert_v2 as _cy_convert_v2
except ImportError:
    _cy_convert_v2 = None

HORSE_ATTACK_JUMPS = (
    (-2, -1, -1, 0), (-2, 1, -1, 0), (-1, -2, 0, -1), (-1, 2, 0, 1),
    (1, -2, 0, -1), (1, 2, 0, 1), (2, -1, 1, 0), (2, 1, 1, 0),
)


class ChineseChess(BaseEnv):
    _move2action: dict[ChessMove, int] = {}
    _action2move: dict[int, ChessMove] = {}
    np_move2action: NDArray[np.int32] = np.zeros((10, 9, 10, 9), dtype=np.int32)
    piece2id: dict[str, int] = {}
    id2piece: dict[int, str] = {}
    advisor_moves: list[ChessMove] = []
    bishop_moves: list[ChessMove] = []
    dest_func: dict[int, PieceMoveFunc] = {}
    n_actions = 2086
    mirror_lr_actions: NDArray[np.int_] = np.zeros((n_actions,), dtype=np.int_)
    mirror_ud_actions: NDArray[np.int_] = np.zeros((n_actions,), dtype=np.int_)
    ID_SWAP_MAP = np.array([
        7, 8, 9, 10, 11, 12, 13,  # 0-6 变为 7-13
        0, 1, 2, 3, 4, 5, 6,  # 7-13 变为 0-6
        14  # 14 保持 14 (代表空位)
    ], dtype=np.int32)

    def __init__(self):
        super().__init__()
        self.shape: tuple[int, int, int] = settings['state_shape']
        self.init_class_dicts()
        # 对局过程状态（重复局面计数 / 出现时被将标记），不进 state 数组
        self.ctx = RuleContext()
        # self.history: list[NDArray[np.float32]] = []
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[NDArray[np.int32], dict]:
        """
        重置游戏, 返回当前棋盘状态
        array.board_shape=(7,10,9),[0-5]层代表最近6步，数字-1代表空白，0-13代表不同棋子。[6]层代表未吃子步数，0-100归一化到0-1
        """
        self.state: NDArray[np.int32] = np.zeros(self.shape, dtype=np.int32)
        board = np.array([
            [7, 8, 9, 10, 11, 10, 9, 8, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 12, -1, -1, -1, -1, -1, 12, -1],
            [13, -1, 13, -1, 13, -1, 13, -1, 13],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [6, -1, 6, -1, 6, -1, 6, -1, 6],
            [-1, 5, -1, -1, -1, -1, -1, 5, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 3, 2, 1, 0],
        ])
        # 历史层也填充为初始棋盘
        self.state[:-1] = board
        self.reset_status()
        # 初始局面记为第 1 次出现（三次重复判定以出现次数计，避免差一错误）
        self.register_current_position()
        return self.state, {}

    def register_current_position(self) -> None:
        """清空对局历史并把当前 state 记为第 1 次出现。
        开局库/残局库直接改写 env.state 的调用方必须跟一次本方法。"""
        self.ctx.reset()
        key = ChineseChess.position_key(self.state, self.player_to_move)
        self.ctx.register(key, ChineseChess.is_check(self.state, self.player_to_move),
                          self.player_to_move, self.steps)

    @classmethod
    def virtual_step(cls, state: NDArray[np.int32], action: int) -> NDArray[np.int32]:
        """只改变state，不计算输赢和奖励"""
        new_state = np.copy(state)
        r, c, tr, tc = cls._action2move[action]
        attacker = new_state[0, r, c]
        target = new_state[0, tr, tc]
        # 统计双方未吃子回合，用于平局
        if target == -1:
            new_state[-1, :, :] += 1
        else:
            new_state[-1, :, :] = 0

        # 历史信息
        new_state[1:6, :, :] = new_state[0:5, :, :]

        # 执行落子
        new_state[0, r, c] = -1
        new_state[0, tr, tc] = attacker
        return new_state

    @classmethod
    def position_key(cls, state: NDArray, player_to_move: int) -> bytes:
        """位置键 = 当前盘面字节 + 轮走方（同盘面不同轮走方是两个不同位置）"""
        return state[0].tobytes() + bytes((player_to_move,))

    @classmethod
    def king_attacked(cls, board: NDArray, kr: int, kc: int, own: int) -> bool:
        """(kr,kc) 的老将当前是否被对方攻击。own: 老将所属方(0红1黑)。
        直接攻击扫描四种攻击子：
        - 车/敌方帅：四向射线，首个命中子为车或帅即被攻（两帅照面 = 敌方帅的射线攻击）
        - 炮：四向射线，跳过首子(炮架)后的第一个命中子为炮即被攻
        - 马：反查 8 个可跳至王位的马位，且对应马腿无挡
        - 兵：敌兵正向一步可吃王位；敌兵已过河时左右一步亦可吃
        士象守而不攻（不出宫/不过河），无需检查。"""
        if own == 0:  # 被攻方是红，敌方为黑子(7-13)
            e_chariot, e_king, e_cannon, e_horse, e_pawn = 7, 11, 12, 8, 13
            pawn_dir = 1        # 黑兵向 +r 前进；能直吃王位的兵在 (kr-pawn_dir, kc)
            crossed = kr >= 5   # 黑兵过河线（r>=5 可横吃）
        else:           # 被攻方是黑，敌方为红子(0-6)
            e_chariot, e_king, e_cannon, e_horse, e_pawn = 0, 4, 5, 1, 6
            pawn_dir = -1
            crossed = kr <= 4   # 红兵过河线（r<=4 可横吃）

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c, screened = kr + dr, kc + dc, False
            while 0 <= r < 10 and 0 <= c < 9:
                p = board[r, c]
                if p != -1:
                    if not screened:
                        if p == e_chariot or p == e_king:
                            return True
                        screened = True  # 首子成为炮架候选
                    else:
                        if p == e_cannon:
                            return True
                        break  # 炮架后第二子非炮，该射线终止
                r += dr
                c += dc

        # 马：候选马位 s = 王位 - 跳步偏移；马腿按"以马为基准"的偏移检查
        for mr, mc, br, bc in HORSE_ATTACK_JUMPS:
            r, c = kr - mr, kc - mc
            if not (0 <= r < 10 and 0 <= c < 9) or board[r, c] != e_horse:
                continue
            if board[r + br, c + bc] == -1:
                return True

        # 兵：正向 + 过河侧向
        pr = kr - pawn_dir
        if 0 <= pr < 10 and board[pr, kc] == e_pawn:
            return True
        if crossed:
            for c in (kc - 1, kc + 1):
                if 0 <= c < 9 and board[kr, c] == e_pawn:
                    return True
        return False

    @classmethod
    def game_over(cls, state: NDArray, player_just_moved: int, action_just_executed=None,
                  ctx: RuleContext = None, valid_actions_next: NDArray = None,
                  arrival_ply: int = None) -> GameResult:
        """终局判定唯一入口（feature version 2：合法着法制）。
        终局条件（相对走子方 player_just_moved）：
        1. 将死/困毙：轮走方无合法着法 → 走子方胜（送将已在 get_valid_actions 过滤，
           "老将被吃"仅作脏数据防御兜底）；
        2. 三次重复：同局面(含轮走方)第 REPETITION_LIMIT 次出现 →
           每次出现轮走方都被将军 **且走子方正在连续不停照将（连续≥3手，一将一闲不罚）
           且"着法循环不变"（两次间隔相等，且相邻两段的着法序列逐一相同——
           仅等间隔不算循环，绕不同路径回到同一局面只是重逢不是复读）** → 将军方(=走子方)长将判负；
           其余满三次（一将一闲交替、间隔不等、散见出现）→ 判和。
           口径出处：中协棋例总纲"任何情况下均不允许单方面长将"＋"长将＝连续不停照将
           而形成循环达三次"；裁决时机取引擎赛规则（同一局面第三次出现即判，人类比赛
           的"先提示变着再判负"在无裁判的 selfplay 里等价于直接判）。
           隐含窗口：吃子严格减子而 key 含全部兵力，同 key 出现之间不可能有吃子，
           三次出现必落在同一个无吃子段内——不存在"隔全局散见累计"的误判。
        3. 时钟：未吃子计数 >= DRAW_NO_CAPTURE_PLIES → 和；
        4. 兵力：双方均无进攻子(车马炮兵) → 和。
        :param ctx: RuleContext。传入时启用 2（对局级）；None 时只做仅凭盘面即可完成的判定
               （搜索树里的假设局面不构成对局事实；叶子判定按"历史次数+1"补上当前局面）。
        :param valid_actions_next: 调用方已算好的轮走方合法着法（树节点复用省一次生成）
        :param arrival_ply: 本次出现的手数（env.step 传 self.steps+1）。None = 跳过等间隔核查，
               仅用于单元测试与兼容路径
        :return: 1胜 0平 -1负 2未分胜负（相对 player_just_moved）"""
        board = state[0]
        if 4 not in board:  # 红帅不在（防御兜底，正常合法对局不会走到）
            return GameResult.WIN if player_just_moved == 1 else GameResult.LOSE
        if 11 not in board:  # 黑帅不在
            return GameResult.WIN if player_just_moved == 0 else GameResult.LOSE

        side_to_move = 1 - player_just_moved
        if valid_actions_next is None:
            valid_actions_next = cls.get_valid_actions(state, side_to_move)
        if valid_actions_next.size == 0:
            # 将死与困毙都是轮走方无路可走，走子方胜
            return GameResult.WIN

        # 未吃子时钟判和（计数在 state 内，搜索树内同样适用）
        if state[-1, 0, 0] >= DRAW_NO_CAPTURE_PLIES:
            return GameResult.DRAW
        # 双方都无进攻棋子判和
        if not ATTACKING_LOOKUP[board + 1].any():
            return GameResult.DRAW

        # 重复局面判罚（仅对局级 ctx；本次出现尚未登记，计数按 历史+1 补上）
        if ctx is not None:
            key = cls.position_key(state, side_to_move)
            total = ctx.count(key) + 1
            if total >= REPETITION_LIMIT:
                checked = cls.is_check(state, side_to_move)
                checks = ctx.checked_count(key, side_to_move) + int(checked)
                if (checks == total and ctx.check_run(player_just_moved) + 1 >= REPETITION_LIMIT
                        and (arrival_ply is None
                             or ctx.loop_ok(key, arrival_ply, action_just_executed))):
                    # 每次出现轮走方都在被将（键含轮走方，将军方恒为走子方），且等间隔循环 → 长将判负
                    return GameResult.LOSE
                return GameResult.DRAW
        return GameResult.ONGOING

    @classmethod
    def check_winner(cls, state: NDArray, player_just_moved: int, action_just_executed: int,
                     ctx: RuleContext = None, valid_actions_next: NDArray = None) -> GameResult:
        """兼容旧接口名，委托 game_over。
        :param state: (7,10,9)落子后的state
        :param player_just_moved: 相对于刚落子玩家的结果，0红1黑"""
        return cls.game_over(state, player_just_moved, action_just_executed,
                             ctx=ctx, valid_actions_next=valid_actions_next)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行落子（对局级判定：送将过滤 + 将死/困毙 + 三次重复 + 长将 + 时钟 + 兵力）
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        player_just_moved = self.player_to_move
        self.state = self.virtual_step(self.state, action)

        # 判定在前、记录在后（game_over 对当前这次出现按 +1 计）；出现序号 = 本步落子后的步数
        arrival_ply = self.steps + 1
        result = ChineseChess.game_over(self.state, player_just_moved, action, ctx=self.ctx,
                                        arrival_ply=arrival_ply)
        side_to_move = 1 - player_just_moved
        self.ctx.register(ChineseChess.position_key(self.state, side_to_move),
                          ChineseChess.is_check(self.state, side_to_move), side_to_move,
                          arrival_ply, action)

        # 处理终局。0,1代表获胜玩家，-1代表平局，2代表未决胜负
        reward = 0.0
        if result != GameResult.ONGOING:
            reward = float(result)
            winner = (player_just_moved if result == GameResult.WIN
                      else side_to_move if result == GameResult.LOSE else -1)
            self.set_winner(winner)

        # 更改玩家
        self.player_to_move = side_to_move
        self.last_action = action
        self.steps += 1

        return self.state, reward, self.terminated, self.truncated, {}

    @classmethod
    def convert_to_network(cls, state: NDArray, current_player: int, version: int = None) -> NDArray:
        """把 state 编码成网络输入面。v2 有 Cython 实现就走 C（逐位等价，见
        tests/test_convert_cython_exact.py），否则回退纯 Python 参考实现 _convert_python。

        为什么分派：v2 转换占 selfplay 每模拟 CPU 的 ~40%（35 个 numpy 调用的固定开销），
        而它是无状态纯函数、API 稳定，正是"Cython 只给影响很大的热路径"该用的地方；
        v1（守擂老模型的 20 面）用量小，继续走 Python。
        """
        if version is None:
            version = int(settings.get('feature_version', 1))
        if version == 2 and _cy_convert_v2 is not None:
            return _cy_convert_v2(state, current_player)
        return cls._convert_python(state, current_player, version)

    @classmethod
    def _convert_python(cls, state: NDArray, current_player: int, version: int = None) -> NDArray:
        """
        将 7x10x9 的 state 编码为神经网络用面张量；**按 feature version 输出对应布局**：
        - version=1：(20,10,9)，498 等v1 老模型出生时的 20 面布局（棋子14+差分5+时钟1）
        - version=2：(26,10,9)，v2 清洁布局（当前训练默认）
        version 缺省取当前 config 的 feature_version（棋盘表示版本，与训练 epoch 无关）。推理引擎暴露自身 feature_version，MCTS 树/采样必须
        按"目标引擎的 feature_version"转换——面语义随表示版本整体变化，不做跨版本切片兼容。
        当前玩家始终在 0-6 层、位于棋盘同一侧（翻转+swap 归一化）。

        version 2 面定义（26 面，一条面=一个事实）：
        - [0~13] 棋子 one-hot：0-6 己方、7-13 对手（与 v1 同语义）
        - [14/15] 上一手（对手刚走）起点/落点格（当前视角）
        - [16/17] 上上手（我方走的）起点/落点格（当前视角）
        - [18]    当前玩家老将被将军（广播）
        - [19]    6 步历史窗口内同局面（含轮走方）已出现过（广播）
        - [20]    窗口内已出现两次（第三次即将触发判罚，广播）
        - [21]    未吃子时钟 = 计数/120（广播，与 DRAW_NO_CAPTURE_PLIES 同口径）
        - [22/23] 我方/对手兵力价值合计 /40（广播）
        - [24/25] 我方老将位 / 对方老将位（单格面）
        设计原则：P1 一面一事实；P2 规则事实（被将/重复/时钟）显式给出；
        P3 走法生成器 O(1) 可算的（last-move、兵力、将位）不让卷积去猜。
        v1 差分面（哪里变了）由 from/to 面（从哪到哪、吃的子在哪）取代。
        注1：19/20 只依赖 state 自带的历史窗口、不看对局 ctx——MCTS 树内叶子与对局
        真实位置的输入分布必须一致；全对局三次重复/长将判罚在 game_over(ctx) 规则端。
        注2：冷启动标记——reset/开局库填充的 6 层全同表示"无历史"（层0==层1 时
        窗口判定不启动），避免把"无历史"误读成"每层都重复"。
        """
        if version is None:
            version = int(settings.get('feature_version', 1))
        # 到对手玩家时翻转棋盘（上下+左右），保证当前玩家始终在下方
        cur_state = state if current_player == 0 else np.flip(state, axis=(1, 2))
        board = cur_state[0].copy()
        board[board == -1] = 14
        if current_player == 1:
            board = cls.ID_SWAP_MAP[board]
        piece_planes = EYE15[board].transpose(2, 0, 1)[:14]

        if version == 1:
            # v1 布局，逐通道与历史实现一致（旧 checkpoint 的唯一正确视觉）
            arr = np.zeros((20, 10, 9), dtype=np.float32)
            arr[:14] = piece_planes
            arr[14:19] = np.equal(cur_state[:5], cur_state[1:6])
            arr[19] = cur_state[-1] / 100.0
            return arr
        if version != 2:
            raise ValueError(f'Unknown chess feature version: {version}')

        arr = np.zeros((26, 10, 9), dtype=np.float32)
        arr[:14] = piece_planes

        # ---------- 最近两手的起点/落点：由相邻历史盘差分恢复 ----------
        # 走子方落点必为己方子、起点必变空，因此两格唯一且可区分（象棋无易位）。
        # 视角：上一手是对手走（子力 id 7-13），上上手是当前玩家走（0-6）。
        b1, b2 = cur_state[1], cur_state[2]
        v1 = np.where(b1 == -1, 14, b1)
        v2 = np.where(b2 == -1, 14, b2)
        if current_player == 1:
            v1 = cls.ID_SWAP_MAP[v1]
            v2 = cls.ID_SWAP_MAP[v2]
        # 上一手（对手走）：起点 = 上盘有 7-13、本盘为空(14)；落点 = 本盘 7-13 且与上盘不同
        moved_from = (v1 >= 7) & (v1 <= 13) & (board == 14)
        moved_to = (board >= 7) & (board <= 13) & (board != v1)
        arr[14][moved_from] = 1.0
        arr[15][moved_to] = 1.0
        # 上上手（我方走）：对比 v2 -> v1
        moved_from2 = (v2 >= 0) & (v2 <= 6) & (v1 == 14)
        moved_to2 = (v1 >= 0) & (v1 <= 6) & (v1 != v2)
        arr[16][moved_from2] = 1.0
        arr[17][moved_to2] = 1.0

        # 被将军（广播）：在完整 state 上判断（视角无关）
        if cls.is_check(state, current_player):
            arr[18] = 1.0

        # 窗口重复（广播）：盘面相同且轮走方相同 <=> 相隔偶数步（2/4 层）
        cur0 = cur_state[0]
        if not np.array_equal(cur0, cur_state[1]):
            eq2 = np.array_equal(cur0, cur_state[2])
            eq4 = np.array_equal(cur0, cur_state[4])
            if eq2 or eq4:
                arr[19] = 1.0
                if eq2 and eq4:
                    arr[20] = 1.0

        # 未吃子时钟（广播）：与判和阈值同口径
        arr[21] = cur_state[-1] / DRAW_NO_CAPTURE_PLIES

        # 兵力合计（广播）：己方 0-6 / 对手 7-13
        hist = np.bincount(board.ravel(), minlength=15)
        arr[22] = (hist[:7] * MATERIAL_VALUES).sum() / MATERIAL_TOTAL
        arr[23] = (hist[7:14] * MATERIAL_VALUES).sum() / MATERIAL_TOTAL

        # 将位（单格面）：飞将/助攻的显式锚点（卷积本也能找到，成本≈0 顺手给）
        arr[24][board == 4] = 1.0
        arr[25][board == 11] = 1.0
        return arr

    @classmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        """获取当前局面的合法动作（feature version 2：伪合法着法 + 送将过滤，
        被牵制的子、飞将照面等着法在此剔除）
        :param player_to_move: 当前玩家。0红1黑
        :param state: (7x10x9)[0-5]近6步盘面,[6]未吃子步数。
        :return: arr: 一维int类型np数组"""
        if _cy_get_valid_actions is not None:  # 优先 cython，环境不允许时落 python
            return _cy_get_valid_actions(state, player_to_move, cls.np_move2action)
        return cls._get_valid_actions_py(state, player_to_move)

    @classmethod
    def _get_pseudo_valid_actions_py(cls, state: NDArray, player_to_move: int) -> list[tuple[int, int, int, int]]:
        """伪合法走法（Python 参考实现，不做送将过滤）。返回 (r,c,tr,tc) 列表"""
        out: list[tuple[int, int, int, int]] = []
        for r in range(10):
            for c in range(9):
                piece = int(state[0, r, c])
                if (player_to_move == 0 and 0 <= piece <= 6) or (player_to_move == 1 and 7 <= piece <= 13):
                    for tr, tc in ChineseChess.dest_func[piece](state, r, c):
                        out.append((r, c, tr, tc))
        return out

    @classmethod
    def _get_valid_actions_py(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        """合法着法 Python 实现：伪合法 + 送将过滤（将走子后老将是否被攻击）"""
        board = state[0]
        king_id = 4 if player_to_move == 0 else 11
        ks = np.argwhere(board == king_id)
        if len(ks) == 0:  # 防御：无王（脏数据）退化为伪合法
            return np.array([cls._move2action[m] for m in cls._get_pseudo_valid_actions_py(state, player_to_move)],
                            dtype=np.int32)
        kr, kc = int(ks[0, 0]), int(ks[0, 1])
        tmp = board.copy()
        actions = []
        for r, c, tr, tc in cls._get_pseudo_valid_actions_py(state, player_to_move):
            piece = tmp[r, c]
            tmp[r, c] = -1
            tmp[tr, tc] = piece
            nkr, nkc = (tr, tc) if piece == king_id else (kr, kc)
            if not cls.king_attacked(tmp, nkr, nkc, player_to_move):
                actions.append(cls._move2action[(r, c, tr, tc)])
            tmp[r, c] = piece
            tmp[tr, tc] = board[tr, tc]
        return np.array(actions, dtype=np.int32)

    def get_valid_action_from_pos(self, pos: tuple[int, int]) -> list[int]:
        """获取给定位置棋子的合法动作列表"""
        row, col = pos
        piece = int(self.state[0, row, col])
        destinations = self.dest_func[piece](self.state, row, col)
        available_actions = []
        for tr, tc in destinations:
            available_actions.append(self._move2action[(row, col, tr, tc)])
        return available_actions

    @staticmethod
    def is_checkmate(state: NDArray, perspective_player: int) -> bool:
        """判断 perspective_player（轮走方视角）是否已无路可走。
        合法着法制下将死/困毙同判负，等价于合法着法集合为空。"""
        return ChineseChess.get_valid_actions(state, perspective_player).size == 0

    @staticmethod
    def is_check(state: NDArray, perspective_player: int) -> bool:
        """判断当前state局面下perspective_player是否被将军（直接攻击扫描，
        车/帅射线、炮隔子、马蹩腿、兵直吃与过河侧吃；两帅照面按被敌方帅攻击处理）。
        每个树叶子都会调用（v2 面24），优先走 cython 直扫加速。"""
        if _cy_is_check is not None:
            return _cy_is_check(state, perspective_player)
        board = state[0]
        king_id = 4 if perspective_player == 0 else 11
        ks = np.argwhere(board == king_id)
        if len(ks) == 0:  # 老将不在盘上（防御兜底）不算被将
            return False
        return ChineseChess.king_attacked(board, int(ks[0, 0]), int(ks[0, 1]), perspective_player)

    @staticmethod
    def get_action_executor(state: NDArray, action_to_execute: int) -> int:
        """根据要执行的动作获取动作执行方，红方0，黑方1"""
        r, c, tr, tc = ChineseChess._action2move[action_to_execute]
        piece = state[0, r, c]
        if 0 <= piece < 7:
            return 0
        elif 7 <= piece < 14:
            return 1
        else:
            raise ValueError("Invalid action to execute")

    @classmethod
    def init_class_dicts(cls) -> None:
        if cls.piece2id:  # 避免重复初始化
            return

        cls.piece2id = {
            '红车': 0, '红马': 1, '红象': 2, '红士': 3, '红帅': 4, '红炮': 5, '红兵': 6,
            '黑车': 7, '黑马': 8, '黑象': 9, '黑士': 10, '黑帅': 11, '黑炮': 12, '黑兵': 13, '一一': -1
        }
        cls.id2piece = {v: k for k, v in cls.piece2id.items()}
        a = 0
        # 垂直水平移动
        for r in range(10):
            for c in range(9):
                for tr in range(10):
                    if tr != r:
                        move = (r, c, tr, c)
                        cls._move2action[move] = a
                        a += 1
                for tc in range(9):
                    if tc != c:
                        move = (r, c, r, tc)
                        cls._move2action[move] = a
                        a += 1
        # 士的动作
        cls.advisor_moves = [
            (0, 3, 1, 4), (0, 5, 1, 4), (1, 4, 0, 3), (1, 4, 0, 5), (1, 4, 2, 3), (1, 4, 2, 5), (2, 3, 1, 4),
            (2, 5, 1, 4), (9, 3, 8, 4), (9, 5, 8, 4), (8, 4, 9, 3), (8, 4, 9, 5), (8, 4, 7, 3), (8, 4, 7, 5),
            (7, 3, 8, 4), (7, 5, 8, 4)
        ]
        for move in cls.advisor_moves:
            cls._move2action[move] = a
            a += 1
        # 象的动作
        cls.bishop_moves = [
            (0, 2, 2, 0), (0, 2, 2, 4), (0, 6, 2, 4), (0, 6, 2, 8), (2, 0, 0, 2), (2, 0, 4, 2), (2, 4, 0, 2),
            (2, 4, 4, 2), (2, 4, 0, 6), (2, 4, 4, 6), (2, 8, 0, 6), (2, 8, 4, 6), (4, 2, 2, 0), (4, 2, 2, 4),
            (4, 6, 2, 4), (4, 6, 2, 8)
        ]
        rival_bishop_moves = [(9 - r, c, 9 - tr, tc) for r, c, tr, tc in cls.bishop_moves]
        cls.bishop_moves.extend(rival_bishop_moves)
        for move in cls.bishop_moves:
            cls._move2action[move] = a
            a += 1
        # 马的动作
        for r in range(10):
            for c in range(9):
                for dr, dc, in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
                    tr, tc = r + dr, c + dc
                    if 0 <= tr < 10 and 0 <= tc < 9:
                        move = (r, c, tr, tc)
                        cls._move2action[move] = a
                        a += 1
        cls._action2move = {v: k for k, v in cls._move2action.items()}

        # 左右、上下翻转后的动作对应
        for action in range(cls.n_actions):
            r, c, tr, tc = cls.action2move(action)
            mirror_lr_move = (r, 8 - c, tr, 8 - tc)
            mirror_ud_move = (9 - r, c, 9 - tr, tc)
            cls.mirror_lr_actions[action] = cls._move2action[mirror_lr_move]
            cls.mirror_ud_actions[action] = cls._move2action[mirror_ud_move]
        # numpy，方便cython
        for (r, c, tr, tc), action in cls._move2action.items():
            cls.np_move2action[r, c, tr, tc] = action

        cls.dest_func = {
            0: cls._get_rook_dest,
            1: cls._get_horse_dest,
            2: cls._get_bishop_dest,
            3: cls._get_advisor_dest,
            4: cls._get_king_dest,
            5: cls._get_cannon_dest,
            6: cls._get_pawn_dest
        }
        for i in range(7, 14):
            cls.dest_func[i] = cls.dest_func[i - 7]

    @classmethod
    def get_board_str(cls, state: NDArray, player_to_move: int, colorize: bool = True) -> str:
        """打印棋盘"""
        board_str = '\n ' + ''.join([f'{i:^5}' for i in range(9)]) + '\n'
        for i in range(10):
            row_str = f'{i}'
            for piece_id in state[0][i]:
                ch = cls.id2piece[int(piece_id)]
                if 0 <= piece_id <= 6 and colorize:
                    row_str += f' \033[91m{ch:2}\033[0m'
                else:
                    row_str += f' {ch:2}'
            board_str += row_str + '\n'

        return board_str

    def render(self) -> None:
        print(self.get_board_str(self.state, self.player_to_move))

    @classmethod
    def handle_human_input(cls, state: NDArray, last_action: int, player_to_move: int) -> int:
        cls.get_board_str(state, player_to_move)
        valid_actions = cls.get_valid_actions(state, player_to_move)
        while True:
            txt = input('输入一个4位数字，前两位代表当前棋子位置，后两位代表移动到的位置，例如红方炮7平4为7774。\n')
            if not (len(txt) == 4 and txt.isdigit()):
                print("输入格式有误！请确保是 4 位数字，例如 7774。")
                continue
            r, c, tr, tc = map(int, txt)
            move = r, c, tr, tc
            if move not in cls._move2action:
                print("该步不在合法动作表中，可能位置超出棋盘或走法无效。")
                continue

            action = cls.move2action(move)
            if action not in valid_actions:
                print("该走法不合法（可能被蹩马脚、被将军、或无该棋子）！请重新输入。")
                continue

            return action
        raise RuntimeError("handle_human_input should never reach here")

    def describe_last_move(self) -> None:
        if self.last_action == -1:
            return
        r, c, tr, tc = self._action2move[self.last_action]
        big_char = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
        red_col = big_char[::-1]
        black_col = list(range(1, 10))
        desc = ''
        piece_id = int(self.state[1, r, c])
        piece = self.id2piece[piece_id]
        desc += piece[1]
        is_red = 0 <= piece_id < 7
        if is_red:
            desc += red_col[c]
            if r == tr:
                desc += '平' + red_col[tc]
            elif r > tr:
                desc += '进'
                if c == tc:
                    desc += big_char[r - tr - 1]
                else:
                    desc += red_col[tc]
            else:
                desc += '退'
                if c == tc:
                    desc += big_char[tr - r - 1]
                else:
                    desc += red_col[tc]
        else:
            desc += str(black_col[c])
            if r == tr:
                desc += f'平{black_col[tc]}'
            elif r > tr:
                desc += '退'
                if c == tc:
                    desc += str(black_col[r - tr - 1])
                else:
                    desc += str(black_col[tc])
            else:
                desc += '进'
                if c == tc:
                    desc += str(black_col[tr - r - 1])
                else:
                    desc += str(black_col[tc])

        eat_piece = self.id2piece[int(self.state[1, tr, tc])]
        result = '' if eat_piece == '一一' else '吃 ' + eat_piece
        print(f'{desc} ({r},{c}) -> ({tr}, {tc}) {result}')

    @classmethod
    def action2move(cls, action: int) -> ChessMove:
        return cls._action2move[action]

    @classmethod
    def move2action(cls, move: ChessMove) -> int:
        """:return 如果move不存在返回 -1"""
        return cls._move2action.get(move, -1)

    @classmethod
    def restore_policy(cls, policy: NDArray, symmetry_idx: int) -> NDArray:
        return mirror_action_policy(policy, symmetry_idx, cls.mirror_lr_actions,
                                    cls.mirror_ud_actions)

    @classmethod
    def switch_side_policy(cls, policy: NDArray) -> NDArray:
        """交换红黑双方，上下和左右都进行了镜像翻转，输出翻转后的policy"""
        lr_policy = mirror_action_policy(policy, 4, cls.mirror_lr_actions, cls.mirror_ud_actions)
        return mirror_action_policy(lr_policy, 5, cls.mirror_lr_actions, cls.mirror_ud_actions)

    @staticmethod
    def _get_rook_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        own_side = board[r, c] < 7  # 0-6红方
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            tr, tc = r, c
            while True:
                tr += dr
                tc += dc
                if not (0 <= tr < 10 and 0 <= tc < 9):
                    break  # 越界
                target = board[tr, tc]

                if target == -1:  # 空位
                    destinations.append((tr, tc))
                elif (target < 7) != own_side:  # 对手棋子，可以吃
                    destinations.append((tr, tc))
                    break
                else:  # 己方棋子
                    break

        return destinations

    @staticmethod
    def _get_horse_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        horse_moves = (
            ((-2, -1), (-1, 0)),  # 上上左，蹩脚点：上
            ((-2, 1), (-1, 0)),  # 上上右
            ((-1, -2), (0, -1)),  # 上左左，蹩脚点：左
            ((-1, 2), (0, 1)),  # 上右右
            ((1, -2), (0, -1)),  # 下左左
            ((1, 2), (0, 1)),  # 下右右
            ((2, -1), (1, 0)),  # 下下左
            ((2, 1), (1, 0)),  # 下下右
        )
        own_side = board[r, c] < 7
        for (dr, dc), (br, bc) in horse_moves:
            tr, tc = r + dr, c + dc
            block_r, block_c = r + br, c + bc
            # 目标位置在棋盘上，且没有蹩脚
            if 0 <= tr < 10 and 0 <= tc < 9:
                if board[block_r, block_c] != -1:
                    continue
                # 目标位置没有棋子或有对方棋子
                target = board[tr, tc]
                if target == -1 or (target < 7) != own_side:
                    destinations.append((tr, tc))
        return destinations

    @staticmethod
    def _get_bishop_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        own_side = board[r, c] < 7
        for from_r, from_c, tr, tc in ChineseChess.bishop_moves:
            if from_r != r or from_c != c:
                continue
            # 象眼
            block_r, block_c = (from_r + tr) // 2, (from_c + tc) // 2
            if board[block_r, block_c] != -1:
                continue

            target = board[tr, tc]
            if target == -1 or (target < 7) != own_side:
                destinations.append((tr, tc))
        return destinations

    @staticmethod
    def _get_advisor_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        own_side = board[r, c] < 7
        for from_r, from_c, tr, tc in ChineseChess.advisor_moves:
            if from_r != r or from_c != c:
                continue
            target = board[tr, tc]
            if target == -1 or (target < 7) != own_side:  # 目标位置为空或为敌方棋子
                destinations.append((tr, tc))
        return destinations

    @staticmethod
    def _get_king_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        candidates = ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
        is_red_king = board[r, c] == 4
        valid_rows = range(7, 10) if is_red_king else range(0, 3)
        rival_king = 11 if is_red_king else 4
        for tr, tc in candidates:
            if tc < 3 or tc > 5:
                continue
            if tr not in valid_rows:
                continue

            target = board[tr, tc]
            if target == -1 or ((target < 7) != is_red_king):
                destinations.append((tr, tc))

        # 两帅照面的情况（照面仅在 3-5 路有意义：帅不离九宫；与 cython 版对齐）
        if 3 <= c <= 5:
            dr = -1 if is_red_king else 1
            tr = r + dr
            while 0 <= tr <= 9:
                target = board[tr, c]
                if target == -1:
                    tr += dr
                elif target == rival_king:
                    destinations.append((tr, c))
                    break
                else:
                    break

        return destinations

    @staticmethod
    def _get_cannon_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        own_side = board[r, c] < 7
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            found_screen = False  # 炮架
            tr, tc = r + dr, c + dc
            while 0 <= tr <= 9 and 0 <= tc <= 8:
                target = int(board[tr, tc])
                if found_screen:
                    if target == -1:
                        pass  # 炮架后空格不可走
                    elif (target < 7) != own_side:  # 对方棋子可吃
                        destinations.append((tr, tc))
                        break
                    else:  # 己方棋子不可吃
                        break
                else:
                    if target == -1:  # 炮架前空格可走
                        destinations.append((tr, tc))
                    else:
                        found_screen = True  # 第一个遇到的棋子是炮架
                tr += dr
                tc += dc

        return destinations

    @staticmethod
    def _get_pawn_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[0]
        destinations = []
        is_red_pawn = board[r, c] == 6
        dr = -1 if is_red_pawn else 1
        has_crossed = (r <= 4) if is_red_pawn else (r > 4)  # 过河
        candidates = [(r + dr, c), (r, c - 1), (r, c + 1)] if has_crossed else [(r + dr, c)]
        for tr, tc in candidates:
            if 0 <= tr < 10 and 0 <= tc < 9:  # 在棋盘上
                target = board[tr, tc]
                if target == -1 or ((target < 7) != is_red_pawn):
                    destinations.append((tr, tc))

        return destinations

    @classmethod
    def symmetry_indices(cls) -> tuple[int, ...]:
        return (0, 4)   # 象棋只允许左右镜像（上下翻转会交换双方）

    @classmethod
    def transform_batch(cls, states: NDArray, policies: NDArray, symmetry_idx: int):
        if symmetry_idx == 0:
            return states, policies
        if symmetry_idx == 4:
            # state: 宽度翻转（apply_symmetry idx4 的批量版）
            st = np.flip(states, axis=3)
            # policy: 按 2086 维置换表重排，语义同 mirror_action_policy（双射，无碰撞）
            pi = np.empty_like(policies)
            pi[:, cls.mirror_lr_actions] = policies
            return st, pi
        raise ValueError(f'ChineseChess symmetry idx must be 0 or 4, got {symmetry_idx}')

    @classmethod
    def augment_data(cls, data: tuple[NDArray, NDArray, float]) -> list[tuple[NDArray, NDArray, float]]:
        """通过旋转和翻转棋盘进行数据增强
            - ChineseChess 只支持左右翻转
        :param data: (state,pi,q)
         :return 增强后的列别[(state,pi,q)]"""
        state, pi, q = data
        augmented_samples = [data]
        transformed_state = apply_symmetry(state, 4)
        transformed_prob = mirror_action_policy(pi, 4, cls.mirror_lr_actions,
                                                cls.mirror_ud_actions)
        augmented_samples.append((transformed_state, transformed_prob, q))
        return augmented_samples
