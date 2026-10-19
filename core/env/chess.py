from numpy.typing import NDArray
from core.utils.config import CONFIG
from core.utils.mirror import mirror_action_policy, apply_symmetry
from core.utils.types import ChessMove, GameResult, PieceMoveFunc

import numpy as np

from .env import BaseEnv

settings = CONFIG['ChineseChess']


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
        return self.state, {}

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
    def check_winner(cls, state: NDArray, player_just_moved: int, action_just_executed: int) -> GameResult:
        """检查胜负情况，相对于perspective_player来说
        :param action_just_executed: 刚刚做过的动作
        :param state: (7,10,9)落子后的state
        :param player_just_moved:相对于刚落子玩家来说的结果， 0红1黑
        :return: 1胜，0平，-1负, 2未分胜负"""

        board = state[0]
        if 4 not in board:  # 红帅被杀
            return GameResult.WIN if player_just_moved == 1 else GameResult.LOSE
        if 11 not in board:  # 黑帅被杀
            return GameResult.WIN if player_just_moved == 0 else GameResult.LOSE

        # 连将判负
        diffs = np.equal(state[:5], state[1:6])
        if state[-1, 0, 0] != 0:
            if np.all(diffs[0] == diffs[2]) and \
                    np.all(diffs[0] == diffs[4]) and \
                    np.all(diffs[1] == diffs[3]):
                if cls.is_check(state, 1 - player_just_moved):
                    return GameResult.LOSE

        # 100步未吃子判和
        if state[-1, 0, 0] >= 50:
            return GameResult.DRAW

        # 双方都无进攻棋子判和,有则游戏继续
        if np.isin(state[0], [0, 1, 5, 6, 7, 8, 12, 13]).any():
            return GameResult.ONGOING
        return GameResult.DRAW

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行落子
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        self.state, reward, terminated, truncated, info = self.step_fn(self.state, action, self.player_to_move)
        # 处理终局。0,1代表获胜玩家，-1代表平局，2代表未决胜负
        if terminated:
            winner = self.player_to_move if reward == 1 else 1 - self.player_to_move if reward == -1 else -1
            self.set_winner(winner)

        # 对手被绝杀，不用等到老将被吃
        # if self.is_checkmate(self.state, 1 - self.player_to_move):
        #     winner = self.player_to_move
        #     reward = 1
        #     self.set_winner(winner)

        # 记录棋谱
        # self.history.append(np.copy(self.state[:, :, 0]))

        # 更改玩家
        self.player_to_move = 1 - self.player_to_move

        self.last_action = action
        self.steps += 1

        return self.state, reward, self.terminated, self.truncated, {}

    @classmethod
    def convert_to_network(cls, state: NDArray, current_player: int) -> NDArray:
        """
        将 7x10x9 的 array 编码为 20x10x9 的 one-hot 张量供神经网络使用。
        当前玩家始终在0-6层，并且位于棋盘同一侧
        输出 arr 的含义：
        - [0~13]：当前棋盘上的棋子类型（0~13），0-6当前玩家，7-13对手
        - [14~18]：最近 5 步之间的棋盘是否保持不变
        - [19]: 未吃子步数 steps/100 归一化

        :return: arr: board_shape=(20,10, 9), dtype=np.float32
        """
        # 到对手玩家时翻转棋盘，上下和左右都要翻转，保证当前玩家始终在下方
        cur_state = state if current_player == 0 else np.flip(state, axis=(1, 2))

        arr = np.zeros((20, 10, 9), dtype=np.float32)
        # 当前盘面编码，0-13代表不同棋子，保证当前玩家始终是0-6
        board = cur_state[0].copy()
        # 避免-1干扰
        board[board == -1] = 14
        if current_player == 1:
            board = cls.ID_SWAP_MAP[board]

        arr[:14] = np.eye(15, dtype=np.float32)[board].transpose(2, 0, 1)[:14]

        # 最近4步差分历史信息
        arr[14:19] = np.equal(cur_state[:5], cur_state[1:6])

        # 编码未吃子步数，超过100判和
        arr[-1] = cur_state[-1] / 100.0

        return arr

    @classmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        """获取当前局面的合法动作
        :param player_to_move: 当前玩家。0红1黑
        :param state: (7x10x9)[0-5]近6步盘面,[6]未吃子步数。
        :return: arr: 一维int类型np数组"""
        try:  # 优先使用cython重写，环境不允许时使用python代码
            from .chess_cython import get_valid_actions
        except ImportError:
            available_actions = []
            for r in range(10):
                for c in range(9):
                    piece = int(state[0, r, c])
                    if (player_to_move == 0 and 0 <= piece <= 6) or (player_to_move == 1 and piece >= 7):
                        destinations = ChineseChess.dest_func[piece](state, r, c)
                        for tr, tc in destinations:
                            available_actions.append(cls._move2action[(r, c, tr, tc)])
            return np.array(available_actions, dtype=np.int32)
        return get_valid_actions(state, player_to_move, cls.np_move2action)

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
        """判断player_to_move是否已被将死，是否还有棋可走"""
        valid_actions = ChineseChess.get_valid_actions(state, perspective_player)
        for action in valid_actions:
            new_state = ChineseChess.virtual_step(state, action)
            if not ChineseChess.is_check(new_state, perspective_player):
                return False
        return True

    @staticmethod
    def is_check(state: NDArray, perspective_player: int) -> bool:
        """判断当前state局面下perspective_player是否被将军"""
        perspective_king = 4 if perspective_player == 0 else 11
        rival = 1 - perspective_player
        valid_actions = ChineseChess.get_valid_actions(state, rival)
        for action in valid_actions:
            _, _, tr, tc = ChineseChess.action2move(action)
            target = state[0, tr, tc]
            if target == perspective_king:
                return True
        return False

    @staticmethod
    def get_action_executor(state: NDArray, action_to_execute: int) -> int:
        """根据要执行的动作获取动作执行方，红方0，黑方1"""
        r, c, tr, tc = ChineseChess._action2move[action_to_execute]
        piece = state[0, r, c]
        if 0 <= piece < 7:
            return 0
        elif 6 <= piece < 14:
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

        # 两帅照面的情况
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
