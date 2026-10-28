# cython: boundscheck=False, wraparound=False, cdivision=True,initializedcheck=False,freethreading_compatible=True
# freethreading_compatible=True: 声明本模块可在无GIL环境下运行——模块级变量均为只读常量表，
# 各函数只读写调用方传入的数组，无模块级可变状态，多线程安全。
import numpy as np
from libc.stdint cimport int32_t
cimport numpy as cnp

ctypedef cnp.int32_t INT32_t

# ----------------------
# 定义常量
# ----------------------
cdef int DIRECTIONS[4][2]
DIRECTIONS = [[1, 0], [-1, 0], [0, 1], [0, -1]]

cdef int HORSE_MOVES[8][4]
HORSE_MOVES = [
    [-2, -1, -1, 0],
    [-2, 1, -1, 0],
    [-1, -2, 0, -1],
    [-1, 2, 0, 1],
    [1, -2, 0, -1],
    [1, 2, 0, 1],
    [2, -1, 1, 0],
    [2, 1, 1, 0]
]

# 四个数字分别为tr,tc,br,bc，即目的地和象眼坐标。某个位置的开始坐标需要通过lookup表查得
cdef int BISHOP_MOVES_DATA[32][4]
BISHOP_MOVES_DATA = [
    [2, 0, 1, 1],
    [2, 4, 1, 3],
    [2, 4, 1, 5],
    [2, 8, 1, 7],
    [0, 2, 1, 1],
    [4, 2, 3, 1],
    [0, 2, 1, 3],
    [0, 6, 1, 5],
    [4, 2, 3, 3],
    [4, 6, 3, 5],
    [0, 6, 1, 7],
    [4, 6, 3, 7],
    [2, 0, 3, 1],
    [2, 4, 3, 3],
    [2, 4, 3, 5],
    [2, 8, 3, 7],
    [7, 0, 6, 1],
    [7, 4, 6, 3],
    [7, 4, 6, 5],
    [7, 8, 6, 7],
    [5, 2, 6, 1],
    [9, 2, 8, 1],
    [5, 2, 6, 3],
    [5, 6, 6, 5],
    [9, 2, 8, 3],
    [9, 6, 8, 5],
    [5, 6, 6, 7],
    [9, 6, 8, 7],
    [7, 0, 8, 1],
    [7, 4, 8, 3],
    [7, 4, 8, 5],
    [7, 8, 8, 7],
]
# 每个数对代表象出在该位置时的offset和合法位置数量。然后查data表可得到具体目标位置信息。
cdef int BISHOP_LOOKUP[10][9][2]
BISHOP_LOOKUP = [
    [[0, 0], [0, 0], [0, 2], [0, 0], [0, 0], [0, 0], [2, 2], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[4, 2], [0, 0], [0, 0], [0, 0], [6, 4], [0, 0], [0, 0], [0, 0], [10, 2]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [12, 2], [0, 0], [0, 0], [0, 0], [14, 2], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [16, 2], [0, 0], [0, 0], [0, 0], [18, 2], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[20, 2], [0, 0], [0, 0], [0, 0], [22, 4], [0, 0], [0, 0], [0, 0], [26, 2]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [28, 2], [0, 0], [0, 0], [0, 0], [30, 2], [0, 0], [0, 0]]
]

cdef int ADVISOR_MOVES_DATA[16][2]
ADVISOR_MOVES_DATA = [
    [1, 4],
    [1, 4],
    [0, 3],
    [0, 5],
    [2, 3],
    [2, 5],
    [1, 4],
    [1, 4],
    [8, 4],
    [8, 4],
    [7, 3],
    [7, 5],
    [9, 3],
    [9, 5],
    [8, 4],
    [8, 4],
]

cdef int ADVISOR_LOOKUP[10][9][2]
ADVISOR_LOOKUP = [
    [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [2, 4], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [6, 1], [0, 0], [7, 1], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [8, 1], [0, 0], [9, 1], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [10, 4], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [14, 1], [0, 0], [15, 1], [0, 0], [0, 0], [0, 0]]
]

cdef int add_rook_dest(INT32_t * board, int r, int c, INT32_t *result) nogil:
    # 预分配足够大的numpy数组（假设最多20个移动）
    cdef int count = 0

    cdef int dr, dc, tr, tc, target, i
    cdef bint own_side = board[r * 9 + c] < 7

    for i in range(4):
        dr = DIRECTIONS[i][0]
        dc = DIRECTIONS[i][1]
        tr = r
        tc = c
        while True:
            tr += dr
            tc += dc
            if tr < 0 or tr >= 10 or tc < 0 or tc >= 9:
                break
            target = board[tr * 9 + tc]
            if target == -1 or (target < 7) != own_side:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = tr
                result[count * 4 + 3] = tc
                count += 1
            if target != -1:
                break

    # 返回实际有效的结果部分
    return count

cdef int add_horse_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """马的合法移动"""
    cdef int count = 0

    cdef int tr, tc, br, bc, target, i

    cdef bint own_side = board[r * 9 + c] < 7

    for i in range(8):
        tr = r + HORSE_MOVES[i][0]
        tc = c + HORSE_MOVES[i][1]
        br = r + HORSE_MOVES[i][2]
        bc = c + HORSE_MOVES[i][3]

        if tr < 0 or tr >= 10 or tc < 0 or tc >= 9:
            continue
        if board[br * 9 + bc] != -1:  #蹩马腿
            continue
        target = board[tr * 9 + tc]
        if target == -1 or (target < 7) != own_side:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = tr
            result[count * 4 + 3] = tc
            count += 1

    return count

cdef int add_bishop_dest(INT32_t * board, int r, int c, INT32_t *result) nogil:
    """相的合法移动"""
    cdef int count = 0, i, target, tr, tc, br, bc
    cdef bint own_side = board[r * 9 + c] < 7
    # 查表获取offset和合法走法数量
    cdef int offset = BISHOP_LOOKUP[r][c][0]
    cdef int move_count = BISHOP_LOOKUP[r][c][1]
    #从moves data中获取具体走法
    for i in range(move_count):
        tr = BISHOP_MOVES_DATA[offset + i][0]
        tc = BISHOP_MOVES_DATA[offset + i][1]
        br = BISHOP_MOVES_DATA[offset + i][2]
        bc = BISHOP_MOVES_DATA[offset + i][3]

        # 蹩象眼
        if board[br * 9 + bc] != -1:
            continue

        target = board[tr * 9 + tc]
        if target == -1 or (target < 7) != own_side:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = tr
            result[count * 4 + 3] = tc
            count += 1
    return count

cdef int add_advisor_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """士的合法移动"""
    cdef int count = 0, i, target, tr, tc
    cdef bint own_side = board[r * 9 + c] < 7
    cdef int offset = ADVISOR_LOOKUP[r][c][0]
    cdef int move_count = ADVISOR_LOOKUP[r][c][1]
    for i in range(move_count):
        tr = ADVISOR_MOVES_DATA[offset + i][0]
        tc = ADVISOR_MOVES_DATA[offset + i][1]

        target = board[tr * 9 + tc]
        if target == -1 or (target < 7) != own_side:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = tr
            result[count * 4 + 3] = tc
            count += 1
    return count

cdef int add_king_dest(INT32_t * board, int r, int c, INT32_t *result) nogil:
    """将/帅的合法移动"""
    cdef int count = 0

    cdef int i, tr, tc, target
    cdef bint is_red = board[r * 9 + c] == 4
    cdef int dr, rival_king

    for i in range(4):
        tr = r + DIRECTIONS[i][0]
        tc = c + DIRECTIONS[i][1]
        if tr < 0 or tr >= 10 or tc < 0 or tc >= 9:
            continue
        if is_red and tr < 7:
            continue
        if not is_red and tr > 2:
            continue
        if tc < 3 or tc > 5:
            continue
        target = board[tr * 9 + tc]
        if target == -1 or (target < 7) != is_red:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = tr
            result[count * 4 + 3] = tc
            count += 1

    # 两帅相对

    if 3 <= c <= 5:
        dr = -1 if is_red else 1
        tr = r + dr
        rival_king = 11 if is_red else 4

        while 0 <= tr < 10:
            target = board[tr * 9 + c]
            if target == -1:
                tr += dr
            elif target == rival_king:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = tr
                result[count * 4 + 3] = c
                count += 1
                break
            else:
                break

    return count

cdef int add_cannon_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """炮的合法移动"""
    cdef int count = 0

    cdef int dr, dc, tr, tc, target, i
    cdef bint own_side = board[r * 9 + c] < 7
    cdef bint found_screen

    for i in range(4):
        dr = DIRECTIONS[i][0]
        dc = DIRECTIONS[i][1]
        tr = r + dr
        tc = c + dc
        found_screen = False

        while 0 <= tr < 10 and 0 <= tc < 9:
            target = board[tr * 9 + tc]
            if not found_screen:
                if target == -1:
                    result[count * 4 + 0] = r
                    result[count * 4 + 1] = c
                    result[count * 4 + 2] = tr
                    result[count * 4 + 3] = tc
                    count += 1
                else:
                    found_screen = True
            else:
                if target != -1:
                    if (target < 7) != own_side:
                        result[count * 4 + 0] = r
                        result[count * 4 + 1] = c
                        result[count * 4 + 2] = tr
                        result[count * 4 + 3] = tc
                        count += 1
                    break
            tr += dr
            tc += dc

    return count

cdef int add_pawn_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """卒/兵合法移动"""
    cdef int count = 0
    cdef int tr, tc, target, dc
    cdef bint is_red = board[r * 9 + c] < 7  # 假设红方棋子编号 < 7

    # 1. 前进一步
    if is_red:
        tr = r - 1
        tc = c
        if tr >= 0:
            target = board[tr * 9 + tc]
            if target == -1 or (target < 7) != is_red:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = tr
                result[count * 4 + 3] = tc
                count += 1
    else:
        tr = r + 1
        tc = c
        if tr < 10:
            target = board[tr * 9 + tc]
            if target == -1 or (target < 7) != is_red:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = tr
                result[count * 4 + 3] = tc
                count += 1

    # 2. 过河后可以左右走
    if (is_red and r <= 4) or (not is_red and r >= 5):

        for dc in range(2):  # -1,1
            tr = r
            # 1-2*dc将0，1映射为1，-1
            tc = c + 1 - 2 * dc
            if 0 <= tc < 9:
                target = board[tr * 9 + tc]
                if target == -1 or (target < 7) != is_red:
                    result[count * 4 + 0] = r
                    result[count * 4 + 1] = c
                    result[count * 4 + 2] = tr
                    result[count * 4 + 3] = tc
                    count += 1

    return count

def get_valid_actions(cnp.ndarray[INT32_t, ndim=3, mode='c'] state,
                      int player_to_move,
                      cnp.ndarray[INT32_t, ndim=4, mode='c'] move2action):
    """合法动作（表示版本 2：R2 合法着法制）：
    伪合法走法 + 送将过滤（走完子后己方老将被攻击的着法剔除；
    牵制子移动、两帅照面等均被覆盖——照面经敌方帅射线攻击判定）。"""
    cdef int32_t * board_ptr = <int32_t *> state.data      # [0] 即当前盘面
    cdef int32_t * move2action_ptr = <int32_t *> move2action.data
    cdef int32_t tmp[90]
    cdef int32_t result_buffer[20 * 4]
    cdef int total_buffer[200]
    cdef int count = 0, i, r, c, piece, action_id, dest_num, tr, tc
    cdef int kr = -1, kc = -1, nkr = 0, nkc = 0, king_id
    cdef int32_t back_from, back_to

    with nogil:
        if player_to_move == 0:
            king_id = 4
        else:
            king_id = 11
        for i in range(90):
            if board_ptr[i] == king_id:
                kr = i / 9
                kc = i % 9
                break

        if kr == -1:
            # 防御：无王脏数据，退化为伪合法
            for r in range(10):
                for c in range(9):
                    piece = board_ptr[r * 9 + c]
                    if piece == -1:
                        continue
                    if (0 <= piece <= 6) == (player_to_move == 0):
                        dest_num = get_destinations(piece, board_ptr, r, c, &result_buffer[0])
                        for i in range(dest_num):
                            tr = result_buffer[i * 4 + 2]
                            tc = result_buffer[i * 4 + 3]
                            total_buffer[count] = move2action_ptr[r * 810 + c * 90 + tr * 9 + tc]
                            count += 1
        else:
            for i in range(90):
                tmp[i] = board_ptr[i]
            for r in range(10):
                for c in range(9):
                    piece = tmp[r * 9 + c]
                    if piece == -1:
                        continue
                    if (0 <= piece <= 6) != (player_to_move == 0):
                        continue
                    dest_num = get_destinations(piece, board_ptr, r, c, &result_buffer[0])
                    for i in range(dest_num):
                        tr = result_buffer[i * 4 + 2]
                        tc = result_buffer[i * 4 + 3]
                        # 在临时盘上执行，测己方老将是否被攻
                        back_from = tmp[r * 9 + c]
                        back_to = tmp[tr * 9 + tc]
                        tmp[r * 9 + c] = -1
                        tmp[tr * 9 + tc] = back_from
                        if piece == king_id:
                            nkr = tr
                            nkc = tc
                        else:
                            nkr = kr
                            nkc = kc
                        if not king_attacked(tmp, nkr, nkc, player_to_move):
                            total_buffer[count] = move2action_ptr[r * 810 + c * 90 + tr * 9 + tc]
                            count += 1
                        tmp[r * 9 + c] = back_from
                        tmp[tr * 9 + tc] = back_to

    result = np.empty(count, dtype=np.int32)
    for i in range(count):
        result[i] = total_buffer[i]
    return result

def get_pseudo_valid_actions(cnp.ndarray[INT32_t, ndim=3, mode='c'] state,
                             int player_to_move,
                             cnp.ndarray[INT32_t, ndim=4, mode='c'] move2action):
    """伪合法动作（不做送将过滤）。仅供测试对拍使用。"""
    cdef int count = 0

    cdef cnp.ndarray[INT32_t, ndim=2, mode='c'] board_2d = state[0]
    cdef int32_t * move2action_ptr = <int32_t *> move2action.data
    cdef int32_t * board_ptr = <int32_t *> board_2d.data
    cdef int r, c, piece, action_id, dest_num, tr, tc, i
    cdef int32_t total_buffer[200], result_buffer[20 * 4]

    with nogil:
        for r in range(10):
            for c in range(9):
                piece = board_ptr[r * 9 + c]
                if piece == -1:
                    continue

                # 检查棋子属于当前玩家
                if (player_to_move == 0 and 0 <= piece <= 6) or (player_to_move == 1 and piece >= 7):
                    # 获取该棋子的所有目标位置
                    dest_num = get_destinations(piece, board_ptr, r, c, &result_buffer[0])

                    # 将每个移动转换为动作ID
                    for i in range(dest_num):
                        tr = result_buffer[i * 4 + 2]
                        tc = result_buffer[i * 4 + 3]

                        action_id = move2action_ptr[r * 810 + c * 90 + tr * 9 + tc]

                        total_buffer[count] = action_id
                        count += 1

    result = np.empty(count, dtype=np.int32)
    for i in range(count):
        result[i] = total_buffer[i]

    return result

def is_check(cnp.ndarray[INT32_t, ndim=3, mode='c'] state, int perspective_player):
    """perspective_player 老将是否被将军（直接攻击扫描，与 Python 实现等价）"""
    cdef int32_t * board_ptr = <int32_t *> state.data
    cdef int king_id = 4 if perspective_player == 0 else 11
    cdef int i, kr = -1, kc = -1, attacked = 0
    with nogil:
        for i in range(90):
            if board_ptr[i] == king_id:
                kr = i / 9
                kc = i % 9
                break
        if kr != -1:
            attacked = king_attacked(board_ptr, kr, kc, perspective_player)
    return bool(attacked)

cdef bint king_attacked(INT32_t * board, int kr, int kc, int own) nogil:
    """(kr,kc) 老将（属 own 方）是否被对方攻击。与 chess.py 的 king_attacked 互为镜像：
    车/敌帅射线(首子)、炮射线(架后第二子)、马(反查8马位+蹩腿)、兵(直吃+过河侧吃)。
    士象守而不攻。修改任何一侧必须同步另一侧并跑 tests/test_chess_rules.py 对拍。"""
    cdef int e_chariot, e_king, e_cannon, e_horse, e_pawn, pawn_dir
    cdef bint crossed
    cdef int dr, dc, r, c, p, i, screened

    if own == 0:
        e_chariot, e_king, e_cannon, e_horse, e_pawn = 7, 11, 12, 8, 13
        pawn_dir = 1
        crossed = (kr >= 5)
    else:
        e_chariot, e_king, e_cannon, e_horse, e_pawn = 0, 4, 5, 1, 6
        pawn_dir = -1
        crossed = (kr <= 4)

    for i in range(4):
        dr = DIRECTIONS[i][0]
        dc = DIRECTIONS[i][1]
        r = kr + dr
        c = kc + dc
        screened = 0
        while 0 <= r < 10 and 0 <= c < 9:
            p = board[r * 9 + c]
            if p != -1:
                if not screened:
                    if p == e_chariot or p == e_king:
                        return True
                    screened = 1
                else:
                    if p == e_cannon:
                        return True
                    break
            r += dr
            c += dc

    for i in range(8):
        r = kr - HORSE_MOVES[i][0]
        c = kc - HORSE_MOVES[i][1]
        if r < 0 or r >= 10 or c < 0 or c >= 9:
            continue
        if board[r * 9 + c] != e_horse:
            continue
        if board[(r + HORSE_MOVES[i][2]) * 9 + (c + HORSE_MOVES[i][3])] == -1:
            return True

    r = kr - pawn_dir
    if 0 <= r < 10 and board[r * 9 + kc] == e_pawn:
        return True
    if crossed:
        c = kc - 1
        if 0 <= c < 9 and board[kr * 9 + c] == e_pawn:
            return True
        c = kc + 1
        if 0 <= c < 9 and board[kr * 9 + c] == e_pawn:
            return True
    return False

cdef int get_destinations(int piece, INT32_t * board, int r, int c, INT32_t *result) nogil:
    if piece == 0 or piece == 7:
        return add_rook_dest(board, r, c, result)
    elif piece == 1 or piece == 8:
        return add_horse_dest(board, r, c, result)
    elif piece == 2 or piece == 9:
        return add_bishop_dest(board, r, c, result)
    elif piece == 3 or piece == 10:
        return add_advisor_dest(board, r, c, result)
    elif piece == 4 or piece == 11:
        return add_king_dest(board, r, c, result)
    elif piece == 5 or piece == 12:
        return add_cannon_dest(board, r, c, result)
    else:
        return add_pawn_dest(board, r, c, result)

def get_destinations_for_piece(cnp.ndarray[INT32_t, ndim=3, mode='c'] state,
                               int player_to_move,
                               cnp.ndarray[INT32_t, ndim=4, mode='c'] move2action,
                               int r,
                               int c):
    """获取一个棋子的合法动作"""
    cdef cnp.ndarray[INT32_t, ndim=2, mode='c'] board_2d = state[0]
    cdef int32_t * move2action_ptr = <int32_t *> move2action.data
    cdef int32_t * board_ptr = <int32_t *> board_2d.data
    cdef int piece, action_id, dest_num, tr, tc, i
    cdef int32_t  result_buffer[20 * 4]

    piece = board_ptr[r * 9 + c]
    if piece == -1:
        return []

    if (0 <= piece <= 6) != (player_to_move == 0):
        return []

    result = []
    dest_num = get_destinations(piece, board_ptr, r, c, &result_buffer[0])
    # 将每个移动转换为动作ID
    for i in range(dest_num):
        tr = result_buffer[i * 4 + 2]
        tc = result_buffer[i * 4 + 3]

        action_id = move2action_ptr[r * 810 + c * 90 + tr * 9 + tc]
        result.append(action_id)

    return result


# ======================================================================================
# convert_v2：feature version 2 的 (26,10,9) 网络输入面，逐位等价于 chess.py 的
# ChineseChess._convert_python(version=2)。把 35 个 numpy 调用的固定开销 (~31µs/叶子)
# 压成一个 C 循环（~2µs）；并发压力下实测吞吐 +60~75%，见 tests/drills/env_cost_ceiling.py。
#
# 面定义（与 Python 侧 docstring 一致，改一处必须同步另一处并跑
# tests/test_convert_cython_exact.py 逐位对拍）：
#   [0~13] 棋子 one-hot（0-6 己方、7-13 对手，视角已归一化）
#   [14/15] 上一手（对手走）起点/落点   [16/17] 上上手（我方走）起点/落点
#   [18] 被将军  [19/20] 窗口内重复一次/两次  [21] 未吃子时钟/120
#   [22/23] 己方/对手兵力价值 /40  [24/25] 双方老将位
#
# 线程安全：全部中间量都是**栈上局部数组**（st/board/v1/v2/hist），不碰模块级状态——
# 本模块 freethreading_compatible=True 的前提就是"模块级只有只读常量表"。
# ======================================================================================
def convert_v2(cnp.ndarray[INT32_t, ndim=3, mode='c'] state, int current_player):
    """state (7,10,9) int32 → (26,10,9) float32；与 Python 参考实现逐位相等。"""
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode='c'] out = np.zeros((26, 10, 9), dtype=np.float32)
    cdef int32_t *sp = <int32_t *> state.data
    cdef cnp.float32_t *op = <cnp.float32_t *> out.data
    cdef int st[630]        # 视角归一化后的 7 层历史（仍是原始 id，-1=空）
    cdef int board[90]      # 当前盘面：-1→14、对手换色
    cdef int v1[90]         # 上一层，同规则
    cdef int v2[90]         # 上上层，同规则
    cdef int hist[15]
    cdef int L = 90
    cdef int l, r, c, i, k, v, pv, clock
    cdef bint flip = current_player == 1
    cdef bint eq1, eq2, eq4
    cdef double m_own, m_opp, MV[7]
    MV[0] = 9.0; MV[1] = 4.0; MV[2] = 0.0; MV[3] = 0.0
    MV[4] = 0.0; MV[5] = 4.5; MV[6] = 1.0

    with nogil:
        # ---- 1) 视角归一化：对手视角时上下+左右翻转（一次做完，后面不再管 flip）----
        for l in range(7):
            for r in range(10):
                for c in range(9):
                    st[l * L + r * 9 + c] = sp[l * L + (9 - r) * 9 + (8 - c) if flip else l * L + r * 9 + c]

        # ---- 2) 三张盘面：空位记 14，对手视角时换色（0-6 <-> 7-13）----
        for i in range(L):
            for l in range(3):
                v = st[l * L + i]
                if v == -1:
                    v = 14
                elif flip and v != 14:
                    v = v + 7 if v < 7 else v - 7
                if l == 0:
                    board[i] = v
                elif l == 1:
                    v1[i] = v
                else:
                    v2[i] = v

        # ---- 3) 面 0~13：棋子 one-hot ----
        for k in range(15):
            hist[k] = 0
        for i in range(L):
            hist[board[i]] += 1
            if board[i] < 14:
                op[board[i] * L + i] = 1.0

        # ---- 4) 面 14~17：上一手/上上手的起点落点（差分恢复，象棋无易位）----
        for i in range(L):
            pv = board[i]
            if v1[i] >= 7 and v1[i] <= 13 and pv == 14:
                op[14 * L + i] = 1.0             # 上一手起点（走完变空）
            if pv >= 7 and pv <= 13 and pv != v1[i]:
                op[15 * L + i] = 1.0             # 上一手落点
            if v2[i] >= 0 and v2[i] <= 6 and v1[i] == 14:
                op[16 * L + i] = 1.0             # 上上手起点
            if v1[i] >= 0 and v1[i] <= 6 and v1[i] != v2[i]:
                op[17 * L + i] = 1.0             # 上上手落点

        # ---- 5) 面 18：被将军（广播）。与 is_check 同口径，用**原始** state 判 ----
        v = 4 if current_player == 0 else 11
        r = -1; c = -1
        for i in range(L):
            if sp[i] == v:
                r = i // 9; c = i % 9
                break
        if r != -1 and king_attacked(sp, r, c, current_player):
            for i in range(L):
                op[18 * L + i] = 1.0

        # ---- 6) 面 19/20：窗口内重复（相隔偶数层；层0==层1 视为"无历史"不启动）----
        eq1 = True
        for i in range(L):
            if st[i] != st[L + i]:
                eq1 = False
                break
        if not eq1:
            eq2 = True; eq4 = True
            for i in range(L):
                if st[i] != st[2 * L + i]:
                    eq2 = False
                if st[i] != st[4 * L + i]:
                    eq4 = False
                if not eq2 and not eq4:
                    break
            if eq2 or eq4:
                for i in range(L):
                    op[19 * L + i] = 1.0
                if eq2 and eq4:
                    for i in range(L):
                        op[20 * L + i] = 1.0

        # ---- 7) 面 21：未吃子时钟（广播，DRAW_NO_CAPTURE_PLIES=120）----
        for i in range(L):
            clock = st[6 * L + i]
            op[21 * L + i] = <cnp.float32_t>(<double>clock / 120.0)

        # ---- 8) 面 22/23：兵力价值合计 /40（广播）----
        m_own = 0.0; m_opp = 0.0
        for k in range(7):
            m_own += hist[k] * MV[k]
            m_opp += hist[k + 7] * MV[k]
        for i in range(L):
            op[22 * L + i] = <cnp.float32_t>(m_own / 40.0)
            op[23 * L + i] = <cnp.float32_t>(m_opp / 40.0)

        # ---- 9) 面 24/25：双方老将位（单格面）----
        for i in range(L):
            if board[i] == 4:
                op[24 * L + i] = 1.0
            if board[i] == 11:
                op[25 * L + i] = 1.0
    return out
