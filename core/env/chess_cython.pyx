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
    """Cython版本的合法动作获取"""
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

    cdef cnp.ndarray[INT32_t, ndim=1] result = np.empty(count, dtype=np.int32)
    for i in range(count):
        result[i] = total_buffer[i]

    return result

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
