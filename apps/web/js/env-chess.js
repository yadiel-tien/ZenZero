/* ZenZero 中国象棋引擎（纯逻辑，与 core/env/chess.py 逐位对齐）
 * state: Int8Array(7*9*10)，层0=当前盘面(-1空, 0-6红 车马相仕帅炮兵, 7-13黑)，
 * 层1-5=历史盘面，层6=未吃子回合计数。
 * 动作编码顺序与 ChineseChess.init_class_dicts 完全一致（车路横竖 -> 士 -> 象 -> 马），共2086个。
 */
(function (global) {
  'use strict';

  const ROWS = 10, COLS = 9, N_ACTIONS = 2086, CELL = ROWS * COLS;

  const PIECE_CHARS = ['车', '马', '象', '士', '帅', '炮', '兵'];
  // id -> {red: bool, kind: 0..6}
  function pieceSide(p) { return p >= 0 && p < 7 ? 0 : (p >= 7 && p < 14 ? 1 : -1); }

  const ADVISOR_MOVES = [
    [0, 3, 1, 4], [0, 5, 1, 4], [1, 4, 0, 3], [1, 4, 0, 5], [1, 4, 2, 3], [1, 4, 2, 5], [2, 3, 1, 4],
    [2, 5, 1, 4], [9, 3, 8, 4], [9, 5, 8, 4], [8, 4, 9, 3], [8, 4, 9, 5], [8, 4, 7, 3], [8, 4, 7, 5],
    [7, 3, 8, 4], [7, 5, 8, 4]
  ];
  const BISHOP_MOVES_RED_SIDE = [
    [0, 2, 2, 0], [0, 2, 2, 4], [0, 6, 2, 4], [0, 6, 2, 8], [2, 0, 0, 2], [2, 0, 4, 2], [2, 4, 0, 2],
    [2, 4, 4, 2], [2, 4, 0, 6], [2, 4, 4, 6], [2, 8, 0, 6], [2, 8, 4, 6], [4, 2, 2, 0], [4, 2, 2, 4],
    [4, 6, 2, 4], [4, 6, 2, 8]
  ];
  const BISHOP_MOVES = BISHOP_MOVES_RED_SIDE.concat(
    BISHOP_MOVES_RED_SIDE.map(([r, c, tr, tc]) => [9 - r, c, 9 - tr, tc])
  );
  const HORSE_JUMPS = [
    [[-2, -1], [-1, 0]], [[-2, 1], [-1, 0]], [[-1, -2], [0, -1]], [[-1, 2], [0, 1]],
    [[1, -2], [0, -1]], [[1, 2], [0, 1]], [[2, -1], [1, 0]], [[2, 1], [1, 0]]
  ];
  const HORSE_DIRS = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]];

  // ---- 动作表（类级，只建一次） ----
  let MOVE2ACTION = null;   // Int16Array(10*9*10*9), -1 = 非法move
  let ACTION2MOVE = null;   // Int16Array(2086*4)

  function buildTables() {
    if (MOVE2ACTION) return;
    MOVE2ACTION = new Int16Array(ROWS * COLS * ROWS * COLS).fill(-1);
    ACTION2MOVE = new Int16Array(N_ACTIONS * 4);
    const moves = [];
    const push = (r, c, tr, tc) => moves.push([r, c, tr, tc]);
    // 1) 垂直/水平移动（车的候选集，覆盖车/马/炮/兵/帅等的直线走法索引基础）
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        for (let tr = 0; tr < ROWS; tr++) if (tr !== r) push(r, c, tr, c);
        for (let tc = 0; tc < COLS; tc++) if (tc !== c) push(r, c, r, tc);
      }
    }
    // 2) 士
    for (const m of ADVISOR_MOVES) push(m[0], m[1], m[2], m[3]);
    // 3) 象
    for (const m of BISHOP_MOVES) push(m[0], m[1], m[2], m[3]);
    // 4) 马
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        for (const [dr, dc] of HORSE_DIRS) {
          const tr = r + dr, tc = c + dc;
          if (tr >= 0 && tr < ROWS && tc >= 0 && tc < COLS) push(r, c, tr, tc);
        }
      }
    }
    if (moves.length !== N_ACTIONS) throw new Error(`action table size ${moves.length} != ${N_ACTIONS}`);
    moves.forEach(([r, c, tr, tc], a) => {
      MOVE2ACTION[((r * COLS + c) * ROWS + tr) * COLS + tc] = a;
      ACTION2MOVE[a * 4] = r; ACTION2MOVE[a * 4 + 1] = c; ACTION2MOVE[a * 4 + 2] = tr; ACTION2MOVE[a * 4 + 3] = tc;
    });
  }

  const mv = (state, l, r, c) => state[l * CELL + r * COLS + c];
  function move2action(r, c, tr, tc) { return MOVE2ACTION[((r * COLS + c) * ROWS + tr) * COLS + tc]; }
  function action2move(a) { return [ACTION2MOVE[a * 4], ACTION2MOVE[a * 4 + 1], ACTION2MOVE[a * 4 + 2], ACTION2MOVE[a * 4 + 3]]; }

  // ---- 走法生成（对应各 _get_*_dest） ----
  function getRookDest(board, r, c) {
    const dest = [], own = board[r * COLS + c] < 7;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      let tr = r, tc = c;
      for (;;) {
        tr += dr; tc += dc;
        if (!(tr >= 0 && tr < ROWS && tc >= 0 && tc < COLS)) break;
        const t = board[tr * COLS + tc];
        if (t === -1) dest.push([tr, tc]);
        else if ((t < 7) !== own) { dest.push([tr, tc]); break; }
        else break;
      }
    }
    return dest;
  }

  function getHorseDest(board, r, c) {
    const dest = [], own = board[r * COLS + c] < 7;
    for (const [[dr, dc], [br, bc]] of HORSE_JUMPS) {
      const tr = r + dr, tc = c + dc;
      if (tr >= 0 && tr < ROWS && tc >= 0 && tc < COLS) {
        if (board[(r + br) * COLS + (c + bc)] !== -1) continue; // 蹩马腿
        const t = board[tr * COLS + tc];
        if (t === -1 || (t < 7) !== own) dest.push([tr, tc]);
      }
    }
    return dest;
  }

  function getBishopDest(board, r, c) {
    const dest = [], own = board[r * COLS + c] < 7;
    for (const [fr, fc, tr, tc] of BISHOP_MOVES) {
      if (fr !== r || fc !== c) continue;
      if (board[((fr + tr) >> 1) * COLS + ((fc + tc) >> 1)] !== -1) continue; // 塞象眼
      const t = board[tr * COLS + tc];
      if (t === -1 || (t < 7) !== own) dest.push([tr, tc]);
    }
    return dest;
  }

  function getAdvisorDest(board, r, c) {
    const dest = [], own = board[r * COLS + c] < 7;
    for (const [fr, fc, tr, tc] of ADVISOR_MOVES) {
      if (fr !== r || fc !== c) continue;
      const t = board[tr * COLS + tc];
      if (t === -1 || (t < 7) !== own) dest.push([tr, tc]);
    }
    return dest;
  }

  function getKingDest(board, r, c) {
    const dest = [];
    const isRed = board[r * COLS + c] === 4;
    const lo = isRed ? 7 : 0, hi = isRed ? 9 : 2;
    const rival = isRed ? 11 : 4;
    for (const [tr, tc] of [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]) {
      if (tc < 3 || tc > 5) continue;
      if (tr < lo || tr > hi) continue;
      const t = board[tr * COLS + tc];
      if (t === -1 || ((t < 7) !== isRed)) dest.push([tr, tc]);
    }
    // 白脸将
    const dr = isRed ? -1 : 1;
    let tr = r + dr;
    while (tr >= 0 && tr <= 9) {
      const t = board[tr * COLS + c];
      if (t === -1) tr += dr;
      else if (t === rival) { dest.push([tr, c]); break; }
      else break;
    }
    return dest;
  }

  function getCannonDest(board, r, c) {
    const dest = [], own = board[r * COLS + c] < 7;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      let screen = false, tr = r + dr, tc = c + dc;
      while (tr >= 0 && tr <= 9 && tc >= 0 && tc <= 8) {
        const t = board[tr * COLS + tc];
        if (screen) {
          if (t === -1) { /* 空位不可走 */ }
          else if ((t < 7) !== own) { dest.push([tr, tc]); break; }
          else break;
        } else {
          if (t === -1) dest.push([tr, tc]);
          else screen = true;
        }
        tr += dr; tc += dc;
      }
    }
    return dest;
  }

  function getPawnDest(board, r, c) {
    const dest = [];
    const isRed = board[r * COLS + c] === 6;
    const dr = isRed ? -1 : 1;
    const crossed = isRed ? (r <= 4) : (r > 4);
    const cands = crossed ? [[r + dr, c], [r, c - 1], [r, c + 1]] : [[r + dr, c]];
    for (const [tr, tc] of cands) {
      if (tr >= 0 && tr < ROWS && tc >= 0 && tc < COLS) {
        const t = board[tr * COLS + tc];
        if (t === -1 || ((t < 7) !== isRed)) dest.push([tr, tc]);
      }
    }
    return dest;
  }

  // 棋子种类 0-6 -> dest 函数；黑方 id-7 复用
  const DEST_FUNC = [getRookDest, getHorseDest, getBishopDest, getAdvisorDest, getKingDest, getCannonDest, getPawnDest];
  function getDest(board, r, c) {
    const p = board[r * COLS + c];
    if (p === -1) return [];
    return DEST_FUNC[p < 7 ? p : p - 7](board, r, c);
  }

  function ownerOf(p) { return p >= 7 ? 1 : (p >= 0 ? 0 : -1); }

  function getValidActions(state, playerToMove) {
    buildTables();
    const board = state.subarray(0, CELL);
    const acts = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const p = board[r * COLS + c];
        if (playerToMove === 0 ? (p >= 0 && p <= 6) : (p >= 7 && p <= 13)) {
          for (const [tr, tc] of getDest(board, r, c)) {
            const a = move2action(r, c, tr, tc);
            if (a >= 0) acts.push(a);
          }
        }
      }
    }
    return acts;
  }

  function getValidActionsFromPos(state, r, c) {
    buildTables();
    const board = state.subarray(0, CELL);
    const acts = [];
    for (const [tr, tc] of getDest(board, r, c)) {
      const a = move2action(r, c, tr, tc);
      if (a >= 0) acts.push(a);
    }
    return acts;
  }

  function virtualStep(state, action) {
    buildTables();
    const next = state.slice();
    const [r, c, tr, tc] = action2move(action);
    const attacker = mv(next, 0, r, c);
    const target = mv(next, 0, tr, tc);
    if (target === -1) {
      for (let i = 6 * CELL; i < 7 * CELL; i++) next[i] += 1;
    } else {
      for (let i = 6 * CELL; i < 7 * CELL; i++) next[i] = 0;
    }
    for (let l = 5; l >= 1; l--) next.set(state.subarray((l - 1) * CELL, l * CELL), l * CELL);
    next[0 * CELL + r * COLS + c] = -1;
    next[0 * CELL + tr * COLS + tc] = attacker;
    return next;
  }

  function isCheck(state, perspectivePlayer) {
    const kingId = perspectivePlayer === 0 ? 4 : 11;
    const acts = getValidActions(state, 1 - perspectivePlayer);
    const board = state.subarray(0, CELL);
    for (const a of acts) {
      const [, , tr, tc] = action2move(a);
      if (board[tr * COLS + tc] === kingId) return true;
    }
    return false;
  }

  function isCheckmate(state, player) {
    const acts = getValidActions(state, player);
    for (const a of acts) {
      if (!isCheck(virtualStep(state, a), player)) return false;
    }
    return true;
  }

  // 胜负判定，相对刚落子方: 1胜 0平 -1负 2未分
  function checkWinner(state, playerJustMoved) {
    const board = state.subarray(0, CELL);
    let hasRedKing = false, hasBlackKing = false, hasAttacker = false;
    const ATTACKERS = new Set([0, 1, 5, 6, 7, 8, 12, 13]);
    for (let i = 0; i < CELL; i++) {
      const p = board[i];
      if (p === 4) hasRedKing = true;
      else if (p === 11) hasBlackKing = true;
      else if (ATTACKERS.has(p)) hasAttacker = true;
    }
    if (!hasRedKing) return playerJustMoved === 1 ? 1 : -1;
    if (!hasBlackKing) return playerJustMoved === 0 ? 1 : -1;

    // 长将判负：无吃子且 盘面0==2==4、1==3 时若对方被将军，落子方负
    const counter = mv(state, 6, 0, 0);
    if (counter !== 0) {
      const d0 = diffEq(state, 0, 1), d1 = diffEq(state, 1, 2), d2 = diffEq(state, 2, 3),
        d3 = diffEq(state, 3, 4), d4 = diffEq(state, 4, 5);
      // Python: diffs[i]=equal(state[i],state[i+1]); all(d0==d2)&&all(d0==d4)&&all(d1==d3)
      if (maskEq(d0, d2) && maskEq(d0, d4) && maskEq(d1, d3)) {
        if (isCheck(state, 1 - playerJustMoved)) return -1;
      }
    }
    if (counter >= 50) return 0;
    return hasAttacker ? 2 : 0;
  }
  function diffEq(state, i, j) {
    const m = new Uint8Array(CELL);
    for (let k = 0; k < CELL; k++) m[k] = state[i * CELL + k] === state[j * CELL + k] ? 1 : 0;
    return m;
  }
  function maskEq(a, b) {
    for (let k = 0; k < CELL; k++) if (a[k] !== b[k]) return false;
    return true;
  }

  const INITIAL_BOARD = [
    7, 8, 9, 10, 11, 10, 9, 8, 7,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 12, -1, -1, -1, -1, -1, 12, -1,
    13, -1, 13, -1, 13, -1, 13, -1, 13,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
    6, -1, 6, -1, 6, -1, 6, -1, 6,
    -1, 5, -1, -1, -1, -1, -1, 5, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 1, 2, 3, 4, 3, 2, 1, 0
  ];

  function initialState() {
    buildTables();
    const state = new Int8Array(7 * CELL);
    for (let l = 0; l < 6; l++) state.set(INITIAL_BOARD, l * CELL);
    return state;
  }

  // 中文记录，如 炮二平五。使用走子前的 board（对应 Python 的 state[1]）
  function describeMove(preBoardFlat, action) {
    const [r, c, tr, tc] = action2move(action);
    const pieceId = preBoardFlat[r * COLS + c];
    if (pieceId < 0 || pieceId > 13) return '';
    const isRed = pieceId < 7;
    const kind = isRed ? pieceId : pieceId - 7;
    const BIG = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
    let desc = PIECE_CHARS[kind];
    if (isRed) {
      desc += BIG[8 - c];
      if (r === tr) desc += '平' + BIG[8 - tc];
      else if (r > tr) desc += '进' + (c === tc ? BIG[r - tr - 1] : BIG[8 - tc]);
      else desc += '退' + (c === tc ? BIG[tr - r - 1] : BIG[8 - tc]);
    } else {
      desc += (c + 1);
      if (r === tr) desc += '平' + (tc + 1);
      else if (r > tr) desc += '退' + (c === tc ? (r - tr - 1) + 1 : tc + 1);
      else desc += '进' + (c === tc ? (tr - r - 1) + 1 : tc + 1);
    }
    // 黑方纵线：Python 用 black_col[c]=c+1 与 '平{tc+1}'；进/退距离为 str(n)（0起算+1处理见Python: str(black_col[x]) 其中索引差-1）
    return desc;
  }

  buildTables();
  global.ChessEnv = {
    ROWS, COLS, N_ACTIONS, CELL,
    initialState, getValidActions, getValidActionsFromPos, virtualStep,
    isCheck, isCheckmate, checkWinner, move2action, action2move,
    describeMove, ownerOf, pieceSide,
    stateToNested(state) {
      const out = [];
      for (let l = 0; l < 7; l++) {
        const layer = [];
        for (let r = 0; r < ROWS; r++) {
          const row = [];
          for (let c = 0; c < COLS; c++) row.push(state[l * CELL + r * COLS + c]);
          layer.push(row);
        }
        out.push(layer);
      }
      return out;
    },
    boardLayer(state) { return state.subarray(0, CELL); }
  };
})(typeof window !== 'undefined' ? window : globalThis);
