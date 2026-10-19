/* ZenZero 五子棋引擎（纯逻辑，与 core/env/gomoku.py 对齐）
 * 内部用 board: Int8Array(225)，0空 1黑(玩家0,先行) 2白(玩家1)。
 * 发送给推理服务的 state 语义：layer0 = 当前行棋方棋子，layer1 = 对方（每步交换）。
 */
(function (global) {
  'use strict';

  const SIZE = 15, N_ACTIONS = SIZE * SIZE;

  function initialState() { return new Int8Array(N_ACTIONS); }

  const move2action = (r, c) => r * SIZE + c;
  const action2move = (a) => [Math.floor(a / SIZE), a % SIZE];

  function getValidActions(board) {
    const acts = [];
    for (let i = 0; i < N_ACTIONS; i++) if (board[i] === 0) acts.push(i);
    return acts;
  }

  // 落子后返回连五的棋子坐标；否则 []（对应 get_win_stones，用刚落子方棋子）
  function getWinStones(board, action, stone) {
    const [h0, w0] = action2move(action);
    for (const [dh, dw] of [[1, 0], [1, 1], [0, 1], [-1, 1]]) {
      const stones = [[h0, w0]];
      for (const dir of [-1, 1]) {
        for (let step = 1; step < 5; step++) {
          const i = h0 + step * dh * dir, j = w0 + step * dw * dir;
          if (i >= 0 && i < SIZE && j >= 0 && j < SIZE && board[i * SIZE + j] === stone) {
            stones.push([i, j]);
            if (stones.length === 5) return stones;
          } else break;
        }
      }
    }
    return [];
  }

  function isDraw(board) {
    for (let i = 0; i < N_ACTIONS; i++) if (board[i] === 0) return false;
    return true;
  }

  // 发送给服务端的 [2,15,15] 数组：playerToMove 视角
  function stateToNested(board, playerToMove) {
    const stone0 = playerToMove === 0 ? 1 : 2;
    const out = [
      [], // 行棋方
      []  // 对方
    ];
    for (let r = 0; r < SIZE; r++) {
      const la = [], lb = [];
      for (let c = 0; c < SIZE; c++) {
        const v = board[r * SIZE + c];
        la.push(v === stone0 ? 1 : 0);
        lb.push(v !== 0 && v !== stone0 ? 1 : 0);
      }
      out[0].push(la);
      out[1].push(lb);
    }
    return out;
  }

  global.GomokuEnv = {
    SIZE, N_ACTIONS, initialState, move2action, action2move,
    getValidActions, getWinStones, isDraw, stateToNested
  };
})(typeof window !== 'undefined' ? window : globalThis);
