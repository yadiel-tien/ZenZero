"""模型对局强度测量（drill，不是单测）：给定两个 (档位, 序号) 打 N 局，报胜/和/负与得分率。

为什么需要：训练侧的"评估胜率"是**新模型 vs 当前守擂**，每代只比上代强一点点，噪声 ±5%；
想知道"整体练到什么程度"必须跨代/跨档直接对打。

用法：
    python -m tests.drills.model_match --a s:34 --b s:24 --games 24 --sims 200
    python -m tests.drills.model_match --a s:34 --b m:498 --games 24 --sims 200

设计：
- 两个 InferenceEngine 常驻，对局里所有落子都走它们的 request_queue（和训练同一条推理路径）；
- 每步用 run_pipelined（K 条虚拟损失 playout 在飞），和自博弈完全一致，只是不做噪声/温度采样，
  模拟次数固定（不随换代爬坡），保证可比；
- 黑白交替，抵消先手优势；和棋算 0.5；
- 得分率的标准误按对局数估计，输出里直接给 ±，避免把噪声当进步。
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np  # noqa: E402

from core.env.functions import get_class  # noqa: E402
from core.mcts.deepMcts import NeuronMCTS  # noqa: E402
from services.inference.engine import InferenceEngine  # noqa: E402
from services.train.selfplay import run_mcts  # noqa: E402


def _parse(spec: str) -> tuple[str, int]:
    prof, _, idx = spec.partition(':')
    return (prof, int(idx)) if idx else ('m', int(prof))


def play_one(env_class, engines: dict[str, InferenceEngine], first: str, second: str,
             sims: int, batch_leaves: int, verbose: bool) -> int:
    """返回 0=first 胜 / 1=second 胜 / -1 和棋"""
    env = env_class()
    seat = {0: first, 1: second}          # player_to_move -> 哪一方
    max_steps = 400
    while not env.terminated and env.steps < max_steps:
        side = seat[env.player_to_move]
        eng = engines[side]
        mcts = NeuronMCTS.make_queue_mcts(
            infer_q=eng.request_queue, env_class=env_class, state=env.state,
            last_action=env.last_action, player_to_move=env.player_to_move,
            is_self_play=False, planes_version=eng.feature_version)
        run_mcts(mcts, sims, batch_leaves)
        pi = mcts.get_pi(0.1)             # 低温度：接近最强手，但保留一点随机性避免僵局
        action = int(np.random.choice(len(pi), p=pi / pi.sum()))
        mcts.shutdown()
        env.step(action)
        if verbose:
            print(f'\r  step {env.steps:3d}', end='', flush=True)
    if verbose:
        print('\r', end='')
    if not env.terminated:
        return -1                          # 步数上限 = 和
    # env.winner 约定：0/1 = 胜方；-1 = 和
    winner = getattr(env, 'winner', -1)
    if winner == -1:
        return -1
    return 0 if seat[winner] == first else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default='ChineseChess')
    ap.add_argument('--a', default='s:34', help='档位:序号，如 s:34 / m:498')
    ap.add_argument('--b', default='s:24')
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--sims', type=int, default=200)
    ap.add_argument('--leaves', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    env_class = get_class(args.game)
    pa, ia = _parse(args.a)
    pb, ib = _parse(args.b)
    print(f'对局：{args.a}  vs  {args.b}   每步 {args.sims} sims × K={args.leaves}，{args.games} 局，黑白交替')

    engines = {}
    for tag, (prof, idx) in (('A', (pa, ia)), ('B', (pb, ib))):
        eng = InferenceEngine(idx, args.game, profile=prof)
        eng.start()
        engines[tag] = eng
        print(f'  引擎 {tag}: profile={prof} index={idx} name={eng.name} fv={eng.feature_version}')

    win = lose = draw = 0
    t0 = time.time()
    try:
        for g in range(args.games):
            first, second = ('A', 'B') if g % 2 == 0 else ('B', 'A')
            r = play_one(env_class, engines, first, second, args.sims, args.leaves, args.verbose)
            if r == -1:
                draw += 1
            elif (r == 0 and first == 'A') or (r == 1 and second == 'A'):
                win += 1
            else:
                lose += 1
            score = win + draw / 2
            print(f'  第 {g + 1:2d} 局（A 执{"先" if first == "A" else "后"}）→ '
                  f'A 累计 {win}胜 {lose}负 {draw}和  得分率 {score / (g + 1):.1%}')
    finally:
        for eng in engines.values():
            eng.shutdown()

    n = win + lose + draw
    score = win + draw / 2
    rate = score / max(n, 1)
    stderr = math.sqrt(max(rate * (1 - rate), 1e-9) / max(n, 1))     # 粗略标准误
    print(f'\n结果：A({args.a}) 对 B({args.b})：{win} 胜 / {lose} 负 / {draw} 和')
    print(f'得分率 {rate:.1%} ± {stderr:.1%}（n={n}，和棋记 0.5），用时 {time.time() - t0:.0f}s')
    if rate - 2 * stderr > 0.5:
        print('判定：A 明显更强（超出 2σ）')
    elif rate + 2 * stderr < 0.5:
        print('判定：B 明显更强（超出 2σ）')
    else:
        print('判定：差距不显著（落在 ±2σ 内）—— 要更强结论请加局数')


if __name__ == '__main__':
    main()
