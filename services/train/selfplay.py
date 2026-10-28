import json
import os
import threading
import time
from datetime import datetime
from concurrent.futures import (FIRST_COMPLETED, ThreadPoolExecutor, as_completed,
                                wait)

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from core.env.functions import get_class
from services.inference.functions import get_checkpoint_path
from core.utils.logger import get_logger
from core.utils.npz_loader import NPZLoader
from core.utils.replay import ReplayBuffer
from core.utils.config import game_name, settings, CONFIG, get_profile
from core.mcts.deepMcts import NeuronMCTS
from services.train import heartbeat as hb
from services.inference.engine import InferenceEngine
from services.inference.train_server import TrainServer
from core.network.functions import read_latest_index, save_best_index, read_best_index
import random


def ramp_simulations(iteration: int, sims: dict | None = None) -> int:
    """每手模拟次数：按**绝对代**从 start 爬到 max（默认 400 → 800，50 代到位）。

    旧实现按"本次启动内第几轮 ÷ max_iters(500)"爬坡，等于每次重启都退回 200，
    实测 perf 行永远是 @200sims/手，爬坡从未生效。改按绝对代后重启会续上。
    纯函数便于测试。
    """
    cfg = sims or settings.get('sims') or {}
    start = int(cfg.get('start', 400))
    top = int(cfg.get('max', 800))
    gens = max(1, int(cfg.get('ramp_generations', 50)))
    if top <= start:
        return top
    frac = min(1.0, max(0, int(iteration)) / gens)
    return int(round(start + (top - start) * frac))


def run_mcts(mcts: NeuronMCTS, n_simulations: int, batch_leaves: int) -> None:
    """按批量叶子数选 MCTS 驱动方式。

    K>1 走流水线（始终 K 条在飞，队列不空、引擎不饿）；K==1 走老的同步单请求
    （人机对战要的是单步延迟，流水线没好处还多一层信号量）。"""
    if batch_leaves > 1:
        mcts.run_pipelined(n_simulations, batch_leaves=batch_leaves)
    else:
        mcts.run(n_simulations)


class SelfPlayManager:
    def __init__(self, n_workers: int, train_server: TrainServer):
        self.logger = get_logger('selfplay')
        self.n_workers = n_workers
        # 进程内训练服务器(推理引擎)，直接方法调用与队列请求，无需socket
        self.train_server = train_server
        # 与 TrainServer 共享同一 buffer 实例（fit 期间 selfplay 天然挂起，无并发读写；
        # 旧版各自建一份，象棋配置下双份 numpy 大数组白占 ~9GB 内存）
        self.buffer = train_server.buffer
        self.buffer.load()
        # 留出验证集（add 时按 val_holdout_ratio 划出，永不参与训练）
        self.val_buffer = ReplayBuffer(max(8_000, settings['buffer_size'] // 16), 2048,
                                       file_name=CONFIG['val_buffer_name'])
        self.val_buffer.load()
        self.env_class = get_class(game_name)
        self.best_index = read_best_index()
        self.pool = ThreadPoolExecutor(self.n_workers)
        self.opening_buffer = NPZLoader("core/env/chess_step10.npz") if game_name == 'ChineseChess' else None
        self.endgame_buffer = NPZLoader("core/env/chess_step50.npz") if game_name == 'ChineseChess' else None
        self.debug_logger = get_logger('debug')
        # 优雅停止：收到信号/HTTP 停止后，把手上正在跑的这波局打完（不再派新局）、
        # 存盘、退出——而不是让 SIGTERM 硬杀丢掉整轮数据。
        self.stop_event = threading.Event()
        # 两种结束方式：
        #   stop_event            = 立即结束（取消没开跑的局、收完在跑的、存盘退出）
        #   stop_after_round=True = 等本轮结束（自博弈→fit→评估 全做完，再退出，不丢本轮成果）
        self.stop_after_round = False

    def run(self, n_games: int) -> None:
        """训练入口
        :param n_games: 每轮的游戏数量"""
        iteration = read_latest_index()
        iteration = 1 if iteration == -1 else iteration + 1

        # max_iters 语义 = **本次启动要训练的轮数**（审计修正）。
        # 旧实现是绝对世代上限：象棋 v2 从 499 重启、上限 500，只会跑 1 轮；
        # 五子棋跑满再启动，一轮都不跑直接退出。
        hb.write(force=True, phase='idle', note='run start')
        for round_index in range(settings['max_iters']):
            if self.stop_event.is_set():
                self.logger.info('收到停止请求，不再开新轮，收尾退出。')
                return
            self.logger.info(f'Starting selfplay, iteration: {iteration}, best index: {self.best_index}, '
                             f'round {round_index + 1}/{settings["max_iters"]} of this run.')
            # 先保证buffer足够大。旧阈值按"写入增强×N份"的容量口径计，
            # 在线增强后 capacity=不同局面数——除以对称群大小，保持开局填仓的局面多样性等量
            # 填仓下限 = 一个 batch：仓库喂不满一步训练，fit 必炸（NoEnoughData），
            # 报错却像内部故障——小 buffer_size 配置（试验/小网）时这是必经防线
            min_fill = max(self.buffer.capacity * 0.4 // max(len(self.buffer.sym_ids), 1),
                           self.buffer.batch_size)
            fill = {'games': 0, 'samples': 0, 'sims': 0, 'wall': 0.0}
            while self.buffer.size < min_fill and not self.stop_event.is_set():
                t_fill = time.time()
                # 填仓用满线程池（旧值 50 在 100 线程池上直接空转一半）
                self.self_play(iteration=iteration, n_games=self.n_workers,
                               round_index=round_index, phase='fill')
                d = self._last_sp
                fill['wall'] += time.time() - t_fill
                fill['games'] += d['games']
                fill['samples'] += d['samples']
                fill['sims'] += d['sims']
                self.logger.info(f'Collecting data.Current buffer size: {self.buffer.size}.')

            # 开始selfplay。新数据会进行积累，直到新模型产生。
            self.train_server.reset_statistic()  # 重置engine的推理统计数据
            n_data = self.self_play(iteration=iteration, n_games=n_games, round_index=round_index)
            sp = self._last_sp
            if self.stop_event.is_set():
                # 数据已经存盘（self_play 结尾 buffer.save()）——不做 fit/eval 直接退出，
                # 半成品模型不该进 checkpoint、更不该拿去做晋升评估
                self.logger.info(f'收到停止请求：本轮已收 {sp["games"]} 局 / {sp["samples"]} 样本并存盘，'
                                 f'跳过 fit 与评估，退出。')
                return

            # 直接调用训练服务器进行模型训练，并保存参数
            # buffer 已与 server 共享（内存即真源，刚 save 过）→ 免重读几 GB 的 pickle
            steps_before = self.train_server.total_steps_trained
            hb.write(force=True, phase='fit', iteration=iteration, round=round_index + 1,
                     game=CONFIG['game_name'], profile=get_profile(),
                     sims_per_move=ramp_simulations(iteration),
                     buffer_size=self.buffer.size, buffer_capacity=self.buffer.capacity,
                     note='fit start')
            t_fit = time.time()
            done = self.train_server.fit(n_collected_samples=n_data, iteration=iteration,
                                          reload_buffer=False, val_buffer=self.val_buffer)
            fit = {'wall': time.time() - t_fit,
                   'steps': self.train_server.total_steps_trained - steps_before,
                   'batch': self.buffer.batch_size}
            self.logger.info(f'Received message :{done}.')

            # 以防模型还没创建好（save 失败此前只记日志，这里给它一个死线，
            # 否则 while 永久等待把训练卡死）
            path = get_checkpoint_path(game_name, iteration=iteration)
            deadline = time.time() + 120
            while not os.path.exists(path):
                if time.time() > deadline:
                    raise FileNotFoundError(
                        f'Checkpoint {path} not created within 120s after fit(); '
                        f'save_checkpoint likely failed - check [fit] logs.')
                self.logger.info(f'Checkpoint {path} does not exist. Retrying.')
                time.sleep(1)
            if self.stop_event.is_set():
                # 停止请求在 fit 期间到达：fit 必须跑完（半成品模型不能存），但评估没必要再烧 1 分钟
                self.logger.info('收到停止请求（发生在 fit 期间）：模型已存盘，跳过评估，退出。')
                self._record_perf(round_index, iteration, sp, fill, fit, None, 0.0)
                hb.write(force=True, phase='idle', note='stopped during fit')
                return
            hb.write(force=True, phase='eval', iteration=iteration, round=round_index + 1,
                     game=CONFIG['game_name'], profile=get_profile(),
                     sims_per_move=ramp_simulations(iteration),
                     buffer_size=self.buffer.size, buffer_capacity=self.buffer.capacity,
                     note='eval start')
            # 评估模型，按结果更新模型
            t_ev = time.time()
            ev = self.evaluation(iteration, n_games=100)
            self._record_perf(round_index, iteration, sp, fill, fit, ev, time.time() - t_ev)

            if self.stop_after_round:
                self.logger.info('收到"等本轮结束"请求：本轮（自博弈+fit+评估+perf）已全部完成，退出。')
                hb.write(force=True, phase='idle', note='stopped after round')
                return
            # 训练网络，保存网络
            iteration += 1

    def _record_perf(self, round_index: int, iteration: int, sp: dict, fill: dict,
                     fit: dict, ev: dict | None, ev_wall: float) -> None:
        """吞吐记录：三段计时 → logs/perf/perf.jsonl 留档 + TensorBoard Perf/* + 终端一行摘要。"""
        sp_w = max(sp['wall'], 1e-9)
        fit_sps = fit['steps'] * fit['batch'] / max(fit['wall'], 1e-9)
        ev = ev or {}
        ev_games = int(ev.get('games', 0))
        rec = {
            'ts': datetime.now().isoformat(timespec='seconds'),
            'profile': get_profile(), 'iteration': iteration, 'round': round_index + 1,
            'selfplay': {'games': sp['games'], 'wall_s': round(sp['wall'], 1),
                         'samples': sp['samples'], 'sims': sp['sims'],
                         'sims_per_move': sp['n_sims'],
                         'batch_leaves': sp.get('batch_leaves', 1),
                         'samples_per_s': round(sp['samples'] / sp_w, 2),
                         'sims_per_s': round(sp['sims'] / sp_w, 1),
                         'games_per_min': round(sp['games'] / sp_w * 60, 2),
                         'infer': sp.get('infer', {}),
                         'mcts': sp.get('mcts', {})},
            'fill': {'games': fill['games'], 'wall_s': round(fill['wall'], 1),
                     'samples': fill['samples']},
            'fit': {'wall_s': round(fit['wall'], 1), 'steps': fit['steps'],
                    'samples_per_s': round(fit_sps, 1),
                    'sec_per_step': round(fit['wall'] / max(fit['steps'], 1), 3)},
            'eval': {'games': ev_games, 'win': int(ev.get('win', 0)), 'lose': int(ev.get('lose', 0)),
                     'draw': int(ev.get('draw', 0)), 'win_rate': round(float(ev.get('win_rate', 0.0)), 4),
                     'promoted': bool(ev.get('promoted', False)), 'wall_s': round(ev_wall, 1)},
            'round_wall_s': round(sp['wall'] + fill['wall'] + fit['wall'] + ev_wall, 1),
            'buffer_size': self.buffer.size,
            'buffer_capacity': self.buffer.capacity,
            'val_buffer_size': self.val_buffer.size,
            'val_buffer_capacity': self.val_buffer.capacity,
        }
        perf_dir = os.path.join(CONFIG['log_dir'], 'perf')
        os.makedirs(perf_dir, exist_ok=True)
        with open(os.path.join(perf_dir, 'perf.jsonl'), 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
        w = getattr(self.train_server, 'writer', None)
        if w is not None:
            w.add_scalar('Perf/selfplay_sims_per_s', rec['selfplay']['sims_per_s'], iteration)
            w.add_scalar('Perf/selfplay_samples_per_s', rec['selfplay']['samples_per_s'], iteration)
            w.add_scalar('Perf/selfplay_games_per_min', rec['selfplay']['games_per_min'], iteration)
            infer = rec['selfplay'].get('infer') or {}
            if infer:
                w.add_scalar('Perf/infer_batch_mean', infer.get('batch_mean', 0), iteration)
                w.add_scalar('Perf/infer_fwd_ms', infer.get('fwd_ms_mean', 0), iteration)
                w.add_scalar('Perf/infer_req_per_s', infer.get('req_per_s', 0), iteration)
            mcts = rec['selfplay'].get('mcts') or {}
            if mcts:
                w.add_scalar('Perf/mcts_select_ms_per_sim', mcts.get('select_ms_per_sim', 0), iteration)
                w.add_scalar('Perf/mcts_wait_ms_per_sim', mcts.get('wait_ms_per_sim', 0), iteration)
            w.add_scalar('Perf/fit_samples_per_s', rec['fit']['samples_per_s'], iteration)
            w.add_scalar('Perf/eval_games_per_min',
                         round(ev_games / max(ev_wall, 1e-9) * 60, 2), iteration)
            w.add_scalar('Perf/round_wall_s', rec['round_wall_s'], iteration)
        infer = rec['selfplay'].get('infer') or {}
        infer_txt = (f" | engine 均批{infer.get('batch_mean', 0):.0f}"
                     f"(p90 {infer.get('batch_p90', 0)}"
                     f", 尾{infer.get('batch_recent_mean', 0):.0f}) {infer.get('fwd_ms_mean', 0):.2f}ms/批"
                     f" {infer.get('req_per_s', 0):.0f}req/s") if infer else ''
        mcts = rec['selfplay'].get('mcts') or {}
        mcts_txt = (f" | 每模拟 选叶子{mcts.get('select_ms_per_sim', 0):.2f}ms"
                    f" 等引擎{mcts.get('wait_ms_per_sim', 0):.2f}ms"
                    f"(CPU占{mcts.get('select_ratio', 0):.0%})") if mcts else ''
        line = (f"[perf] round {round_index + 1} it{iteration} [{get_profile()}]: "
                f"selfplay {sp['games']}局/{sp['wall']:.0f}s = {rec['selfplay']['games_per_min']:.1f}局/min "
                f"({sp['samples']}样本 {rec['selfplay']['samples_per_s']:.1f}样本/s, "
                f"{rec['selfplay']['sims_per_s']:.0f}sims/s @{sp['n_sims']}sims/手 x{sp.get('batch_leaves', 1)}叶){infer_txt}{mcts_txt} | "
                f"fit {fit['steps']}步/{fit['wall']:.0f}s = {rec['fit']['samples_per_s']:.0f}样本/s | "
                f"eval {ev_games}局/{ev_wall:.0f}s 胜率{float(ev.get('win_rate', 0.0)):.1%} | "
                f"轮总 {rec['round_wall_s']:.0f}s")
        self.logger.info(line)
        print(line, flush=True)

    def self_play(self, iteration: int, n_games=100, round_index: int | None = None,
                  phase: str = 'selfplay') -> int:
        """自博弈收集数据
        :param iteration: 当前迭代轮次
        :param n_games: 本轮自博弈的对局数量。线程池按 n_workers 滚动补位：
               局数 > 线程数时会自动"下一局完成→立刻开新局"，不再是一波同时开跑、
               尾部只剩几局在独撑（那几局的批会被虚拟损失兜住，见 mcts_batch_leaves）。
        :param round_index: 本次启动内的进度（sim 爬坡用；缺省退回绝对 iteration）"""

        start = time.time()
        # 动态 n_simulations：按绝对代爬坡（重启续上，不退回起点）
        n_simulations = ramp_simulations(iteration)
        round_t0 = time.time()

        # 每手同时挂起的 playout 数（虚拟损失批量）。批大小 ≈ 在跑局数 × K。
        batch_leaves = int(CONFIG.get('mcts_batch_leaves', 8))
        hb.write(force=True, phase=phase, iteration=iteration, round=(round_index or 0) + 1,
                 game=CONFIG['game_name'], profile=get_profile(), round_t0=round_t0,
                 games_done=0, games_total=n_games, sims_per_move=n_simulations,
                 batch_leaves=batch_leaves, buffer_size=self.buffer.size,
                 buffer_capacity=self.buffer.capacity,
                 val_buffer_size=self.val_buffer.size, val_buffer_capacity=self.val_buffer.capacity)
        # 规则 R2 硬保证每局必然终局（时钟/长将判负/三次重复都是终结器）——旧"1.5×超供
        # + stop_signal 掐尾局"是给缺陷规则时代防下不完的棋兜底的；如今掐死的是能收真 z
        # 的完整数据、白烧的是算力 bound 机器上最贵的算力。删除截断：恰提 n_games 局局局下完
        futures = [self.pool.submit(self.self_play_worker, n_simulations, batch_leaves)
                   for _ in range(n_games)]
        data_count, win_count, lose_count, draw_count, completed, sims_count = 0, 0, 0, 0, 0, 0
        sel_s = wait_s = 0.0
        stop_announced = False
        # 用 wait(FIRST_COMPLETED, timeout=1s) 轮询而不是 as_completed：取消动作必须在
        # "拿到下一个结果之前"发生，否则 pending 的局没人 cancel，as_completed 会一直干等
        # （第一版就这么自锁的）；轮询还能让停止响应 ≤1s。
        pending = set(futures)
        game_errors = 0
        with tqdm(total=n_games, desc='Self-play') as pbar:
            while pending:
                if self.stop_event.is_set() and not stop_announced:
                    # 已在跑的局**不打断也不丢**（它们手里的样本是有效数据），只取消还没开跑的，
                    # 然后继续收，直到手上这波全部落袋再存盘退出。
                    # 注意：concurrent.futures.wait() **不把 CANCELLED 当 done**（未通知的取消
                    # 会永远留在 pending 里，实测自锁），所以取消成功的要自己摘出集合。
                    stop_announced = True
                    n_cancel = 0
                    for f in list(pending):
                        if f.cancel():
                            pending.discard(f)
                            n_cancel += 1
                    self.logger.info(f'收到停止请求：已完成 {completed}/{n_games} 局，'
                                     f'取消未开始的 {n_cancel} 局，等手上这波收尾。')
                    # 强写心跳告知"正在收尾"：runner 看到新鲜的 stopping 标记就不会在
                    # 收局期间把 SIGINT 升级成 SIGKILL（10/28 演练：4 局在飞 150s 没收完，
                    # 被升级杀掉，白丢一轮）
                    hb.write(force=True, phase=phase, iteration=iteration,
                             game=CONFIG['game_name'], profile=get_profile(),
                             round=(round_index or 0) + 1, round_t0=round_t0,
                             games_done=completed, games_total=n_games, stopping=True,
                             buffer_size=self.buffer.size, buffer_capacity=self.buffer.capacity)
                if stop_announced:
                    # 收尾期每个轮询周期都跳一下心跳（内部 2s 节流）：在跑的局可能
                    # 两三分钟才完一局，心跳断了 runner 会误判"卡死"而升级信号（演练 C 实锤）。
                    hb.write(phase=phase, iteration=iteration, game=CONFIG['game_name'],
                             profile=get_profile(), round=(round_index or 0) + 1,
                             round_t0=round_t0, games_done=completed, games_total=n_games,
                             stopping=True, buffer_size=self.buffer.size,
                             buffer_capacity=self.buffer.capacity)
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    if future.cancelled():
                        continue
                    try:
                        samples, winner, final_state, mstat = future.result()
                    except (ConnectionError, FileNotFoundError):
                        raise
                    except Exception as e:                   # noqa: BLE001 单局炸掉不该拖垮整轮
                        game_errors += 1
                        self.logger.exception(f'第 {completed + 1} 局异常（已记 {game_errors} 次）：{e}')
                        pbar.update(1)
                        completed += 1
                        if game_errors > max(3, n_games // 100):   # 系统性故障才停
                            raise
                        continue
                    sel_s += mstat['select_s']
                    wait_s += mstat['wait_s']
                    pbar.update(1)
                    completed += 1
                    pbar.set_postfix(rate=f'{completed / max(time.time() - start, 1e-9) * 60:.1f}g/min')
                    win_count += winner == 0
                    lose_count += winner == 1
                    draw_count += winner == -1
                    data_count += len(samples)
                    sims_count += len(samples) * n_simulations   # 吞吐分母：本局消耗的 MCTS 模拟数

                    # 观战播报：训练期间终端就能看出对局形态（先手/后手胜率、局长、样本密度）
                    every = CONFIG.get('watch_every_games', 0)
                    if every and completed % every == 0:
                        label = '先手胜' if winner == 0 else '后手胜' if winner == 1 else '和棋'
                        pbar.write(f'[watch] {completed}/{n_games} 局: {label}, 本局{len(samples)}手 | '
                                   f'累计 先手{win_count} 后手{lose_count} 和{draw_count}, '
                                   f'均{data_count / completed:.1f}样本/局')
                    bevery = CONFIG.get('watch_board_every_games', 0)
                    if bevery and completed % bevery == 0:
                        try:
                            side = 0 if winner == -1 else 1 - winner
                            pbar.write(self.env_class.get_board_str(final_state, side))
                        except Exception as e:   # 观战失败绝不影响训练
                            pbar.write(f'[watch] 棋盘打印失败: {e}')

                    # 心跳：面板的"实时 buffer / 进度"就靠它（节流 2s，写失败静默）
                    hb.write(phase=phase, iteration=iteration, round=(round_index or 0) + 1,
                             game=CONFIG['game_name'], profile=get_profile(), round_t0=round_t0,
                             games_done=completed, games_total=n_games,
                             stopping=self.stop_event.is_set(),
                             sims_per_move=n_simulations, batch_leaves=batch_leaves,
                             samples=data_count, sims=sims_count,
                             buffer_size=self.buffer.size, buffer_capacity=self.buffer.capacity,
                             val_buffer_size=self.val_buffer.size,
                             val_buffer_capacity=self.val_buffer.capacity)
                    holdout = CONFIG['val_holdout_ratio']
                    for sample in samples:
                        # 验证样本从训练池互斥划出，loss 曲线才有"没见过的数据"含义
                        if holdout > 0 and random.random() < holdout:
                            self.val_buffer.add(sample)
                        else:
                            self.buffer.add(sample)

        # 本段自博弈吞吐统计（供 run() 汇总进 [perf] 记录）
        mcts_txt = {}
        if sims_count:
            mcts_txt = {'select_ms_per_sim': round(sel_s / sims_count * 1e3, 3),
                        'wait_ms_per_sim': round(wait_s / sims_count * 1e3, 3),
                        'select_ratio': round(sel_s / max(sel_s + wait_s, 1e-9), 3)}
        self._last_sp = {'games': completed, 'samples': data_count, 'sims': sims_count,
                         'wall': time.time() - start, 'n_sims': n_simulations,
                         'mcts': mcts_txt,
                         'batch_leaves': batch_leaves,
                         'infer': self.train_server.infer_stats()}
        self.buffer.save()
        self.val_buffer.save()
        hb.write(force=True, phase=phase, iteration=iteration, round=(round_index or 0) + 1,
                 game=CONFIG['game_name'], profile=get_profile(), round_t0=round_t0,
                 games_done=completed, games_total=n_games, samples=data_count, sims=sims_count,
                 stopping=True,
                 buffer_size=self.buffer.size, buffer_capacity=self.buffer.capacity,
                 val_buffer_size=self.val_buffer.size, val_buffer_capacity=self.val_buffer.capacity,
                 wall_s=round(time.time() - start, 1))
        # self.midgame_buffer.save()
        # 总结
        duration = time.time() - start
        self.logger.info(
            f'selfplay {completed}局游戏，每步模拟{n_simulations}次，收集到原始数据{data_count}条,\n'
            f'win rate:{win_count / completed:.2%},lose rate:{lose_count / completed:.2%},'
            f'draw rate:{draw_count / completed :.2%}。')
        self.logger.info(
            f'总用时{duration:.2f}秒, 平均步数{data_count / completed:.2f}, 平均每条数据用时{(duration / data_count) if data_count else float('inf'):.4f}秒。'
        )
        return data_count

    def self_play_worker(self, n_simulations: int, batch_leaves: int = 1) -> tuple[list[
        tuple[NDArray, NDArray, float]], int, NDArray, dict]:
        """进行一局游戏，收集经验
        :return ([(state,pi_move,q),...], winner, final_state)
                winner 0/1=获胜玩家，-1=平局；final_state 只供观战打印棋盘，不入池"""

        env = self.env_class()
        # 70%概率选择开局库开局，增加多样性
        p = random.random()
        used_book = False
        if p < 0.3 and self.opening_buffer is not None:
            env.state, env.last_action, env.steps, = self.opening_buffer.sample()
            env.player_to_move = env.steps % 2
            used_book = True
        elif p < 0.7 and self.endgame_buffer is not None:
            env.state, env.last_action, env.steps, = self.endgame_buffer.sample()
            env.player_to_move = env.steps % 2
            used_book = True
        if used_book and hasattr(env, 'register_current_position'):
            # 直接改写 state 后必须重登记对局历史（否则重复/长将判定的起点计数错位）
            env.register_current_position()
        if used_book:
            # 库里可能存着已经终局的局面（无合法着法/已被判终局）。直接改写 state 不会重算
            # terminated，于是下一步 MCTS 根节点 valid_actions 为空 → get_pi 里 np.max([]) 崩
            # （10/28 05:14 就是这样把跑了 4 小时的训练一击打死）。宁可退回全新开局。
            if env.terminated or not len(env.get_valid_actions(env.state, env.player_to_move)):
                env = self.env_class()
                used_book = False

        # 进程内直接通过队列请求训练服务器的推理引擎
        mcts = NeuronMCTS.make_queue_mcts(infer_q=self.train_server.request_queue,
                                          env_class=self.env_class,
                                          state=env.state,
                                          last_action=env.last_action,
                                          player_to_move=env.player_to_move,
                                          is_self_play=True,
                                          planes_version=self.train_server.feature_version)
        samples = []
        exploration_steps = settings['selfplay']['exploration_steps']
        tau_decay_rate = settings['selfplay']['tau_decay_rate']
        while not env.terminated:   # 终局由规则硬保证（时钟/长将/重复都有裁定）
            run_mcts(mcts, n_simulations, batch_leaves)  # 模拟（流水线：K 条 playout 常驻在飞）

            if env.terminated or not len(mcts.root.valid_actions):
                break                                        # 终局（保险：规则应已置 terminated）
            # 采集原始概率分布。象棋需要交换红黑双方位置对应的概率分布
            pi_target = mcts.get_pi(1.0)
            if env.player_to_move == 1 and hasattr(env, "switch_side_policy"):
                pi_target = env.switch_side_policy(pi_target)
            # 象棋表示state和神经网络state不一样，需要转换。五子棋也进行了接口匹配
            # 按训练引擎的 feature_version 生成特征面（buffer/fit 与模型同版本）
            state = env.convert_to_network(env.state, env.player_to_move,
                                           getattr(self.train_server, 'feature_version', None))
            # q代表对上个玩家的回报，-q代表当前玩家的回报
            q = mcts.root.w / mcts.root.n
            samples.append((state, pi_target, -q, env.player_to_move))

            # 前期高温，后期低温。根据mcts模拟的概率分布进行落子
            if env.steps < exploration_steps:
                tau = np.power(tau_decay_rate, env.steps)
            else:
                tau = 0.1
            pi = mcts.get_pi(tau)  # 获取mcts的概率分布pi
            action = np.random.choice(len(pi), p=pi)
            env.step(action)  # 执行落子
            # env.render()
            mcts.apply_action(action)  # mcts也要根据action进行对应裁剪

        mcts.shutdown()
        # 能退出主循环必然是完整终局：每条样本都带真 z，截断通道已随规则完善一起删除
        # env.render()
        self.logger.info(f'Game end. Winner: {env.winner}, steps: {env.steps}')

        # 短局不再丢弃：R2 规则下"12 手绝杀"是搜索找到强制胜的高质量样本，
        # 按局长概率丢弃会系统性抽稀速杀线路、破坏 on-policy 分布（旧补丁针对的
        # "随机网速串味"已由开局库注入多样性取代）

        # 以当前玩家视角获取reward，胜1负-1平0
        for i in range(len(samples)):
            state, pi, q, p = samples[i]
            z = -1.0 if env.winner == 1 - p else 1.0 if env.winner == p else 0.0
            # 奖励折扣，鼓励短赢和长输
            # gamma = 0.99
            # z = z * np.power(gamma, steps)
            # q与z加权使用
            alpha = 0.5
            v = alpha * z + (1 - alpha) * q

            samples[i] = state, pi, v

        # 驱动方式耗时拆分：选叶子(CPU) vs 等引擎(延迟)。判别"利用率不高"的根因用
        sims = env.steps * n_simulations
        mcts_stat = {'select_s': mcts.stat_select_s, 'wait_s': mcts.stat_wait_s, 'sims': sims}
        return samples, env.winner, env.state.copy(), mcts_stat

    def evaluation(self, iteration: int, n_games: int) -> None:
        """评估模型，对战守擂模型胜率超过 win_threshold 才晋升"""
        start_time = time.time()
        # 冷启动（审计修正）：没有历史 best 时"新模型 vs 随机引擎"毫无信息量，
        # 且不晋升的话主引擎权重永远停在随机初始 → selfplay 永远产随机数据，整个训练卡死在冷启动。
        # 与 AlphaZero 一致：首个模型无守擂者，直接晋升。
        if self.best_index == -1:
            save_best_index(iteration)
            self.best_index = iteration
            self.train_server.update_eval_model(iteration)
            self.logger.info(f'Cold start: no previous best, promoting {iteration} directly.')
            return {'games': 0, 'win': 0, 'lose': 0, 'draw': 0, 'win_rate': 0.0,
                    'promoted': True, 'wall': time.time() - start_time}
        # 为参赛的两个模型创建进程内推理引擎，无需hub/socket
        # 为参赛的两个模型创建进程内推理引擎，无需hub/socket
        engine_new = InferenceEngine(iteration, game_name)
        engine_new.start()
        engines = {iteration: engine_new}
        if self.best_index == iteration:
            engines[self.best_index] = engine_new
        else:
            engine_old = InferenceEngine(self.best_index, game_name)
            engine_old.start()
            engines[self.best_index] = engine_old
        try:
            # 提交 n_games 局；提前分出胜负即 break（在跑的局靠 stop_signal 截断收尾）
            stop_signal = threading.Event()
            futures = [self.pool.submit(self.evaluation_worker, iteration, stop_signal, engines) for _ in
                       range(int(n_games))]
            win_count, lose_count, draw_count, win_rate, total_steps = 0, 0, 0, 0.0, 0
            threshold = CONFIG['win_threshold']
            # 进度条
            with tqdm(total=n_games, desc=f'{iteration} VS {self.best_index}') as pbar:
                for future in as_completed(futures):
                    if future.exception() is not None:
                        self.logger.info(f'exception:{future.exception()}')
                        stop_signal.set()
                        raise future.exception()
                    else:
                        winner, steps = future.result()

                    pbar.update(1)
                    total_steps += steps
                    win_count += winner == 0
                    lose_count += winner == 1
                    draw_count += winner == -1
                    score = win_count + draw_count / 2
                    completed = win_count + lose_count + draw_count
                    win_rate = score / completed
                    pbar.set_postfix({'win_rate': f'{win_rate:.2%}'})  # 进度条后面添加当前胜率
                    hb.write(phase='eval', iteration=iteration,
                             game=CONFIG['game_name'], profile=get_profile(), games_done=completed,
                             games_total=int(n_games), eval_win_rate=round(float(win_rate), 4),
                             buffer_size=self.buffer.size, buffer_capacity=self.buffer.capacity)

                    # 胜负已定，取消后面的比赛
                    already_won = score / n_games > threshold
                    already_lost = (score + n_games - completed) / n_games <= threshold

                    if completed >= n_games or already_won or already_lost:
                        stop_signal.set()
                        break
                    if self.stop_event.is_set():
                        # 立即结束：评估局不入池，掐尾没有数据损失
                        self.logger.info('收到停止请求：截断评估（%d 局已完成），退出。' % completed)
                        stop_signal.set()
                        break
        finally:
            engine_new.shutdown()
            if self.best_index != iteration:
                engines[self.best_index].shutdown()

        self.logger.info(
            f'Model {iteration} VS {self.best_index}，胜:{win_count},负:{lose_count},平:{draw_count},总:{completed}, 胜率{win_rate:.2%}。')
        if win_rate > threshold:  # 通过测试
            self.best_index = iteration
            save_best_index(iteration)
            self.train_server.update_eval_model(iteration)
            self.logger.info(
                f'更新最佳模型为{iteration}，平均步数{total_steps // n_games}步，评估用时{time.time() - start_time:.2f}秒.')
        else:
            self.logger.info(
                f'最佳模型仍旧为{self.best_index}，平均步数{total_steps // n_games}步， 评估用时{time.time() - start_time:.2f}秒.')
            self.train_server.remove_failed_model(iteration)
        return {'games': completed, 'win': win_count, 'lose': lose_count, 'draw': draw_count,
                'win_rate': win_rate, 'promoted': bool(win_rate > threshold),
                'wall': time.time() - start_time}

    def evaluation_worker(self, iteration: int, stop_signal: threading.Event,
                          engines: dict[int, InferenceEngine]) -> tuple[int, int]:
        """iteration对战最佳模型，随机先手顺序。
        :return 0新模型胜，1老模型胜，-1平"""
        env = self.env_class()
        # 70%概率采用开局库开局
        if self.opening_buffer is not None and random.random() < 0.7:
            env.state, env.last_action, env.steps, = self.opening_buffer.sample()
            env.player_to_move = env.steps % 2
            if hasattr(env, 'register_current_position'):
                env.register_current_position()
        # 随机先后手
        model_list = [iteration, self.best_index] if random.random() < 0.5 else [self.best_index, iteration]
        competitors = [NeuronMCTS.make_queue_mcts(
            infer_q=engines[index].request_queue,
            env_class=self.env_class,
            state=env.state,
            last_action=env.last_action,
            player_to_move=env.player_to_move,
            planes_version=engines[index].feature_version  # 跨版本 对战：各引擎吃各自布局的面
        ) for index in model_list]

        # 随机模拟测试
        n_simulations = 300
        batch_leaves = int(CONFIG.get('mcts_batch_leaves', 8))
        exploration_steps = settings['evaluation']['exploration_steps']
        tau_decay_rate = settings['evaluation']['tau_decay_rate']

        while not env.terminated and not env.truncated:
            mcts = competitors[env.player_to_move]
            run_mcts(mcts, n_simulations, batch_leaves)
            # 前几步赋予随机性，避免棋局雷同
            if env.steps < exploration_steps:
                tau = np.power(tau_decay_rate, env.steps)
                pi = mcts.get_pi(tau)
                action = np.random.choice(len(pi), p=pi)
            else:
                action = int(np.argmax(mcts.root.child_n))
            env.step(action)
            # 双方都要对应剪枝
            for mcts in competitors:
                mcts.apply_action(action)
            if stop_signal.is_set():
                env.truncated = True
        for mcts in competitors:
            mcts.shutdown()
        if env.winner in (-1, 2):  # 和棋或被提前终止
            return env.winner, env.steps

        winner = model_list[env.winner]

        # env.render()
        self.logger.info(f'Evaluation result - winner: {env.winner}, tester win:{winner == iteration}, steps: {env.steps}')

        if winner == iteration:
            return 0, env.steps
        else:
            return 1, env.steps

    def shutdown(self):
        self.pool.shutdown()
