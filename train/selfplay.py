import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from env.functions import get_class
from inference.functions import get_checkpoint_path
from utils.logger import get_logger
from utils.npz_loader import NPZLoader
from utils.replay import ReplayBuffer
from utils.config import game_name, settings, CONFIG
from mcts.deepMcts import NeuronMCTS
from inference.client import require_fit, require_eval_model_update, \
    require_model_removal, require_statistic_reset
from network.functions import read_latest_index, save_best_index, read_best_index
import random


class SelfPlayManager:
    def __init__(self, n_workers: int):
        self.logger = get_logger('selfplay')
        self.n_workers = n_workers
        self.buffer = ReplayBuffer(settings['buffer_size'], 2048)
        self.buffer.load()
        self.env_class = get_class(game_name)
        self.best_index = read_best_index()
        self.pool = ThreadPoolExecutor(self.n_workers)
        self.opening_buffer = NPZLoader("env/chess_step10.npz")
        self.endgame_buffer = NPZLoader("env/chess_step50.npz")
        self.debug_logger = get_logger('debug')

    def run(self, n_games: int) -> None:
        """训练入口
        :param n_games: 每轮的游戏数量"""
        iteration = read_latest_index()
        iteration = 1 if iteration == -1 else iteration + 1

        while iteration < settings['max_iters']:
            self.logger.info(f'Starting selfplay, iteration: {iteration}, best index: {self.best_index}.')
            # 先保证buffer足够大
            while self.buffer.size < self.buffer.capacity * 0.4:
                self.self_play(iteration=iteration, n_games=50)
                self.logger.info(f'Collecting data.Current buffer size: {self.buffer.size}.')

            # 开始selfplay。新数据会进行积累，直到新模型产生。
            require_statistic_reset()  # 重置engine的推理统计数据
            n_data = self.self_play(iteration=iteration, n_games=n_games)

            # 服务端进行模型训练，并保存参数，升级infer model
            done = require_fit(iteration, n_data)
            self.logger.info(f'Received message :{done}.')

            # 以防模型还没创建好
            path = get_checkpoint_path(game_name, iteration=iteration)
            while not os.path.exists(path):
                self.logger.info(f'Checkpoint {path} does not exist. Retrying.')
                time.sleep(1)
            # 评估模型，按结果更新模型
            self.evaluation(iteration, n_games=100)

            # 训练网络，保存网络
            iteration += 1

    def self_play(self, iteration: int, n_games=100) -> int:
        """自博弈收集数据
        :param iteration: 当前迭代轮次
        :param n_games: 每次自博弈进行的对局数量"""

        start = time.time()
        # 动态n_simulations,最大到1200
        n_simulations = 200 + iteration * 600 // settings['max_iters']

        stop_signal = threading.Event()
        futures = [self.pool.submit(self.self_play_worker, n_simulations, stop_signal) for _ in
                   range(int(n_games * 1.5))]
        data_count, win_count, lose_count, draw_count, truncate_count, completed = 0, 0, 0, 0, 0, 0
        with tqdm(total=n_games, desc='Self-play') as pbar:
            for future in as_completed(futures):
                try:
                    samples, winner = future.result()
                except (ConnectionError, FileNotFoundError):
                    raise
                except Exception as e:
                    self.logger.error(f'Exception occurred: {e}')
                    stop_signal.set()
                    raise
                if winner == 2:
                    continue
                pbar.update(1)

                completed += 1
                win_count += winner == 0
                lose_count += winner == 1
                draw_count += winner == -1
                truncate_count += winner == 2
                data_count += len(samples)

                for sample in samples:
                    self.buffer.add(sample)
                if completed >= n_games:
                    stop_signal.set()
                    break

        self.buffer.save()
        # self.midgame_buffer.save()
        # 总结
        duration = time.time() - start
        self.logger.info(
            f'selfplay {completed}局游戏，每步模拟{n_simulations}次，收集到原始数据{data_count}条,\n'
            f'win rate:{win_count / completed:.2%},lose rate:{lose_count / completed:.2%},'
            f'draw rate:{draw_count / completed :.2%},truncate rate:{truncate_count / completed :.2%}。')
        self.logger.info(
            f'总用时{duration:.2f}秒, 平均步数{data_count / completed:.2f}, 平均每条数据用时{(duration / data_count) if data_count else float('inf'):.4f}秒。'
        )
        return data_count

    def self_play_worker(self, n_simulations: int, stop_signal: threading.Event) -> tuple[list[
        tuple[NDArray, NDArray, float]], int]:
        """进行一局游戏，收集经验
        :return [(state,pi_move,q),...],winner.winner0,1代表获胜玩家，-1代表平局"""

        env = self.env_class()
        # 70%概率选择开局库开局，增加多样性
        p=random.random()
        if p < 0.3:
            env.state, env.last_action, env.steps, = self.opening_buffer.sample()
            env.player_to_move = env.steps % 2
        elif p<0.7:
            env.state, env.last_action, env.steps, = self.endgame_buffer.sample()
            env.player_to_move = env.steps % 2

        mcts = NeuronMCTS.make_selfplay_mcts(state=env.state,
                                             env_class=self.env_class,
                                             last_action=env.last_action,
                                             player_to_move=env.player_to_move)
        samples = []
        exploration_steps = settings['selfplay']['exploration_steps']
        tau_decay_rate = settings['selfplay']['tau_decay_rate']
        while not env.terminated and not env.truncated:
            mcts.run(n_simulations)  # 模拟

            # 采集原始概率分布。象棋需要交换红黑双方位置对应的概率分布
            pi_target = mcts.get_pi(1.0)
            if env.player_to_move == 1 and hasattr(env, "switch_side_policy"):
                pi_target = env.switch_side_policy(pi_target)
            # 象棋表示state和神经网络state不一样，需要转换。五子棋也进行了接口匹配
            state = env.convert_to_network(env.state, env.player_to_move)
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

            # 收到停止信号，截断游戏
            if stop_signal.is_set():
                env.truncated = True

        mcts.shutdown()

        # 截断的不收集数据
        if env.truncated:
            return [], env.winner
        env.render()
        print(f'winner: {env.winner},steps: {env.steps}')

        # 平局或着截断数据大部分丢弃
        # if env.winner in (-1, 2) and random.random() < 0.5:
        #     return [], env.winner

        if env.steps < 15:
            keep_prob = (env.steps / 15) ** 2
            if random.random() > keep_prob:
                return [], 2

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

        return samples, env.winner

    def evaluation(self, iteration: int, n_games: int) -> None:
        """评估模型，对战就模型胜率超过55%才更新模型"""
        start_time = time.time()
        # 额外提交20%任务，总数达到后停止，避免个别长时间等待
        stop_signal = threading.Event()
        futures = [self.pool.submit(self.evaluation_worker, iteration, stop_signal) for _ in
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

                # 胜负已定，取消后面的比赛
                already_won = score / n_games > threshold
                already_lost = (score + n_games - completed) / n_games <= threshold

                if completed >= n_games or already_won or already_lost:
                    stop_signal.set()
                    break

        self.logger.info(
            f'Model {iteration} VS {self.best_index}，胜:{win_count},负:{lose_count},平:{draw_count},总:{completed}, 胜率{win_rate:.2%}。')
        if win_rate > threshold:  # 通过测试
            self.best_index = iteration
            save_best_index(iteration)
            require_eval_model_update(iteration)
            self.logger.info(
                f'更新最佳模型为{iteration}，平均步数{total_steps // n_games}步，评估用时{time.time() - start_time:.2f}秒.')
        else:
            self.logger.info(
                f'最佳模型仍旧为{self.best_index}，平均步数{total_steps // n_games}步， 评估用时{time.time() - start_time:.2f}秒.')
            require_model_removal(iteration_to_remove=iteration)

    def evaluation_worker(self, iteration: int, stop_signal: threading.Event) -> tuple[int, int]:
        """iteration对战最佳模型，随机先手顺序。
        :return 0新模型胜，1老模型胜，-1平"""
        env = self.env_class()
        # 70%概率采用开局库开局
        if random.random() < 0.7:
            env.state, env.last_action, env.steps, = self.opening_buffer.sample()
            env.player_to_move = env.steps % 2
        # 随机先后手
        model_list = [iteration, self.best_index] if random.random() < 0.5 else [self.best_index, iteration]
        competitors = [NeuronMCTS.make_socket_mcts(
            env_class=self.env_class,
            state=env.state,
            last_action=env.last_action,
            player_to_move=env.player_to_move,
            model_id=index
        ) for index in model_list]

        # 随机模拟测试
        n_simulations = 300
        exploration_steps = settings['evaluation']['exploration_steps']
        tau_decay_rate = settings['evaluation']['tau_decay_rate']

        while not env.terminated and not env.truncated:
            mcts = competitors[env.player_to_move]
            mcts.run(n_simulations)
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

        env.render()
        print(f'winner: {env.winner},tester win:{winner == iteration},steps: {env.steps}')

        if winner == iteration:
            return 0, env.steps
        else:
            return 1, env.steps

    def shutdown(self):
        self.pool.shutdown()
        # require_train_server_shutdown()
