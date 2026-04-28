import typing
from abc import ABC, abstractmethod
import random
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from utils.types import GameResult

if typing.TYPE_CHECKING:
    from player.player import Player


class BaseEnv(gym.Env, ABC):
    n_actions: int

    def __init__(self):
        self.state: NDArray | None = None
        self.shape: tuple[int, int, int] = (0, 0, 0)
        self.last_action: int = -1
        self.player_to_move: int = 0  # 当前走棋方，五子棋0代表黑，1代表白。象棋0代表红，1代表黑
        self.winner: int = 2  # 0,1代表获胜玩家，-1代表平局，2代表未决胜负
        self.terminated: bool = False
        self.truncated: bool = False
        self.steps: int = 0

    @classmethod
    @abstractmethod
    def handle_human_input(cls, state: NDArray, last_action: int, player_to_move: int) -> int:
        """无UI的情况下，通过控制台交互，将用户输入转换为动作输出"""
        raise NotImplementedError

    @abstractmethod
    def describe_last_move(self) -> None:
        """用于无UI界面，在控制台描述刚刚所走棋步."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_to_network(cls, state: NDArray, current_player: int) -> NDArray:
        """将逻辑state转换为适合神经网络的one-hot array"""
        raise NotImplementedError

    @classmethod
    def step_fn(cls, state: NDArray, action: int, player_to_move: int) -> tuple[
        NDArray, float, bool, bool, dict]:
        """脱离环境的step方法，可用于MCTS"""
        new_state = cls.virtual_step(state, action)
        player_just_moved = player_to_move

        result = cls.check_winner(new_state, player_just_moved, action)  # 1胜，0平，-1负, 2未分胜负
        reward = float(result) if result != GameResult.ONGOING else 0.0
        terminated = result != GameResult.ONGOING

        return new_state, reward, terminated, False, {}

    def run(self, players: Sequence['Player'], silent: bool = False) -> int:
        """模拟玩家比赛，玩家1胜返回0，玩家2胜返回1，平局返回-1"""
        self.reset()
        for player in players:
            player.silent = silent
            player.reset()

        index = 0
        while True:
            player = players[index]
            if not silent:
                print(f'-----player{index + 1} {player.description}-----')
            player.update_state(self.state, self.last_action, self.player_to_move)
            action = player.pending_action
            _, reward, terminated, truncated, _ = self.step(action)
            if hasattr(player, 'win_rate') and not silent:
                print(f'win rate:{player.win_rate:.2%}')
            if not silent:
                self.describe_last_move()
                self.render()
            if terminated or truncated:
                outcome = self.winner
                break
            index = 1 - index
        self.describe_result(outcome, players, silent)
        return outcome

    def describe_result(self, outcome: int, players: Sequence["Player"], silent: bool = False) -> None:
        """用于控制台描述胜负结果outcome:0玩家1胜，1玩家2胜，-1平局"""
        if not silent:
            if outcome == -1:
                print('---------平局---------')
            else:
                player = '玩家2 ' if outcome else '玩家1 '
                print(f'-----{player}{players[outcome].description} 获胜-----')

    def random_order_play(self, players: Sequence['Player'], silent: bool = False) -> int:
        """随机一局对弈，先手顺序随机
        return:玩家编号0或1，-1代表平"""
        n = random.randint(0, 1)
        p1, p2 = players
        current_players = [p1, p2] if n == 0 else [p2, p1]
        winner = self.run(current_players, silent=silent)
        if winner == -1:
            return -1
        if n == 0:
            return winner
        else:
            return 1 - winner

    def evaluate(self, players: Sequence['Player'], n_rounds: int = 100) -> list[int]:
        """2玩家对弈，打印胜率"""
        outcomes = []
        for i in range(n_rounds):
            print(f'第{i + 1}局:', end=' ')
            # 改变先手顺序
            outcome = self.random_order_play(players)
            outcomes.append(outcome)
            print(f'比赛结果：{outcome}')

        print(f"Player 1 Win Percentage:{outcomes.count(0) / n_rounds:.2%}", )
        print(f"Player 2 Win Percentage:{outcomes.count(1) / n_rounds:.2%}")
        print(f"Draw Percentage:{outcomes.count(-1) / n_rounds:.2%}")
        return outcomes

    @property
    def valid_actions(self) -> NDArray[np.int_]:
        return self.get_valid_actions(self.state, self.player_to_move)

    @classmethod
    @abstractmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        """获取合法动作的类方法"""
        raise NotImplementedError

    def set_winner(self, winner: int) -> None:
        """设置winner，0,1代表获胜玩家，-1代表平局，2代表未决胜负。winner!=2时终止游戏terminated = True"""
        self.winner = winner
        if winner != 2:
            self.terminated = True

    @classmethod
    @abstractmethod
    def check_winner(cls, state: NDArray, player_just_moved: int, action_just_executed: int) -> GameResult:
        """检查胜负情况，相对于player_just_moved来说
               :param action_just_executed: 刚刚做过的动作
               :param state: 棋盘表示
               :param player_just_moved:相对于刚落子的玩家来说的结果， 0或1
               :return: 1胜，0平，-1负, 2未分胜负"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def virtual_step(cls, state: NDArray[np.float32], action: int) -> NDArray[np.float32]:
        """只改变state，不计算输赢和奖励"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def action2move(cls, action: int) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def move2action(cls, move: tuple[int, ...]) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def restore_policy(cls, policy: NDArray, symmetry_idx: int) -> NDArray:
        """根据symmetric_idx将神经网络产生的policy还原回去，因为喂给神经网络前进行了镜像反转"""
        raise NotImplementedError

    def reset_status(self) -> None:
        self.player_to_move = 0
        self.terminated = False
        self.truncated = False

    @classmethod
    @abstractmethod
    def augment_data(cls, data: tuple[NDArray, NDArray, float]) -> list[tuple[NDArray, NDArray, float]]:
        """通过旋转和翻转棋盘进行数据增强
            - ChineseChess 支持水平翻转
            - Gomoku 支持8种增强
        :param data: (state,pi,q)
        :return 增强后的列别[(state,pi,q)]"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_board_str(cls, state: NDArray, player_to_move: int, colorize: bool = True) -> str:
        raise NotImplementedError
