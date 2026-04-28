import collections
import queue
import socket
from typing import Self, cast

import numpy as np
from numpy.typing import NDArray

from env.env import BaseEnv
from inference.client import send_request, apply_for_socket_path
from utils.config import CONFIG
from utils.types import GameResult, EnvName


class DummyNode(object):
    """作为根节点的父节点使用，简化逻辑"""

    def __init__(self):
        self.parent = None
        self.child_n = collections.defaultdict(float)
        self.child_w = collections.defaultdict(float)


class NeuronNode:
    def __init__(self, state: NDArray,
                 action_just_executed: int,
                 player_to_move: int,
                 env_class: type[BaseEnv],
                 parent: Self = None,
                 c=5):
        self.state = state
        self.parent = parent if parent else DummyNode()
        self.c = c
        self.env = env_class  # env类，用于调用类通用方法
        self.n_actions = env_class.n_actions
        self.player_to_move = player_to_move
        self.last_action = action_just_executed

        self.valid_actions = self.env.get_valid_actions(state, self.player_to_move)
        self.children = {}
        self.child_n = np.zeros(self.n_actions, dtype=np.float32)  # 访问次数
        self.child_w = np.zeros(self.n_actions, dtype=np.float32)  # 累计价值
        self.child_p = np.zeros(self.n_actions, dtype=np.float32)  # 先验概率
        self.is_expanded = False
        self.leaf_reward = GameResult.ONGOING  # 终局的奖励，1胜，0平，-1负，2未结束

    def __repr__(self) -> str:
        return f'move={self.env.action2move(self.last_action) if self.last_action != -1 else -1},N={self.n},W={self.w:.2f}'

    @property
    def win_rate(self) -> float:
        return (self.w / self.n + 1) / 2 if self.n > 0 else -1.0

    @property
    def depth(self) -> int:
        if not self.children:
            return 1
        return max(child.depth + 1 for child in self.children.values())

    @property
    def n(self) -> float:
        """节点访问次数"""
        return self.parent.child_n[self.last_action]

    @n.setter
    def n(self, value: float) -> None:
        self.parent.child_n[self.last_action] = value

    @property
    def w(self) -> float:
        """累积价值"""
        return self.parent.child_w[self.last_action]

    @w.setter
    def w(self, value: float) -> None:
        self.parent.child_w[self.last_action] = value

    @property
    def child_q(self) -> NDArray[np.float32]:
        """q=w/(1+n),代表节点价值，利用项，是mcts搜索出来的。用一维numpy列表存放所有孩子的q"""
        return self.child_w / (np.float32(1.0) + self.child_n)  # 避免除0

    @property
    def child_u(self) -> NDArray[np.float32]:
        """u=c*p*sqrt(sum(n))/(1+n),u代表探索项，优先访问访问次数少的。用一维numpy列表存放所有孩子的u"""
        return self.c * self.child_p * np.sqrt(self.n) / (1 + self.child_n)

    @property
    def child_scores(self) -> NDArray[np.float32]:
        """节点价值=u+q"""
        return self.child_q + self.child_u

    def get_child(self, action: int) -> Self:
        """在当前动作下执行action后产生一个子Node返回"""
        if action in self.valid_actions:
            if action not in self.children:  # 扩展真正节点
                new_state = self.env.virtual_step(self.state, action)
                self.children[action] = NeuronNode(
                    state=new_state,
                    action_just_executed=action,
                    player_to_move=1 - self.player_to_move,
                    env_class=self.env,
                    parent=self
                )
            return self.children[action]
        raise ValueError('Invalid action')

    def select(self) -> Self:
        """计算P UCT得分最高的孩子节点，选取该孩子返回，没有创建时先创建"""
        index = np.argmax(self.child_scores[self.valid_actions])
        action = self.valid_actions[index]
        return self.get_child(int(action))

    def evaluate(self, infer_queue: queue.Queue, sock: socket.socket, is_self_play=False) -> float:
        """评估当前state，返回其对上个玩家的平均奖励，以便上个玩家选择奖励最大的动作"""
        # 因为选择时会从子节点中选择最大的，直接使用上个玩家视角，选择其奖励最大的。
        # check_winner函数也要求使用落子玩家的视角

        # 避免重复 check winner
        if self.leaf_reward == GameResult.ONGOING:
            self.leaf_reward = self.env.check_winner(self.state,
                                                     1 - self.player_to_move,
                                                     self.last_action)  # 1胜，0平，-1负, 2未分胜负
        if self.leaf_reward != GameResult.ONGOING:
            return float(self.leaf_reward)
        # 转换为适合神经网络的表示
        state = self.env.convert_to_network(self.state, self.player_to_move)
        # 发送到推理进程推理，获取policy和value
        policy, value = send_request(sock, state, cast(EnvName, self.env.__name__), infer_queue, is_self_play)
        # 象棋采用了红黑交换，需要对应反转概率。类似象棋这样的env都要有switch_side_policy函数实现该功能
        if self.player_to_move == 1 and hasattr(self.env, "switch_side_policy"):
            policy = self.env.switch_side_policy(policy)

        # 概率归一化
        scale = policy.sum()
        if scale > 0:
            policy = policy / scale
        # 给孩子赋予先验概率
        self.child_p = policy
        # 在根节点添加噪声，增加对弈的随机性
        if is_self_play and isinstance(self.parent, DummyNode):
            self.inject_noise()
        self.is_expanded = True
        # z代表从该state出发，当前玩家最终得到的奖励。
        # q代表从最终结果倒推而来，代表上个玩家落子后到达当前state的平均奖励。
        # 若value拟合的z，代表从该state出发，当前玩家最终得到的平均奖励，结果需要取反
        # 若value拟合的q，代表对上个玩家的平均奖励，则无需取反
        return -value

    def back_propagate(self, result: float) -> None:
        """反向传播评估结果"""
        node = self
        while not isinstance(node, DummyNode):
            node.n += 1
            node.w += result
            result = -result
            node = node.parent

    def inject_noise(self) -> None:
        """根节点添加狄利克雷噪声，增加随机性"""
        legal_len = len(self.valid_actions)
        if legal_len == 0:
            return
        # # 根据概率信息熵调节weight
        # probs = self.child_p[self.valid_actions]
        # entropy = -np.sum(probs * np.log(probs + 1e-10))
        # max_entropy = np.log(legal_len)
        # ratio = entropy / max_entropy if max_entropy > 0 else 0
        #
        # max_eps = 0.5
        # min_eps = 0.1
        # eps = max_eps - (max_eps - min_eps) * ratio
        eps = 0.25
        # 按照alphazero的参数，根据合法动作个数调整
        alpha = 0.03 * 361 / legal_len
        noise = np.random.dirichlet([alpha] * legal_len)
        self.child_p[self.valid_actions] = noise * eps + (1 - eps) * self.child_p[
            self.valid_actions]
        self.child_p[self.valid_actions] /= self.child_p[self.valid_actions].sum()

    def cleanup(self) -> None:
        """手动清理，断开连接，因为node间相互引用，gc清理较慢"""
        self.parent = None
        self.child_n = None
        self.child_w = None
        self.child_p = None
        for child in self.children.values():
            child.cleanup()
        self.children.clear()


class NeuronMCTS:
    def __init__(self, state: NDArray,
                 env_class: type[BaseEnv],
                 last_action=-1,
                 player_to_move: int = 0):
        self.root = NeuronNode(
            state=state,
            action_just_executed=last_action,
            player_to_move=player_to_move,
            env_class=env_class,
            parent=None,
        )
        self.infer_queue: queue.Queue | None = None
        self.sock: socket.socket | None = None
        self.is_self_play = False

    @classmethod
    def make_queue_mcts(cls,
                        infer_q: queue.Queue,
                        env_class: type[BaseEnv],
                        state: NDArray,
                        last_action: int,
                        player_to_move: int) -> Self:
        """多线程queue直接传递state，简单，无需启动后台服务"""
        mcts = cls(state, env_class, last_action, player_to_move)
        mcts.infer_queue = infer_q
        return mcts

    @classmethod
    def make_selfplay_mcts(cls,
                           env_class: type[BaseEnv],
                           state: NDArray,
                           last_action: int,
                           player_to_move: int) -> Self:
        """训练时使用的mcts，直接将state通过socket发送至train server。可以用3.14 no gil运行mcts。需另外启动train server"""
        mcts = cls(state, env_class, last_action, player_to_move)
        mcts.is_self_play = True
        mcts.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mcts.sock.connect(CONFIG['train_socket_path'])
        return mcts

    @classmethod
    def make_socket_mcts(cls,
                         env_class: type[BaseEnv],
                         state: NDArray,
                         last_action: int,
                         player_to_move: int,
                         model_id: int
                         ) -> Self:
        """通过hub分发socket path，适合并发的推理任务，支持多模型同时运行。支持3.14 no gil运行mcts。需启动hub"""
        mcts = cls(state, env_class, last_action, player_to_move)
        sock_path = apply_for_socket_path(model_id, env_class.__name__)
        mcts.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mcts.sock.connect(sock_path)
        return mcts

    def set_root(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """将root初始化为新的值"""
        env_class = self.root.env
        self.root.cleanup()
        self.root = NeuronNode(state, last_action, player_to_move, env_class)

    def choose_action(self) -> int:
        """选择访问量最大的孩子作为根节点，并裁剪树,第一步有随机性，其他无随机性"""
        if self.root.last_action == -1:
            pi = self.get_pi(3)
            action = np.random.choice(len(pi), p=pi)
        else:
            action = int(np.argmax(self.root.child_n))
        self.apply_action(action)
        return action

    def apply_action(self, action: int) -> None:
        """根据lsat_action向下推进树，裁剪掉多余的"""
        old_root = self.root
        self.root = self.root.get_child(action)
        n, w = self.root.n, self.root.w

        # 避免内存泄漏，及时清理资源
        old_root.children.pop(action)
        old_root.cleanup()

        # n,w在父节点存储，更改父节点需要迁移
        self.root.parent = DummyNode()  # 删除多余分叉
        self.root.n, self.root.w = n, w

        # 新的根节点添加噪声
        if self.is_self_play:
            self.root.inject_noise()

    def run(self, n_simulation=1000) -> None:
        """进行MCTS模拟，模拟完成后可根据孩子节点状态选择优势动作"""
        for i in range(n_simulation):
            node = self.root

            # selection & Expansion
            while node.is_expanded:
                node = node.select()
            # Evaluation
            value = node.evaluate(self.infer_queue, self.sock, is_self_play=self.is_self_play)
            # Back Propagation
            node.back_propagate(value)

    def get_pi(self, temperature=1.0) -> NDArray[np.float32]:
        """将不同孩子节点的访问次数转换为概率分布
        :param temperature: 温度控制概率分布的集中程度，越小越集中，当为0集中到一点：时最大访问次数孩子的概率为1，其他都为0；为1时等同访问次数分布"""
        pi_full = np.zeros_like(self.root.child_n, dtype=np.float32)
        child_n = self.root.child_n[self.root.valid_actions]

        # 计算已扩展子节点的概率
        if temperature == 0:
            # 完全贪婪,访问次数最多的概率为1，其余为0
            pi = np.zeros_like(child_n)
            pi[np.argmax(child_n)] = 1
        else:
            # 直接计算可能溢出，使用对数方法变换
            log_n = np.log(child_n + 1e-8)
            scaled = log_n / temperature
            # 减最大值避免正溢出产生inf，负溢出=0不影响结果.
            scaled -= np.max(scaled)
            exp_n = np.exp(scaled)
            pi = exp_n / np.sum(exp_n)

            # 直接的计算方法，存在溢出风险，inf/inf会产生NaN
            # child_n = child_n ** (1 / temperature)
            # pi = child_n / np.sum(child_n)

        # 返回所有动作的概率，包括不合法的(概率0）

        pi_full[self.root.valid_actions] = pi
        return pi_full

    def __repr__(self):
        return f'root=({self.root.__repr__()})'

    def shutdown(self):
        """关闭时断开socket连接"""
        if self.sock:
            self.sock.close()
            self.sock = None
