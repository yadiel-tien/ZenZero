import collections
import queue
from typing import Self, cast

import numpy as np
from numpy.typing import NDArray

import time

from core.env.env import BaseEnv
from services.inference.client import harvest, issue, send_many, send_via_queue
from services.inference.request import RequestLane
from core.utils.config import CONFIG, settings
from core.utils.types import GameResult, EnvName

# 虚拟损失：一条 playout 正在树上"飞行"时，给它路过的每条边记一次假失败（+N / -W），
# 让后续 playout 不要挤同一条路径。取值 1.0 = AlphaZero 原始设定（KataGo 用 1）。
VIRTUAL_LOSS = 1.0


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
                 c: int = None,
                 planes_version: int = None):
        self.state = state
        self.parent = parent if parent else DummyNode()
        self.c = c if c is not None else int(settings.get('c_puct', 5))
        self.env = env_class  # env类，用于调用类通用方法
        self.n_actions = env_class.n_actions
        self.player_to_move = player_to_move
        self.last_action = action_just_executed
        # 目标推理引擎的棋盘表示版本：convert_to_network 按它选布局（None=当前 config 的 feature_version）
        self.planes_version = planes_version

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
        """节点价值=u+q（**只在合法动作上算**）。

        旧版先在 2086 个动作的全宽数组上算 q+u，再切出 41 个合法项——每次 select
        白分配 3 个 ~8KB 的临时数组，深度搜索下这是每模拟 15~30µs 的纯浪费。
        合法动作数才是真实规模（象棋 20~45），先在合法子集上取值再算。
        """
        va = self.valid_actions
        # 代数化简：q + u = (w + c·p·√N) / (1 + n)。少两级临时数组、少两次 numpy 调用
        # （41 个元素的运算已经不是带宽瓶颈，numpy 每次调用的固定开销才是）
        return (self.child_w[va] + self.c * np.sqrt(self.n) * self.child_p[va]) / (1 + self.child_n[va])

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
                    parent=self,
                    c=self.c,
                    planes_version=self.planes_version
                )
            return self.children[action]
        raise ValueError('Invalid action')

    def select(self) -> Self:
        """计算 P UCT 得分最高的孩子节点，选取该孩子返回，没有创建时先创建"""
        index = int(np.argmax(self.child_scores))     # child_scores 已是合法动作序
        action = self.valid_actions[index]
        return self.get_child(int(action))

    def check_terminal(self) -> float | None:
        """终局判定（结果缓存）。终局→返回该叶子的奖励；未终局→None（需要问网络）。
        奖励视角 = 上个玩家（走进本节点的一方），与 back_propagate 的首层符号一致。"""
        if self.leaf_reward == GameResult.ONGOING:
            # 叶子节点的合法着法在 __init__ 已生成，传给 check_winner 省一次 R2 全量生成
            self.leaf_reward = self.env.check_winner(self.state,
                                                     1 - self.player_to_move,
                                                     self.last_action,
                                                     ctx=None,
                                                     valid_actions_next=self.valid_actions)  # 1胜，0平，-1负, 2未分胜负
        if self.leaf_reward == GameResult.ONGOING:
            return None
        return float(self.leaf_reward)

    def network_input(self) -> NDArray:
        """转换为适合神经网络的表示（按目标引擎的 feature version 布局）"""
        return self.env.convert_to_network(self.state, self.player_to_move, self.planes_version)

    def apply_policy_value(self, policy: NDArray, value: float, is_self_play: bool = False) -> float:
        """把网络的 (policy, value) 落进本节点。返回该节点对上个玩家的奖励（= -value）。

        value 头约定的视角是"轮走方"，而 w 那一层存的是"走进本节点的一方"，所以要取反。
        """
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
        return -value

    def evaluate(self, infer_queue: queue.Queue, is_self_play=False) -> float:
        """评估当前state，返回其对上个玩家的平均奖励，以便上个玩家选择奖励最大的动作。
        单请求同步路径（批大小=1）：AI 对战/评估这类一次一手、要求最低延迟的调用用它；
        自博弈用 NeuronMCTS.run(batch_leaves=K) 的批量路径。"""
        reward = self.check_terminal()
        if reward is not None:
            return reward
        # 发送到推理引擎推理，获取policy和value
        policy, value = send_via_queue(self.network_input(), cast(EnvName, self.env.__name__),
                                       infer_queue, is_self_play)
        return self.apply_policy_value(policy, value, is_self_play)

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
        eps = float(CONFIG['dirichlet'])  # 根节点噪声权重，config 统一调参
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
                 player_to_move: int = 0,
                 planes_version: int = None):
        self.planes_version = planes_version
        self.c = int(settings.get('c_puct', 5))
        self.root = NeuronNode(
            state=state,
            action_just_executed=last_action,
            player_to_move=player_to_move,
            env_class=env_class,
            parent=None,
            c=int(settings.get('c_puct', 5)),
            planes_version=planes_version,
        )
        self.infer_queue: queue.Queue | None = None
        self.is_self_play = False
        # 驱动方式的耗时拆分（只给流水线填）：选叶子的 CPU 时间 vs 等引擎的时间。
        # 这两个数直接回答"利用率不高是因为算不过来还是等不到活儿"。
        self.stat_select_s = 0.0
        self.stat_wait_s = 0.0

    @classmethod
    def make_queue_mcts(cls,
                        infer_q: queue.Queue,
                        env_class: type[BaseEnv],
                        state: NDArray,
                        last_action: int,
                        player_to_move: int,
                        is_self_play: bool = False,
                        planes_version: int = None) -> Self:
        """进程内多线程queue直接传递state，简单，无需启动后台服务。
        is_self_play=True时每次请求对state做随机对称变换以增加数据多样性(仅训练用)
        planes_version：目标推理引擎的棋盘表示版本 feature_version（跨表示版本对战/评估必须传）"""
        mcts = cls(state, env_class, last_action, player_to_move, planes_version=planes_version)
        mcts.infer_queue = infer_q
        mcts.is_self_play = is_self_play
        return mcts

    def set_root(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """将root初始化为新的值"""
        env_class = self.root.env
        self.root.cleanup()
        self.root = NeuronNode(state, last_action, player_to_move, env_class,
                               c=self.c, planes_version=self.planes_version)

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

    def _select_leaf(self) -> tuple[NeuronNode, list[tuple[NeuronNode, NeuronNode]]]:
        """从根往下选到叶子，返回 (叶子, 路径)。路径元素是 (父, 子) 边。

        每条边在飞行期间扣虚拟损失：父节点的 child_n += VL、child_w -= VL。
        w 的口径是"走进子节点的一方"的累计价值，所以飞行中先当成它输了——
        后来者看到这条边 q 很低，自然会分流去探索别的分支（这正是批量的目的：
        同时展开 K 个不同的叶子，而不是 K 条 playout 挤同一条路）。
        """
        node = self.root
        path: list[tuple[NeuronNode, NeuronNode]] = []
        while node.is_expanded:
            child = node.select()
            path.append((node, child))
            node.child_n[child.last_action] += VIRTUAL_LOSS
            node.child_w[child.last_action] -= VIRTUAL_LOSS
            node = child
        return node, path

    @staticmethod
    def _undo_virtual_loss(path: list[tuple[NeuronNode, NeuronNode]]) -> None:
        for parent, child in path:
            parent.child_n[child.last_action] -= VIRTUAL_LOSS
            parent.child_w[child.last_action] += VIRTUAL_LOSS

    def _run_playouts(self, n_playouts: int) -> int:
        """一次批量推进：选 K 个叶子 → 一次性问网络 → 撤销虚拟损失 → 逐个回传。
        返回实际消耗的 playout 数（重复选中同一叶子会跳过，不算预算）。"""
        pending: list[tuple[NeuronNode, list]] = []
        seen: set[int] = set()
        done = 0
        for _ in range(n_playouts):
            leaf, path = self._select_leaf()
            if id(leaf) in seen:
                # 同一未展开叶子被多条 playout 同时选中（树还很浅时才会发生）：
                # 撤掉刚扣的 VL，别再发一次重复请求，这一条不计入预算
                self._undo_virtual_loss(path)
                continue
            seen.add(id(leaf))
            reward = leaf.check_terminal()
            if reward is None:
                pending.append((leaf, path))
            else:
                # 终局叶子不需要网络，也不会有后续 playout 再走它 → 立刻撤 VL 并回传
                self._undo_virtual_loss(path)
                leaf.back_propagate(reward)
            done += 1

        if pending:
            states = [leaf.network_input() for leaf, _ in pending]
            results = send_many(states, cast(EnvName, self.root.env.__name__),
                                self.infer_queue, self.is_self_play)
            for (leaf, path), (policy, value) in zip(pending, results):
                self._undo_virtual_loss(path)
                leaf.back_propagate(leaf.apply_policy_value(policy, value, self.is_self_play))
        # 至少有第一发（除非 K 条全撞同一叶子——不可能，seen 初始为空）
        return max(done, 1)

    def run_pipelined(self, n_simulation=1000, batch_leaves: int = 8,
                      timeout: float = 300.0) -> None:
        """流水线式虚拟损失批量 MCTS：**始终保持 K 条 playout 在飞**。

        与 run(n, K) 的区别只在一处，但决定成败：run() 是"发 K 条 → 等齐 K 条 → 再发一批"，
        等齐的这段时间线程不产出新请求，引擎把队列吃空，GPU 占空比掉到 34%（线上实测
        window 5.17ms vs fwd 2.90ms，一半以上时间在空转）。这里改成每回一条就立刻补一条
        新叶子进队，队列常驻 K×局数 条，引擎每次都能凑满批。

        语义与 run(K) 完全一致：仍然最多 K 条同时"飞行"（K 条叶子扣着虚拟损失），
        只是不再有"一批结束才开下一批"的同步屏障。
        """
        env_name = cast(EnvName, self.root.env.__name__)
        lane = RequestLane()
        inflight: list[tuple[object, int, NeuronNode, list]] = []   # (请求, 对称编号, 叶子, 路径)
        remaining = int(n_simulation)
        k = max(1, int(batch_leaves))

        while remaining > 0 or inflight:
            # —— 补齐到 K 条：每补一条就占一个模拟预算 ——
            t_sel = time.perf_counter()
            while remaining > 0 and len(inflight) < k:
                leaf, path = self._select_leaf()
                if any(id(leaf) == id(item[2]) for item in inflight):
                    # 树还浅时，未展开的叶子没有孩子可供分流，K 次选择会全落在同一条上。
                    # 撤掉刚扣的 VL，**不消耗预算**，跳出去等这批结果回来把树撑开再补。
                    # （第一版这里写了 remaining -= 1，48 个模拟被"重复跳过"一口气烧光，
                    #  根还没展开就返回，get_pi 拿到空分布直接炸——本地复现过。）
                    self._undo_virtual_loss(path)
                    break
                reward = leaf.check_terminal()
                if reward is not None:
                    self._undo_virtual_loss(path)
                    leaf.back_propagate(reward)
                    remaining -= 1
                    continue
                request, symmetric_idx = issue(leaf.network_input(), env_name,
                                               self.infer_queue, self.is_self_play, lane)
                inflight.append((request, symmetric_idx, leaf, path))
                remaining -= 1

            self.stat_select_s += time.perf_counter() - t_sel
            if not inflight:
                break
            # —— 等任意一条回来（许可数与完成数一一对应），收一条、补一条 ——
            t_wait = time.perf_counter()
            acquired = lane.acquire(timeout=timeout)
            self.stat_wait_s += time.perf_counter() - t_wait
            if not acquired:
                raise TimeoutError(f'Inference pipelined request timed out after {timeout}s')
            t_sel = time.perf_counter()
            for i, (request, symmetric_idx, leaf, path) in enumerate(inflight):
                if not getattr(request, 'done', False):
                    continue
                inflight.pop(i)
                self._undo_virtual_loss(path)
                policy, value = harvest(request, symmetric_idx, env_name)
                leaf.back_propagate(leaf.apply_policy_value(policy, value, self.is_self_play))
                break

    def run(self, n_simulation=1000, batch_leaves: int = 1) -> None:
        """进行MCTS模拟，模拟完成后可根据孩子节点状态选择优势动作。

        batch_leaves=K>1 时启用虚拟损失批量：同时在飞 K 条 playout，一次入队 K 个 state。
        批大小从此 ≈ 同时在跑的局数 × K，而不是"局数"——尾段剩 1~2 局也照样能凑满一批。
        K=1 退化为老的同步单请求路径（逐位等价）。
        """
        remaining = int(n_simulation)
        k = max(1, int(batch_leaves))
        while remaining > 0:
            remaining -= self._run_playouts(min(k, remaining))

    def get_pi(self, temperature=1.0) -> NDArray[np.float32]:
        """将不同孩子节点的访问次数转换为概率分布
        :param temperature: 温度控制概率分布的集中程度，越小越集中，当为0集中到一点：时最大访问次数孩子的概率为1，其他都为0；为1时等同访问次数分布"""
        pi_full = np.zeros_like(self.root.child_n, dtype=np.float32)
        if not len(self.root.valid_actions):
            # 空根（终局局面/开局库给了死局）时 numpy 只会报
            # "zero-size array to reduction operation maximum"，根本看不出是哪里死了
            raise ValueError('MCTS 根节点没有合法着法（局面已终局？），无法生成策略目标')
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
        """释放引用"""
        self.infer_queue = None
