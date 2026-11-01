from typing import Self, cast

import threading

import numpy as np
from numpy.typing import NDArray
from core.mcts.deepMcts import NeuronMCTS
from core.utils.types import EnvName
from services.inference.engine import InferenceEngine
from services.inference.hub import ServerHub
from .player import Player
from core.utils.logger import get_logger

logger = get_logger('ai_server')


class AIServer(Player):
    """服务器端AI玩家。通过进程内hub获取共享的推理引擎，MCTS走队列请求推理。"""

    def __init__(self, env_name: EnvName, model_id: int, n_simulation=500, verbose=True,
                 hub: ServerHub = None, profile: str | None = None, temperature: float = 0.0) -> None:
        super().__init__(env_name)
        self.model_id = model_id
        # 档位：不传就用进程档位。对弈页要能开到 s 档（新训练）而不是只有 m 档老模型
        self.profile = profile
        self._n_simulation = n_simulation
        # 采样温度：0=纯 argmax（原行为：首步 get_pi(3) 随机、其余 argmax）；
        # >0 时每步按访问次数分布 get_pi(temp) 采样（越大越随机）
        self.temperature = max(0.0, float(temperature))
        self.mcts: NeuronMCTS | None = None
        self.verbose = verbose  # verbose=True打印日志信息
        self.win_rate = -1.0  # -1代表无数据
        self.hub = hub if hub is not None else ServerHub()
        self._own_hub = hub is None  # 自己创建的hub随玩家一起清理
        self._engine_name: str | None = None  # 已占用引擎名称，shutdown时释放
        self._dead = False  # reset() 后置位，在途 /update 据此快速失败
        self._op_lock = threading.Lock()  # update 与 reset 互斥（防在途请求撞销毁）

    @staticmethod
    def _root_win_rate(mcts) -> float:
        """根局面胜率的稳定口径：MCTS 访问均值与根的网络原始评估混合。
        浅树时 W/N 噪声大，网络先验占比重；访问数上来后统计自然接管
        （权重 n/(n+60)，上限 0.85）。返回 root 轮走方胜率 ∈ [0,1]，无数据 -1。"""
        root = mcts.root
        if root.n <= 0:
            return -1.0
        search = root.w / root.n
        v_net = getattr(root, 'v_net', None)
        if v_net is None:
            return (search + 1) / 2
        w_search = min(0.85, root.n / (root.n + 60))
        return (w_search * search + (1 - w_search) * v_net + 1) / 2

    def _print_verbose(self, msg: str) -> None:
        """只在verbose 模式下打印"""
        if self.verbose:
            logger.info(msg)

    @property
    def description(self):
        return f'Server {self.model_id}'

    def update_state(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """同步当前棋盘。与 reset 互斥：已销毁则直接返回（/update 据此回 404）"""
        with self._op_lock:
            if self._dead:
                return
            self._update_locked(state, last_action, player_to_move)

    def _update_locked(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        self._print_verbose(f'Thinking... (Model: {self.model_id})')
        if self.mcts is None:  # 新建
            # 向hub获取(或复用)推理引擎，引用计数+1
            engine, self.model_id = self.hub.get_engine(cast(EnvName, self.env_class.__name__),
                                                        self.model_id, profile=self.profile)
            self._engine_name = engine.name
            self.mcts = NeuronMCTS.make_queue_mcts(
                infer_q=engine.request_queue,
                state=state,
                env_class=self.env_class,
                last_action=last_action,
                player_to_move=player_to_move,
                planes_version=engine.feature_version  # 人机对战：按所选模型的 feature_version 布局生成特征面
            )
        else:  # 扩展并裁剪
            child_state = self.mcts.root.get_child(last_action).state
            if np.array_equal(state, child_state):  # state匹配，向下裁剪树
                self.mcts.apply_action(last_action)
                rival_win_rate = self._root_win_rate(self.mcts)  # root是对手视角
                # -1代表无数据，其他情况计算对手胜率
                self.win_rate = 1 - rival_win_rate if rival_win_rate != -1 else -1.0
            else:  # state与当前state不匹配，新建root节点
                self.mcts.set_root(state, last_action, player_to_move)

        self.mcts.run(self._n_simulation)
        if self.temperature > 0:
            pi = self.mcts.get_pi(self.temperature)
            self.pending_action = int(np.random.choice(len(pi), p=pi))
            self.mcts.apply_action(self.pending_action)
        else:
            self.pending_action = self.mcts.choose_action()
        self.win_rate = self._root_win_rate(self.mcts)
        self.is_thinking = False

    def reset(self) -> None:
        with self._op_lock:
            super().reset()
            self._dead = True   # 已销毁：后续在途 /update 返回 404，避免 500
            self.shutdown()
            self.win_rate = -1.0

    def shutdown(self) -> None:
        if self.mcts:
            self.mcts.shutdown()
            self.mcts = None
        if self._engine_name:
            # 释放引擎引用，hub在没有使用方后会回收引擎
            self.hub.release(self._engine_name)
            self._engine_name = None
        if self._own_hub:
            self.hub.shutdown()
            self._own_hub = False

    def __del__(self) -> None:
        self.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
