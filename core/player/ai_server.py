from typing import Self, cast

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
                 hub: ServerHub = None) -> None:
        super().__init__(env_name)
        self.model_id = model_id
        self._n_simulation = n_simulation
        self.mcts: NeuronMCTS | None = None
        self.verbose = verbose  # verbose=True打印日志信息
        self.win_rate = -1.0  # -1代表无数据
        self.hub = hub if hub is not None else ServerHub()
        self._own_hub = hub is None  # 自己创建的hub随玩家一起清理
        self._engine_name: str | None = None  # 已占用引擎名称，shutdown时释放

    def _print_verbose(self, msg: str) -> None:
        """只在verbose 模式下打印"""
        if self.verbose:
            logger.info(msg)

    @property
    def description(self):
        return f'Server {self.model_id}'

    def update_state(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """同步当前棋盘，如果是root的子节点就扩展，否则新建.然后进行mcts模拟走子"""
        self._print_verbose(f'Thinking... (Model: {self.model_id})')
        if self.mcts is None:  # 新建
            # 向hub获取(或复用)推理引擎，引用计数+1
            engine, self.model_id = self.hub.get_engine(cast(EnvName, self.env_class.__name__), self.model_id)
            self._engine_name = engine.name
            self.mcts = NeuronMCTS.make_queue_mcts(
                infer_q=engine.request_queue,
                state=state,
                env_class=self.env_class,
                last_action=last_action,
                player_to_move=player_to_move
            )
        else:  # 扩展并裁剪
            child_state = self.mcts.root.get_child(last_action).state
            if np.array_equal(state, child_state):  # state匹配，向下裁剪树
                self.mcts.apply_action(last_action)
                rival_win_rate = self.mcts.root.win_rate  # root是对手视角
                # -1代表无数据，其他情况计算对手胜率
                self.win_rate = 1 - rival_win_rate if rival_win_rate != -1 else -1.0
            else:  # state与当前state不匹配，新建root节点
                self.mcts.set_root(state, last_action, player_to_move)

        self.mcts.run(self._n_simulation)
        self.pending_action = self.mcts.choose_action()
        self.win_rate = self.mcts.root.win_rate
        self.is_thinking = False

    def reset(self) -> None:
        super().reset()
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
