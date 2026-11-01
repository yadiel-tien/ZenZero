import queue

from numpy.typing import NDArray

from core.env.functions import get_class
from core.utils.types import EnvName
from core.utils.mirror import random_mirror_state
from .request import QueueRequest, RequestLane


class InferenceResult(tuple):
    """(policy, value) 二元组 + wdl 附加属性：既有解包零破坏，
    需要结局概率的调用方读 .wdl（[负,平,胜]，非 WDL 架构为 None）。"""

    def __new__(cls, policy: NDArray, value: float, wdl: NDArray | None = None):
        self = super().__new__(cls, (policy, value))
        self.wdl = wdl
        return self


def issue(state: NDArray, env_name: EnvName, q: queue.Queue, is_self_play=False,
          lane: RequestLane | None = None) -> tuple[QueueRequest, int]:
    """投一条请求进引擎并立刻返回句柄（不等结果）——流水线的"发"侧。

    流水线 MCTS 每回一条就要再发一条，所以"发"与"收"必须拆开；
    老的 send_via_queue 是 put 一个就 wait 一个，同一条线程永远只有 1 个请求在飞。
    """
    state, symmetric_idx = preprocess_state(state, env_name, is_self_play)
    request = QueueRequest(state, lane=lane)
    q.put(request)
    return request, symmetric_idx


def harvest(request: QueueRequest, symmetric_idx: int, env_name: EnvName) -> tuple[NDArray, float]:
    """把已完成的句柄取成 (policy, value)——流水线的"收"侧；.wdl 附在结果对象上。"""
    return InferenceResult(postprocess_policy(request.policy, symmetric_idx, env_name),
                           request.value, request.wdl)


def send_many(states: list[NDArray], env_name: EnvName, q: queue.Queue,
              is_self_play=False) -> list[tuple[NDArray, float]]:
    """一次提交 K 个 state 并等齐（评估/一次性批量调用用；自博弈走路流水线 issue/harvest）。

    关键点仍然是"先全部入队、再统一等待"——否则第一发就阻塞，批又只剩 1。
    """
    lane = RequestLane()
    issued = [issue(s, env_name, q, is_self_play, lane) for s in states]
    for _ in issued:
        if not lane.acquire(timeout=300):  # 防御：推理丢失/卡死时尽早报错，而非永久挂起
            raise TimeoutError('Inference request timed out after 300s')
    return [harvest(r, idx, env_name) for r, idx in issued]


def send_via_queue(state: NDArray, env_name: EnvName, q: queue.Queue, is_self_play=False) -> tuple[
    NDArray, float]:
    """传入state，输出policy和value。is_self_play=True时对数据随机进行对称变换。
    单条请求，等价于 send_many([state])——保持旧签名给评估/人机对战等一次一手的调用方。"""
    return send_many([state], env_name, q, is_self_play)[0]


def preprocess_state(state: NDArray, env_name: str, is_self_play: bool) -> tuple[NDArray, int]:
    """selfplay时随机翻转state，并返回翻转id，否则不改变state，id=0"""
    symmetric_idx = 0
    if is_self_play:
        state, symmetric_idx = random_mirror_state(state, env_name)
    return state, symmetric_idx


def postprocess_policy(policy: NDArray, symmetry_idx: int, env_name: EnvName) -> NDArray:
    """将policy翻转回去与原始的state相匹配"""
    if symmetry_idx == 0:
        return policy
    return get_class(env_name).restore_policy(policy, symmetry_idx)
