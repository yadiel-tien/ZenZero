import queue
import threading

from numpy.typing import NDArray

from core.env.functions import get_class
from core.utils.types import EnvName
from core.utils.mirror import random_mirror_state
from .request import QueueRequest


def send_via_queue(state: NDArray, env_name: EnvName, q: queue.Queue, is_self_play=False) -> tuple[
    NDArray, float]:
    """传入state，输出policy和value。is_self_play=True时对数据随机进行对称变换。
    进程内多线程请求推理：请求方put一个QueueRequest并等待event，推理引擎批量处理后回填结果"""
    # 对state进行随机变换
    state, symmetric_idx = preprocess_state(state, env_name, is_self_play)

    # 通过queue发送请求
    event = threading.Event()
    request = QueueRequest(state, event=event)
    # 发给推理引擎，等待结果
    q.put(request)
    event.wait()  # 阻塞等待推理线程处理完并设置 event

    policy = postprocess_policy(request.policy, symmetric_idx, env_name)
    return policy, request.value


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
