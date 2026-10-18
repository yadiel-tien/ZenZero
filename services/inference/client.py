import queue
import threading

from numpy.typing import NDArray
import socket

from core.env.functions import get_class
from core.utils.types import EnvName
from .functions import send, recv
from core.utils.config import CONFIG
from core.utils.mirror import random_mirror_state
from .request import QueueRequest


def require_hub_shutdown() -> None:
    """通知hub关闭"""
    _require_shutdown(CONFIG['hub_socket_path'])


def _require_shutdown(sock_path: str) -> None:
    """发送shutdown命令"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        payload = {'command': 'shutdown'}
        send(s, payload)


def apply_for_socket_path(model_id: int, env_name: str) -> str:
    """在infer hub注册infer，注册后才会建立infer使用
    :return socket_path,用于建立到infer的socket连接"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(CONFIG['hub_socket_path'])
        payload = {'command': 'register', 'model_id': model_id, 'env_name': env_name}
        send(s, payload)
        socket_path = recv(s)
        return socket_path


def require_infer_removal(model_name: str) -> None:
    """通知hub移除infer"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(CONFIG['hub_socket_path'])
        payload = {'command': 'remove', 'model_name': model_name}
        send(s, payload)


def send_request(
        sock: socket.socket,
        state: NDArray,
        env_name: EnvName,
        infer_queue: queue.Queue[QueueRequest] | None,
        is_self_play=False
) -> tuple[NDArray, float]:
    """根据是否提供queue来选择request的发送方式"""
    if infer_queue is None:
        return send_via_socket(sock, state, env_name, is_self_play)
    return send_via_queue(state, env_name, infer_queue, is_self_play)


def send_via_socket(sock: socket.socket, state: NDArray, env_name: EnvName, is_self_play=False) -> tuple[
    NDArray, float]:
    """传入state，输出policy和value。is_self_play=True时对数据随机进行对称变换。
    适用于不同进程间进行请求和推理。设计用来3.14多线程selfplay发送请求到3.13推理进程"""
    state, symmetric_idx = preprocess_state(state, env_name, is_self_play)

    # 通过socket发送请求，再通过socket接收结果回传
    send(sock, state)
    policy, value = recv(sock)
    # 还原
    policy = postprocess_policy(policy, symmetric_idx, env_name)
    return policy, value


def send_via_queue(state: NDArray, env_name: EnvName, q: queue.Queue, is_self_play=False) -> tuple[
    NDArray, float]:
    """传入state，输出policy和value。is_self_play=True时对数据随机进行对称变换。
    适用于通过多线程来请求和推理，请求和推理共用queue队列"""
    # 对state进行随机变换
    state, symmetric_idx = preprocess_state(state, env_name, is_self_play)

    # 通过queue发送请求
    event = threading.Event()
    request = QueueRequest(state, event=event)
    # 发给推理进程，等待结果
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
