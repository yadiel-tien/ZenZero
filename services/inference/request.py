import threading

from numpy.typing import NDArray


class ReferenceRequest:
    def __init__(self, state: NDArray):
        self.state = state


class QueueRequest(ReferenceRequest):
    def __init__(self, state: NDArray, event: threading.Event):
        super().__init__(state)
        self.policy = None
        self.value = None
        self.event = event  # 用来告知完成推理运算
