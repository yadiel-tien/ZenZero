import threading

from numpy.typing import NDArray


class ReferenceRequest:
    def __init__(self, state: NDArray):
        self.state = state


class RequestLane:
    """一条线程的流水线完成信号：每回填一条请求释放一个许可。

    与"等齐 K 条再走"的组语义相反：流水线 MCTS 每回一条就要立刻补一条新 playout
    进队，队列才不会在批次之间被吃空（引擎空转、GPU 占空比只有 34% 的根因）。
    用信号量而不是 Event：K 条各自完成、各自释放，等待方按许可数逐条收割。
    """

    __slots__ = ('_sem',)

    def __init__(self) -> None:
        self._sem = threading.Semaphore(0)

    def release(self) -> None:
        self._sem.release()

    def acquire(self, timeout: float = 300.0) -> bool:
        return self._sem.acquire(timeout=timeout)


class QueueRequest(ReferenceRequest):
    def __init__(self, state: NDArray, event: threading.Event | None = None,
                 lane: RequestLane | None = None):
        super().__init__(state)
        self.policy = None
        self.value = None
        self.wdl = None       # WDL 架构的 [负,平,胜] 概率（其余架构恒 None）
        self.done = False      # 回填标记：流水线侧靠它判断"这条可以收割了"
        self.event = event     # 单条同步等待（人机对战/评估的一发一等）
        self.lane = lane       # 流水线信号（与 event 二选一）

    def fill(self, policy: NDArray, value: float,
             wdl: NDArray | None = None) -> threading.Event | None:
        """回填推理结果，返回需要 set() 的 Event（流水线请求返回 None，改由 lane 放行）。"""
        self.policy = policy
        self.value = value
        self.wdl = wdl
        self.done = True       # 必须先置位再唤醒等待方
        if self.lane is not None:
            self.lane.release()
            return None
        return self.event
