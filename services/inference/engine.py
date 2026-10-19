import contextlib
import os
import queue
import threading
import time
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from core.utils.logger import get_logger
from core.utils.types import EnvName
from .functions import get_checkpoint_path, get_model_name
from .request import QueueRequest
from core.network.network import Net


class _ModelAccess:
    """模型访问门：forward 读者共享（多 worker 并发推理），relocate 写者独占（搬模型）。
    写者优先：有 writer 等待时新 reader 先排队，relocate 最多等一个批次的 forward，
    避免持续流量下迁移饿死。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextlib.contextmanager
    def shared(self):
        with self._cond:
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def exclusive(self):
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
            finally:
                self._waiting_writers -= 1
                self._cond.notify_all()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()

    def active_readers(self) -> int:
        with self._cond:
            return self._readers


class InferenceEngine:
    def __init__(self, model_index: int, env_name: EnvName):
        # 推理model
        self.eval_model, self.model_index = self.load_model(model_index, env_name)
        self.env_name = env_name
        self.name = get_model_name(self.env_name, self.model_index)
        # 负责单个request的发送接收
        self.request_queue: queue.Queue[QueueRequest] = queue.Queue()
        self._collector_thread: threading.Thread | None = None
        # 负责多个request组成batch转为tensor
        self.preprocess_queue: queue.Queue[list[QueueRequest]] = queue.Queue()
        self._preprocess_thread: threading.Thread | None = None
        # 负责推理（CPU 多 worker 并发 forward，见下方并行策略注释）
        self.infer_queue: queue.Queue[tuple[list[QueueRequest], torch.Tensor]] = queue.Queue()
        self._infer_threads: list[threading.Thread] = []
        # 负责推理结果的处理
        self.result_queue: queue.Queue[tuple[list[QueueRequest], NDArray, NDArray]] = queue.Queue()
        self._result_thread: threading.Thread | None = None

        self.stop_event = threading.Event()
        self.model_lock = _ModelAccess()
        self.logger = get_logger('inference')
        # CPU 并行策略（i9-14900K/32 逻辑核 × Gomoku 256ch 网实测）：
        #   单 worker 大线程池 t8×w1 ≈ 54 req/s；线程加到 24+ 反而崩塌
        #   （b1 延迟 8T=11ms → 24T=28ms → 32T=1192ms，超 P 核数后超额订阅）
        #   多 worker 小线程池 t4×w8 ≈ 244 req/s —— 多核吞吐靠 worker 数，不是线程数
        # GPU 保持单 worker：大 batch 本身已把卡喂饱
        ncpu = os.cpu_count() or 4
        is_cpu = torch.device(self._model_device()).type == 'cpu'
        try:
            self._infer_workers = max(1, int(os.environ['ZZ_INFER_WORKERS']))
        except (KeyError, ValueError):
            self._infer_workers = max(1, min(8, ncpu // 4)) if is_cpu else 1
        # 动态 intra-op 线程：单活跃 forward 时升线程保人机对战响应
        # （b1 实测 16T=10ms vs 4T=18ms）；多并发回小池，总线程 ≈ ncpu 防超额订阅
        self._threads_single = max(2, min(16, ncpu))
        self._threads_multi = max(1, ncpu // self._infer_workers)
        self._split_batches = is_cpu and self._infer_workers > 1
        self._omp_pinned = bool(os.environ.get('OMP_NUM_THREADS'))
        # backend 按本模型参数实际所在设备配置（initialize 可能已自行降级），
        # 不读全局 CONFIG['device']：其他引擎/模型加载失败时的 _degrade_cpu
        # 会改写全局配置，污染本引擎的输入设备选择
        self._configure_backend(self._model_device())
        self.start_time = time.time()
        self.total_requests = 0
        self.finished_requests = 0
        self.max_batch_size = 50

    @classmethod
    def load_model(cls, model_index: int, env_name: EnvName, eval_mode=True) -> tuple[Net, int]:
        """尝试加载模型，加载失败时使用初始参数，index为-1"""
        model_path = get_checkpoint_path(env_name, model_index)
        model, success = Net.load_from_checkpoint(model_path, eval_model=eval_mode)
        print(f'Success: {success}')
        index = model_index if success else -1
        print(f'model index is {index}')
        return model, index

    def _configure_backend(self, device: str) -> None:
        """按设备类型配置 autocast / pin_memory 策略"""
        self._ac_dev = torch.device(device).type
        if self._ac_dev == 'cpu':
            self._tune_intraop_threads(self._threads_single)
        # CPU 上 autocast(bf16) 是重负优化（真实网络实测 b1 慢 ~90 倍，
        # 此 torch 构建无好的 oneDNN bf16 路径），只在 GPU 类后端启用
        self._ac_factory = (lambda: torch.amp.autocast(self._ac_dev)) \
            if self._ac_dev in ('cuda', 'xpu') else (lambda: contextlib.nullcontext())
        if self._ac_dev == 'cuda':
            self._pin = torch.cuda.is_available()
        elif self._ac_dev == 'xpu':
            self._pin = hasattr(torch, 'xpu') and torch.xpu.is_available()
        else:
            self._pin = False

    def _tune_intraop_threads(self, want: int) -> None:
        """调整 torch intra-op 线程池。set_num_threads 只是原子写一个全局值，
        多个 worker 并发调用安全；谁最后写谁生效，误差只影响一两拍性能。"""
        if self._omp_pinned:
            return  # 用户显式设了 OMP_NUM_THREADS，不越权
        if torch.get_num_threads() != want:
            torch.set_num_threads(want)

    def _model_device(self) -> str:
        """本引擎模型参数当前实际所在设备——输入张量唯一的可信来源"""
        return str(next(self.eval_model.parameters()).device)

    def relocate(self, device: str) -> None:
        """在线迁移引擎到另一设备（等当前批次推理完成后搬模型）；失败自动回滚"""
        old_dev = self._model_device()
        if old_dev == str(device):
            return
        with self.model_lock.exclusive():
            # backend 配置放锁内：与搬模型原子化，推理线程在共享锁内读到的
            # 参数设备 / _pin / _ac_factory 永远一致
            try:
                self.eval_model.to(device)
            except Exception:
                try:
                    self.eval_model.to(old_dev)
                    self._configure_backend(old_dev)
                except Exception:
                    pass
                raise
            self._configure_backend(device)
        self.logger.info(f'Engine {self.name} relocated: {old_dev} -> {device}')

    def start(self) -> None:
        """启动推理线程"""
        self._collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collector_thread.start()
        self._preprocess_thread = threading.Thread(target=self._pre_infer_loop, daemon=True)
        self._preprocess_thread.start()
        self._infer_threads = [
            threading.Thread(target=self._inference_loop, args=(i,),
                             name=f'infer-{i}-{self.name}', daemon=True)
            for i in range(self._infer_workers)]
        for t in self._infer_threads:
            t.start()
        self._result_thread = threading.Thread(target=self._post_infer_loop, daemon=True)
        self._result_thread.start()

    def _collect_loop(self):
        """从request queue收集数据，尽可能多的收集数据打包"""
        self.start_time = time.time()
        while not self.stop_event.is_set():
            batch_size = 1
            threshold = 32
            # max_size = 50
            n_pending = self.infer_queue.qsize()
            delay = 1e-4 + 1e-3 * n_pending  # 根据queue排队情况，动态调整
            phase = 'ramp up'
            requests = []

            while len(requests) < min(batch_size, self.max_batch_size):
                # 思路参考tcp拥塞控制
                try:
                    request = self.request_queue.get(timeout=delay)
                    requests.append(request)
                    if phase == 'ramp up':
                        if batch_size < threshold:
                            batch_size *= 2
                        else:
                            batch_size += 1
                            phase = 'steady increase'
                    elif phase == 'steady increase':
                        batch_size += 1

                except queue.Empty:
                    threshold = max(1, batch_size // 2)
                    batch_size = threshold
                    phase = 'ramp up'
                    # 注意：超时退出时必须先交付已收集的 requests，否则请求方的
                    # event.wait() 将永久挂起（曾导致 CPU 慢推理下整树卡死）
                    break
            if requests:
                self.preprocess_queue.put(requests)
            if self.stop_event.is_set():
                break

    def _pre_infer_loop(self):
        """做推理前准备工作，将batch_queue接收到的数据查缓存，打包，转tensor"""
        while not self.stop_event.is_set():
            # 获取推理列表
            try:
                requests = self.preprocess_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 处理batch，转为tensor。CPU 多 worker 时把大 batch 均分切片
            # （每片 ≈ n/workers），让各 worker 真正并行；实测小批并行
            # t4×w8=244req/s 远快于单 worker 吞大 batch。GPU 大 batch 才是
            # 吞吐之王，不切。
            n = len(requests)
            chunk = max(1, n // self._infer_workers) if self._split_batches else n
            for i in range(0, n, chunk):
                part = requests[i:i + chunk]
                batch_tensor = torch.from_numpy(np.stack([r.state for r in part]))
                if self._pin:
                    batch_tensor = batch_tensor.pin_memory()
                self.infer_queue.put((part, batch_tensor))

    def _inference_loop(self, worker_id: int = 0):
        """推理worker loop：接收切片 batch，并发 forward，回填结果"""
        while not self.stop_event.is_set():
            try:
                requests, tensor = self.infer_queue.get(timeout=0.5)
                # 交模型推理，取回结果。输入设备必须跟本引擎模型参数走，
                # 不能读全局 CONFIG['device']：别的模型加载失败时 _degrade_cpu
                # 会把全局配置改成 cpu，而本引擎权重可能仍驻留 cuda:0，
                # 曾因此产生 F.conv2d "weight is on cuda:0 / input on cpu" 崩溃。
                # shared 锁多 worker 并发；relocate 持 exclusive 时所有
                # worker 排空，锁内取参/调线程与搬模型原子一致。
                with torch.no_grad():
                    with self.model_lock.shared():
                        active = self.model_lock.active_readers()
                        if self._ac_dev == 'cpu':
                            # 单活跃 forward 升线程保延迟，多并发降线程防超额订阅
                            self._tune_intraop_threads(
                                self._threads_single if active <= 1 else self._threads_multi)
                        batch_tensor = tensor.to(self._model_device(), dtype=torch.float32,
                                                 non_blocking=self._pin)
                        with self._ac_factory():  # 混合精度(按设备类型自适应)
                            logits, values = self.eval_model(batch_tensor)
                probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
                values = values.float().cpu().numpy()
                self.result_queue.put((requests, probs, values))
            except queue.Empty:
                continue

    def _post_infer_loop(self):
        while not self.stop_event.is_set():
            try:
                requests, probs, values = self.result_queue.get(timeout=0.5)
                self.deliver_result(requests, probs, values)

                self.finished_requests += len(requests)
                # 每 1000 个请求打印一次吞吐量统计，避免频繁刷新控制台
                if self.finished_requests % 1000 == 0:
                    avg_cost = (time.time() - self.start_time) * 1000 / self.finished_requests
                    msg = f"Throughput: {len(requests):>2} batch | Pending: {self.infer_queue.qsize():2} | "
                    msg += f"Total: {self.finished_requests:>8} | Avg: {avg_cost:.4f}s/1k req"
                    self.logger.info(msg)

            except queue.Empty:
                continue

    def reset_statistic(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.finished_requests = 0

    def deliver_result(self, requests: list[QueueRequest], probs: Sequence[NDArray], values: Sequence[float]) -> None:
        # 结果交给请求方，通知request继续
        for r, p, v in zip(requests, probs, values):
            r.policy = p
            r.value = v
            r.event.set()

    def deliver_one(self, request: QueueRequest, prob: NDArray, value: float) -> None:
        request.policy = prob
        request.value = value
        request.event.set()

    def shutdown(self) -> None:
        """清理资源，关闭推理线程"""
        self.stop_event.set()
        if self.eval_model:
            del self.eval_model
            self.eval_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 清空队列
        for q in [self.request_queue, self.preprocess_queue, self.infer_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        for t in [self._collector_thread, self._preprocess_thread,
                  *self._infer_threads, self._result_thread]:
            if t:
                t.join()

        self._collector_thread = self._preprocess_thread = self._result_thread = None
        self._infer_threads = []
