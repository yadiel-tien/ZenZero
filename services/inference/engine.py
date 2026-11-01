import contextlib
import os
import queue
import threading
import time
from collections import Counter, deque
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from core.utils.config import CONFIG
from core.utils.logger import get_logger
from core.utils.types import EnvName
from core.utils.config import get_profile
from .functions import get_checkpoint_path, get_model_name
from .request import QueueRequest
from core.network.network import Net
from core.network.registry import DEFAULT_ARCH, get as get_arch


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
    def __init__(self, model_index: int, env_name: EnvName, profile: str | None = None):
        # 推理model（profile 缺省 = 进程档位；引擎是三身份载体：arch/feature_version/profile）
        self.profile = (profile or get_profile()).strip().lower()
        self.eval_model, self.model_index = self.load_model(model_index, env_name, profile=self.profile)
        self.env_name = env_name
        self.name = get_model_name(self.env_name, self.model_index, self.profile)
        self.arch_name = getattr(self.eval_model, 'arch_name', 'resnet_v1')
        # 负责单个request的发送接收
        self.request_queue: queue.Queue[QueueRequest] = queue.Queue()
        self._collector_thread: threading.Thread | None = None
        # 负责多个request组成batch转为tensor
        self.preprocess_queue: queue.Queue[list[QueueRequest]] = queue.Queue()
        self._preprocess_thread: threading.Thread | None = None
        # 负责推理（CPU 多 worker 并发 forward，见下方并行策略注释）
        # 三元组：(请求切片, 输入张量, 该批开始收集的时刻) —— 时间戳用于分段计时
        self.infer_queue: queue.Queue[tuple[list[QueueRequest], torch.Tensor, float]] = queue.Queue()
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
        # 批大小上限。GPU 上 t(b) 实测接近常数（b=8 1.24ms → b=256 2.37ms/批，
        # 边际仅 ~4.5µs/样本），所以批越大越划算，256 是吞吐/单请求延迟的折中；
        # CPU 上多 worker 小批才是正解（历史结论），保持 50。
        default_batch = 256 if not is_cpu else 50
        try:
            self.max_batch_size = max(1, int(os.environ['ZZ_MAX_BATCH']))
        except (KeyError, ValueError):
            self.max_batch_size = default_batch
        # 收批窗口：队列里没活时最多等这么久再交付。按最近的批大小自适应——
        # 批一直很小（人机对战/评估）就几乎不等，保单步延迟；批很大（自博弈）才开窗口。
        self._ema_batch = 0.0
        # 每批计时/批大小直方图（供日志与 [perf] 汇总）
        self._recent_batches: deque = deque(maxlen=2000)
        self._batch_hist: Counter = Counter()
        self._fwd_ms: deque = deque(maxlen=2000)   # (forward+后处理 毫秒, 批大小)
        self._post_ms: deque = deque(maxlen=2000)  # 结果投递毫秒
        # 统计数据的读写跨线程（收集/推理/投递各写各的，统计行与 [perf] 汇总来读）。
        # free-threading 的 3.14t 会**真的**并发：边 append 边迭代 deque 抛
        # RuntimeError('deque mutated during iteration')，边写边遍历 Counter 抛
        # dict changed size——线上已炸过一次（投递线程死掉）。所有读写都进这把锁。
        self._stat_lock = threading.Lock()
        # 棋盘表示版本（模型身份）：调用方 convert_to_network 必须按本引擎 feature_version
        # 生成对应布局的面。跨 version 面语义不同，不做切片兼容——通道数不符直接报错。
        cfg = getattr(self.eval_model, 'config', None)
        self._in_channels = int(cfg['in_channels']) if isinstance(cfg, dict) and 'in_channels' in cfg else 0
        self.feature_version = int(getattr(self.eval_model, 'feature_version', 1))
        self._is_wdl = bool(getattr(self.eval_model, 'is_wdl', False))

    @classmethod
    def load_model(cls, model_index: int, env_name: EnvName, eval_mode=True,
                   profile: str | None = None) -> tuple[Net, int]:
        """尝试加载模型（envelope 经 registry 分发架构），加载失败时用当前档位默认配置
        新建，index=-1"""
        model_path = get_checkpoint_path(env_name, model_index, profile)
        # 无档新建走当前默认架构（有档时 envelope 的 arch 优先，registry 内部分发）
        fresh_cls = get_arch(CONFIG.get(env_name, {}).get('default_net', {}).get('arch', DEFAULT_ARCH))
        model, success = fresh_cls.load_from_checkpoint(model_path, eval_model=eval_mode)
        print(f'Success: {success}')
        index = model_index if success else -1
        print(f'model index is {index}')
        return model, index

    def _configure_backend(self, device: str) -> None:
        """按设备类型配置 autocast / 输入搬运策略"""
        self._ac_dev = torch.device(device).type
        if self._ac_dev == 'cpu':
            self._tune_intraop_threads(self._threads_single)
        # CPU 上 autocast(bf16) 是重负优化（真实网络实测 b1 慢 ~90 倍，
        # 此 torch 构建无好的 oneDNN bf16 路径），只在 GPU 类后端启用
        self._ac_factory = (lambda: torch.amp.autocast(self._ac_dev)) \
            if self._ac_dev in ('cuda', 'xpu') else (lambda: contextlib.nullcontext())
        # pin_memory 关掉：这里没有任何重叠调度（单 worker 顺序跑），pin 只等于
        # "先拷进 pinned 暂存区再 cudaMemcpy"的额外一次拷贝。实测 b=256 时
        # 3.31ms → 2.15ms/批（scripts/bench_batch_sweep.py），大 batch 下纯亏。
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

    def _spawn(self, name: str, target, *args) -> threading.Thread:
        """启动一条带异常兜底的引擎线程。

        引擎线程一旦抛异常就是"静默死亡"：没人接手它那份队列，所有请求只能等到
        300s 超时才报错，而日志里一个字都没有（8/27 投递线程 deque 竞态即此形态）。
        这里把栈完整写进 logs/inference，并把 stop_event 置位让引擎尽快停而不是半死。
        """
        def _run():
            try:
                target(*args)
            except BaseException:                       # noqa: BLE001 兜底一切，含 KeyboardInterrupt
                self.logger.exception(f'引擎线程 {name} 崩溃（该线程已退出，后续请求将超时）')
                self.stop_event.set()
        thread = threading.Thread(target=_run, name=f'{name}-{self.name}', daemon=True)
        thread.start()
        return thread

    def start(self) -> None:
        """启动推理线程（每条都经 _spawn 兜底，异常必进日志）"""
        self._collector_thread = self._spawn('collect', self._collect_loop)
        self._preprocess_thread = self._spawn('preprocess', self._pre_infer_loop)
        self._infer_threads = [self._spawn(f'infer{i}', self._inference_loop, i)
                               for i in range(self._infer_workers)]
        self._result_thread = self._spawn('post', self._post_infer_loop)

    def _collect_window(self) -> float:
        """收批窗口：队列空时最多等多久再交付。批越大越舍得等——批小的时候
        （人机对战/评估，一步一手要低延迟）几乎不等。

        流水线 MCTS 之后请求是"每完成一条补一条"地流入的，不再是整批涌进来，
        所以中等批量时也要给一点窗口，否则引擎会把 8 条一局的流水拆成 1~3 条一发。
        """
        if self._ema_batch >= 32:
            return 2e-3
        if self._ema_batch >= 8:
            return 1e-3
        if self._ema_batch >= 3:
            return 5e-4
        return 1e-4

    def _collect_loop(self):
        """从 request queue 尽可能多地收批。

        旧版是"TCP 拥塞控制"式把 batch_size 从 1 翻倍爬坡，一次超时就把目标打回 1——
        在虚拟损失批量 MCTS 之后请求是成批到达的，批目标从 1 爬反而永远追不上。
        现在：先等第一条，然后一直收到 max_batch_size 或队列空为止（空时最多等一个
        自适应窗口）。批大小由在飞请求数决定，而不是由这个循环猜。
        """
        self.start_time = time.time()
        while not self.stop_event.is_set():
            n_pending = self.infer_queue.qsize()
            delay = 1e-4 + 1e-3 * n_pending  # 空闲时先等这么久（有活时 get 立即返回）
            t_start = time.perf_counter()
            requests = []
            try:
                requests.append(self.request_queue.get(timeout=delay))
            except queue.Empty:
                continue
            window = self._collect_window()
            while len(requests) < self.max_batch_size:
                try:
                    requests.append(self.request_queue.get(timeout=window))
                except queue.Empty:
                    break
            self._record_batch(len(requests), (time.perf_counter() - t_start) * 1e3)
            self._ema_batch = 0.8 * self._ema_batch + 0.2 * len(requests)
            self.preprocess_queue.put(requests)
            if self.stop_event.is_set():
                break

    def _pre_infer_loop(self):
        """做推理前准备工作：把已收集的 requests 打包成 tensor 送推理队列"""
        while not self.stop_event.is_set():
            # 获取推理列表
            try:
                requests = self.preprocess_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.perf_counter()

            # 处理batch，转为tensor。CPU 多 worker 时把大 batch 均分切片
            # （每片 ≈ n/workers），让各 worker 真正并行；实测小批并行
            # t4×w8=244req/s 远快于单 worker 吞大 batch。GPU 大 batch 才是
            # 吞吐之王，不切。
            n = len(requests)
            chunk = max(1, n // self._infer_workers) if self._split_batches else n
            for i in range(0, n, chunk):
                part = requests[i:i + chunk]
                batch = np.stack([r.state for r in part])
                # version 校验：面的通道数必须等于模型出生时的面数（语义随 version 整体变化，
                # 静默降级比崩溃毒得多）。不符说明调用方转换时传错了 planes_version。
                if self._in_channels and batch.shape[1] != self._in_channels:
                    msg = (f'feature-version mismatch: input has {batch.shape[1]} planes, '
                           f'engine {self.name} (version {self.feature_version}) expects '
                           f'{self._in_channels}; convert_to_network must use planes_version=engine.feature_version')
                    self.logger.error(msg)
                    raise ValueError(msg)
                # 不做 pin_memory（见 _configure_backend）；.to() 在推理线程里，
                # 这样 H2D 拷贝与本线程的打包不抢同一条线程的时间片。
                self.infer_queue.put((part, torch.from_numpy(batch), t0))

    def _inference_loop(self, worker_id: int = 0):
        """推理worker loop：接收切片 batch，并发 forward，回填结果"""
        while not self.stop_event.is_set():
            try:
                requests, tensor, t0 = self.infer_queue.get(timeout=0.5)
                t_fwd = time.perf_counter()
                # 交模型推理，取回结果。输入设备必须跟本引擎模型参数走，
                # 不能读全局 CONFIG['device']：别的模型加载失败时 _degrade_cpu
                # 会把全局配置改成 cpu，而本引擎权重可能仍驻留 cuda:0，
                # 曾因此产生 F.conv2d "weight is on cuda:0 / input on cpu" 崩溃。
                # shared 锁多 worker 并发；relocate 持 exclusive 时所有
                # worker 排空，锁内取参/调线程与搬模型原子一致。
                with torch.inference_mode():
                    with self.model_lock.shared():
                        active = self.model_lock.active_readers()
                        if self._ac_dev == 'cpu':
                            # 单活跃 forward 升线程保延迟，多并发降线程防超额订阅
                            self._tune_intraop_threads(
                                self._threads_single if active <= 1 else self._threads_multi)
                        batch_tensor = tensor.to(self._model_device(), dtype=torch.float32,
                                                 non_blocking=False)
                        with self._ac_factory():  # 混合精度(按设备类型自适应)
                            if self._is_wdl:
                                logits, wdl_logits = self.eval_model.forward_wdl(batch_tensor)
                                wdl_p = torch.softmax(wdl_logits.float(), dim=-1)
                                values = wdl_p[:, 2] - wdl_p[:, 0]   # 期望分与概率同源
                            else:
                                logits, values = self.eval_model(batch_tensor)
                probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
                values = values.float().cpu().numpy()
                wdls = wdl_p.cpu().numpy() if self._is_wdl else None
                self._record_fwd((time.perf_counter() - t_fwd) * 1e3, len(requests))
                self.result_queue.put((requests, probs, values, wdls))
            except queue.Empty:
                continue

    def _post_infer_loop(self):
        while not self.stop_event.is_set():
            try:
                requests, probs, values, wdls = self.result_queue.get(timeout=0.5)
                t0 = time.perf_counter()
                self.deliver_result(requests, probs, values, wdls)
                self._record_post((time.perf_counter() - t0) * 1e3)

                self.finished_requests += len(requests)
                # 每 1000 个请求打印一次统计，避免频繁刷新控制台
                if self.finished_requests % 1000 == 0:
                    self.logger.info(self._stat_line())

            except queue.Empty:
                continue

    def _record_batch(self, n: int, window_ms: float) -> None:
        with self._stat_lock:
            self._recent_batches.append((n, window_ms))
            self._batch_hist[n] += 1

    def _record_fwd(self, ms: float, n: int) -> None:
        with self._stat_lock:
            self._fwd_ms.append((ms, n))

    def _record_post(self, ms: float) -> None:
        with self._stat_lock:
            self._post_ms.append(ms)

    def _stats_snapshot(self) -> dict:
        """锁内一次性拷成普通 list——free-threading 下绝不能边迭代边让别人 append"""
        with self._stat_lock:
            return {'batch': [b for b, _ in self._recent_batches],
                    'window': [w for _, w in self._recent_batches],
                    'fwd': [x for x, _ in self._fwd_ms],
                    'post': list(self._post_ms),
                    'hist': dict(self._batch_hist)}

    def _stat_line(self) -> str:
        """一行吞吐/批大小/分段耗时，够判断瓶颈在收批、前向还是投递"""
        elapsed = max(time.time() - self.start_time, 1e-9)
        snap = self._stats_snapshot()
        bs = snap['batch'] or [0]
        top = ', '.join(f'{k}:{v}' for k, v in
                        sorted(snap['hist'].items(), key=lambda kv: -kv[1])[:5])
        fwd = snap['fwd'] or [0.0]
        post = snap['post'] or [0.0]
        win = snap['window'] or [0.0]
        hist = snap['hist']
        total_b = sum(hist.values())
        mean_all = (sum(k * v for k, v in hist.items()) / total_b) if total_b else 0.0
        return (f"[infer] {self.finished_requests / elapsed:.0f} req/s | batch mean {mean_all:.0f} "
                f"(recent {np.mean(bs):.0f}) "
                f"(max {max(bs)}, top {top}) | window {np.mean(win):.2f}ms | "
                f"fwd {np.mean(fwd):.2f}ms | post {np.mean(post):.3f}ms | "
                f"pending {self.infer_queue.qsize()} | total {self.finished_requests}")

    def infer_stats(self) -> dict:
        """给 [perf] 汇总用的推理侧指标快照。

        **均值按整轮累计直方图算，不是滑动窗口**：一轮末尾几十局的批会缩到 8~16
        （正是"剩几局"的尾部），滑动窗口（最近 2000 批）会被尾巴完全占满，
        于是整轮头条变成 均批16——看着像批处理坏了，其实前面 90% 的批都是 256。
        这里同时给出 `batch_recent_mean`（最近窗口）当尾部指示。
        """
        elapsed = max(time.time() - self.start_time, 1e-9)
        snap = self._stats_snapshot()
        hist = snap['hist']
        total = sum(hist.values())
        reqs = sum(k * v for k, v in hist.items())
        bs = snap['batch'] or [0]
        fwd = snap['fwd'] or [0.0]

        def pct(p: float) -> int:
            if not total:
                return 0
            need, acc = total * p, 0
            for size in sorted(hist):
                acc += hist[size]
                if acc >= need:
                    return int(size)
            return int(max(hist))

        return {'batches': int(total),
                'batch_mean': round(reqs / total, 2) if total else 0.0,   # 恒 float：统计口径要稳定
                'batch_recent_mean': round(float(np.mean(bs)), 2),
                'batch_p50': pct(0.5),
                'batch_p90': pct(0.9),
                'batch_max': int(max(hist)) if hist else 0,
                'fwd_ms_mean': round(float(np.mean(fwd)), 3),
                'req_per_s': round(self.finished_requests / elapsed, 1)}

    def reset_statistic(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.finished_requests = 0
        with self._stat_lock:
            self._recent_batches.clear()
            self._batch_hist.clear()
            self._fwd_ms.clear()
            self._post_ms.clear()

    def deliver_result(self, requests: list[QueueRequest], probs: Sequence[NDArray], values: Sequence[float],
                       wdls: Sequence[NDArray] | None = None) -> None:
        # 结果交给请求方，通知 request 继续。组请求（批量 MCTS 一局 K 发）由最后一条
        # 回填者唤醒，省掉 K-1 次线程唤醒（一批 256 时这是半个批次的量级）。
        for i, (r, p, v) in enumerate(zip(requests, probs, values)):
            event = r.fill(p, v, None if wdls is None else wdls[i])
            if event is not None:
                event.set()

    def deliver_one(self, request: QueueRequest, prob: NDArray, value: float) -> None:
        event = request.fill(prob, value)
        if event is not None:
            event.set()

    def shutdown(self) -> None:
        """清理资源，关闭推理线程。

        顺序很重要：**先停线程、再丢模型**。旧版先 `eval_model = None` 再 join，
        于是正在前向的线程会在 `_model_device()` 里撞上 None：
        `AttributeError: 'NoneType' object has no attribute 'parameters'`
        （10/28 04:46 就是这样把评估引擎的 infer0 线程搞崩的）。"""
        self.stop_event.set()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 清空队列
        for q in [self.request_queue, self.preprocess_queue, self.infer_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # join 必须带超时：线程若是卡在 CUDA/驱动里，无超时 join 会让"优雅停止"永远吊在退役阶段，
        # 外部只能升级成 SIGKILL（10/27 23:29 就是这样：数据都存好了，进程却退不掉）。
        for t in [self._collector_thread, self._preprocess_thread,
                  *self._infer_threads, self._result_thread]:
            if t:
                t.join(timeout=5.0)
                if t.is_alive():
                    self.logger.warning(f'引擎线程 {t.name} 5s 内没退出（卡在驱动/回收），放弃等待')
        # 线程都停了，这时才释放模型（顺序见 docstring）
        if getattr(self, 'eval_model', None) is not None:
            del self.eval_model
            self.eval_model = None

        self._collector_thread = self._preprocess_thread = self._result_thread = None
        self._infer_threads = []
