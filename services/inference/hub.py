import threading
import time

from services.inference.engine import InferenceEngine
from services.inference.functions import get_model_name
from core.network.functions import read_latest_index
from core.utils.config import get_profile
from core.utils.logger import get_logger
from core.utils.types import EnvName


class ServerHub:
    """进程内推理引擎管理中心：按 (env_name, model_id) 创建/复用 InferenceEngine，
    让同一进程内的多个应用 (AIServer/MCTS) 共享引擎。

    纯进程内实现（无socket）：使用方通过 get_engine() 获取引擎并走 request_queue 请求推理，
    用完后 release()；引用计数归零的引擎由后台线程定期回收。
    """

    def __init__(self, evict_interval: float = 30.0):
        # name:engine形式存储，可以让不同应用共享engine
        self.engines: dict[str, InferenceEngine] = {}
        # 引用计数：每个使用方(AIServer)计数1，归零后可被回收
        self.ref_counts: dict[str, int] = {}
        self.logger = get_logger('hub')
        # 避免多线程同时操作engines造成数据错误
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.evict_interval = evict_interval
        # 空闲引擎回收
        threading.Thread(target=self._evict_loop, daemon=True).start()
        # 定期显示状态
        threading.Thread(target=self.show_status, daemon=True).start()
        self.logger.info('Server hub started (in-process).')

    def get_engine(self, env_name: EnvName, model_id: int,
                   profile: str | None = None) -> tuple[InferenceEngine, int]:
        """获取(必要时创建)推理引擎，引用计数+1
        :param model_id: 0代表使用最新模型
        :param profile: 模型档位（m/s/l）；缺省=本进程档位。对弈面板要能开到 s 档的新模型
        :return: engine与其实际加载的model_id (加载失败时为-1)"""
        prof = (profile or get_profile()).strip().lower()
        if model_id == 0:  # model_id=0代表使用最新模型
            model_id = read_latest_index(env_name, prof)
        with self.lock:
            key = get_model_name(env_name, model_id, prof)
            engine = self.engines.get(key)
            loaded_id = model_id
            if engine is None:
                # 尝试加载，按实际加载结果修正key (加载失败时index为-1)
                _, loaded_id = InferenceEngine.load_model(model_id, env_name, profile=prof)
                key = get_model_name(env_name, loaded_id, prof)
                engine = self.engines.get(key)
                if engine is None:
                    # 都不存在，创建引擎
                    engine = InferenceEngine(loaded_id, env_name, profile=prof)
                    self.engines[key] = engine
                    engine.start()
                    self.logger.info(f'Created new engine {key}.')
            self.ref_counts[key] = self.ref_counts.get(key, 0) + 1
            return engine, loaded_id

    def relocate_all(self, device: str, rollback_to: str) -> str | None:
        """把所有已加载引擎迁移到 device；任一失败则已迁移的回滚，返回错误原因"""
        with self.lock:
            engines = list(self.engines.values())
        moved = []
        for eng in engines:
            try:
                eng.relocate(device)
                moved.append(eng)
            except Exception as e:
                from core.network.network import _brief_cuda_error
                for m in moved:
                    try:
                        m.relocate(rollback_to)
                    except Exception:
                        self.logger.error('relocate rollback failed', exc_info=True)
                return _brief_cuda_error(e) if 'CUDA' in str(e) or 'cuda' in str(e) else str(e)[:150]
        return None

    def release(self, name: str) -> None:
        """使用方退出时释放引用，计数归零后引擎等待回收"""
        with self.lock:
            if name in self.ref_counts:
                self.ref_counts[name] -= 1
                if self.ref_counts[name] <= 0:
                    self.ref_counts[name] = 0

    def _evict_loop(self) -> None:
        """定期回收没有使用方的引擎(释放GPU显存)"""
        while not self.stop_event.wait(self.evict_interval):
            with self.lock:
                idle = [name for name, cnt in self.ref_counts.items() if cnt <= 0]
                for name in idle:
                    engine = self.engines.pop(name, None)
                    self.ref_counts.pop(name, None)
                    if engine:
                        engine.shutdown()
                        self.logger.info(f'{name} has been removed (no users).')

    def show_status(self) -> None:
        """定期显示引擎使用情况"""
        while not self.stop_event.wait(30):
            with self.lock:
                if not self.engines:
                    continue
                status_msg = " | ".join(
                    [f"{name}: {self.ref_counts.get(name, 0)} users" for name in self.engines])
                self.logger.info(f"Inference Engines Status: {status_msg}")

    def snapshot(self) -> list[dict]:
        """给控制台的引擎清单：谁在用、在哪个设备、吞吐多少——判断显存被谁占着。"""
        out = []
        with self.lock:
            items = list(self.engines.items())
            counts = dict(self.ref_counts)
        for name, eng in items:
            try:
                st = eng.infer_stats()
            except Exception:                            # noqa: BLE001 统计失败不影响清单
                st = {}
            try:
                dev = str(eng._model_device())           # 参数实际所在设备（降级后可能与全局不符）
            except Exception:                            # noqa: BLE001
                dev = '—'
            out.append({'name': name, 'env': eng.env_name, 'profile': eng.profile,
                        'index': eng.model_index, 'device': dev,
                        'users': int(counts.get(name, 0)),
                        'idle': counts.get(name, 0) <= 0,
                        'queued': eng.request_queue.qsize(),
                        'req_per_s': st.get('req_per_s'), 'fwd_ms_mean': st.get('fwd_ms_mean'),
                        'batch_recent_mean': st.get('batch_recent_mean'),
                        'arch': eng.arch_name, 'feature_version': eng.feature_version,
                        'since': time.time() - eng.start_time})
        return out

    def release_idle(self) -> list[str]:
        """立即回收所有没有使用方的引擎（平时靠 30s 的 evict 线程；面板上按"释放"不等它）。
        有使用方（对弈会话/评估）的引擎一律不碰——那会掐死正在下的棋。"""
        freed = []
        with self.lock:
            idle = [name for name, cnt in self.ref_counts.items() if cnt <= 0]
            for name in idle:
                engine = self.engines.pop(name, None)
                self.ref_counts.pop(name, None)
                if engine:
                    engine.shutdown()
                    freed.append(name)
        if freed:
            self.logger.info(f'主动释放空闲引擎: {", ".join(freed)}')
        return freed

    def shutdown(self) -> None:
        """清理所有资源"""
        self.stop_event.set()
        with self.lock:
            for engine in self.engines.values():
                engine.shutdown()
            self.engines.clear()
            self.ref_counts.clear()
        self.logger.info('Server hub has been shut down!')
