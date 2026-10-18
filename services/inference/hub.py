import threading

from services.inference.engine import InferenceEngine
from services.inference.functions import get_model_name
from core.network.functions import read_latest_index
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
        self._evict_interval = evict_interval
        # 空闲引擎回收
        threading.Thread(target=self._evict_loop, daemon=True).start()
        # 定期显示状态
        threading.Thread(target=self.show_status, daemon=True).start()
        self.logger.info('Server hub started (in-process).')

    def get_engine(self, env_name: EnvName, model_id: int) -> tuple[InferenceEngine, int]:
        """获取(必要时创建)推理引擎，引用计数+1
        :param model_id: 0代表使用最新模型
        :return: engine与其实际加载的model_id (加载失败时为-1)"""
        if model_id == 0:  # model_id=0代表使用最新模型
            model_id = read_latest_index(env_name)
        with self.lock:
            key = get_model_name(env_name, model_id)
            engine = self.engines.get(key)
            loaded_id = model_id
            if engine is None:
                # 尝试加载，按实际加载结果修正key (加载失败时index为-1)
                _, loaded_id = InferenceEngine.load_model(model_id, env_name)
                key = get_model_name(env_name, loaded_id)
                engine = self.engines.get(key)
                if engine is None:
                    # 都不存在，创建引擎
                    engine = InferenceEngine(loaded_id, env_name)
                    self.engines[key] = engine
                    engine.start()
                    self.logger.info(f'Created new engine {key}.')
            self.ref_counts[key] = self.ref_counts.get(key, 0) + 1
            return engine, loaded_id

    def release(self, name: str) -> None:
        """使用方退出时释放引用，计数归零后引擎等待回收"""
        with self.lock:
            if name in self.ref_counts:
                self.ref_counts[name] -= 1
                if self.ref_counts[name] <= 0:
                    self.ref_counts[name] = 0

    def _evict_loop(self) -> None:
        """定期回收没有使用方的引擎(释放GPU显存)"""
        while not self.stop_event.wait(self._evict_interval):
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

    def shutdown(self) -> None:
        """清理所有资源"""
        self.stop_event.set()
        with self.lock:
            for engine in self.engines.values():
                engine.shutdown()
            self.engines.clear()
            self.ref_counts.clear()
        self.logger.info('Server hub has been shut down!')
