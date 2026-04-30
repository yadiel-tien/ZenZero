import datetime
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from services.inference.functions import recv, send, get_model_name
from services.inference.infer_server import InferServer
from core.network.functions import read_latest_index
from core.utils.config import CONFIG
from core.utils.logger import get_logger
from core.utils.types import EnvName


class ServerHub:
    """作为一个管理中心，负责推理infer的管理，包括新增，删除，不负责训练infer"""

    def __init__(self):
        self._socket = None
        # name:infer形式存储，可以让不同应用共享infer
        self.infers: dict[str, InferServer] = {}
        self.stop_event = threading.Event()
        self.logger = get_logger('hub')
        # 避免多线程同时操作infers造成数据错误
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=100)

    def start(self) -> None:
        """启动管理服务"""
        # 建立socket服务，地址在config中配置
        self._clean_socket()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(CONFIG['hub_socket_path'])
        self._socket.listen()
        # 设置超时，以便检查运行状态
        self._socket.settimeout(1)
        self.logger.info(f"Server hub started. Now is listening to {CONFIG['hub_socket_path']}.")
        # 启动状态显示
        self.pool.submit(self.show_status)

        while not self.stop_event.is_set():
            try:
                conn, _ = self._socket.accept()
                self.pool.submit(self.handle_connection, conn)
            except socket.timeout:
                continue

    def handle_connection(self, conn: socket.socket) -> None:
        """响应客户请求，单独一个函数以便多线程处理"""
        try:
            with conn:
                data = recv(conn)
                if isinstance(data, dict):
                    # 根据不同的命令，进行不同的响应
                    if data['command'] == 'register':
                        send(conn, self.register(data['env_name'], data['model_id']))
                    elif data['command'] == 'remove':
                        self.remove_infer(data['model_name'])
                    elif data['command'] == 'shutdown':
                        self.shutdown()

        except ConnectionError as e:  # 对方断开
            self.logger.info(e)
        except socket.timeout:
            self.logger.info('Socket timed out')

    def register(self, env_name: EnvName, model_id: int) -> str:
        """注册infer，并返回连接该infer的socket地址"""
        if model_id == 0:  # model_id=0代表使用最新模型
            model_id = read_latest_index(env_name)
        key = get_model_name(env_name, model_id)
        with self.lock:
            # 查询原始key是否存在
            if key in self.infers:
                infer = self.infers[key]
            else:
                # 尝试加载，根据加载结果查询key是否存在
                _, loaded_id = InferServer.load_model(model_id, env_name)
                key = get_model_name(env_name, loaded_id)
                if key in self.infers:
                    infer = self.infers[key]
                else:
                    # 都不存在，创建infer
                    infer = InferServer(loaded_id, env_name)
                    self.infers[key] = infer
                    infer.start()
                    self.logger.info(f'Registered new infer {infer.name}.')
        # 等待sock文件创建
        start = time.time()
        while not infer.is_ready:
            time.sleep(0.01)
            if time.time() - start > 10:
                raise RuntimeError(f"Infer server model {infer.name} socket creation timed out!")
        return infer.socket_path

    def remove_infer(self, model_name: str) -> None:
        """没有应用在使用infer，将其移除清理。检查使用情况在infer内部"""
        infer = None
        with self.lock:
            if model_name in self.infers:
                infer = self.infers.pop(model_name)

        if infer:
            infer.shutdown()
            self.logger.info(f'{model_name} has been removed!')

    def show_status(self):
        """定期显示连接数量"""
        while not self.stop_event.wait(30):
            with self.lock:
                if not self.infers:
                    continue
                
                status_msg = " | ".join([f"{name}: {infer.client_count} clients" for name, infer in self.infers.items()])
                self.logger.info(f"Inference Engines Status: {status_msg}")

    @staticmethod
    def _clean_socket() -> None:
        """使用前后都需要清理socket文件"""
        if os.path.exists(CONFIG['hub_socket_path']):
            os.remove(CONFIG['hub_socket_path'])

    def shutdown(self):
        """清理资源"""
        self.stop_event.set()
        # 关闭socket
        if self._socket:
            self._socket.close()
            self._socket = None
        self._clean_socket()
        # 清理infer
        with self.lock:
            for infer in self.infers.values():
                infer.shutdown()
            self.infers.clear()
        # 清理线程池
        self.pool.shutdown()
        self.logger.info('Server hub has been shut down!')
