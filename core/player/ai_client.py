import json
import threading
import time
from enum import Enum, auto
from typing import Any

import numpy as np
import requests
from numpy.typing import NDArray

from core.utils.types import EnvName
from .player import Player
from core.utils.config import CONFIG
from core.utils.logger import get_logger

logger = get_logger('ai_client')


class ClientStatus(Enum):
    UNINITIATED = auto()
    DEACTIVATED = auto()
    IDLE = auto()
    UPDATED = auto()


class AIClient(Player):
    def __init__(self, model_idx: int, env_name: EnvName) -> None:
        super().__init__(env_name)
        self.model_id = model_idx
        self.pid = ''
        self.time_stamp = 0
        self.session = requests.Session()
        self.session.trust_env = False  # 禁用代理
        self.alive = True
        self.win_rate = -1.0
        self.status = ClientStatus.UNINITIATED

    @property
    def description(self) -> str:
        return f'AI({self.model_id})'

    def update(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        """不能阻塞，实时更新"""
        if not self.is_thinking:
            self.is_thinking = True
            self.update_state(state, last_action, player_to_move)

    def update_state(self, state: NDArray, last_action: int, player_to_move: int) -> None:
        if self.status == ClientStatus.UNINITIATED:
            threading.Thread(target=self.request_setup, args=(state, last_action, player_to_move),
                             daemon=True).start()
        elif self.status == ClientStatus.DEACTIVATED:
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        elif self.status == ClientStatus.IDLE:
            threading.Thread(target=self.request_update, args=(state, last_action, player_to_move),
                             daemon=True).start()
        else:
            threading.Thread(target=self.request_action, daemon=True).start()

    def request_update(self, state: np.ndarray, last_action: int, player_to_move: int) -> None:
        """给server发请求，更新盘面"""
        url = CONFIG['base_url'] + 'update'
        payload = {
            'array': state.tolist(),
            'action': last_action,
            'pid': self.pid,
            'player_to_move': player_to_move
        }
        response = self.post_request(url, payload)
        self.win_rate = response.get('win_rate')
        self.status = ClientStatus.UPDATED
        self.is_thinking = False

    def request_action(self) -> None:
        """给server发请求，获取action"""
        url = CONFIG['base_url'] + 'get_action'
        payload = {
            'pid': self.pid,
        }
        response = self.post_request(url, payload)

        self.pending_action = response.get('action')
        self.win_rate = response.get('win_rate')
        self.model_id = response.get('model_id')
        self.status = ClientStatus.IDLE
        self.is_thinking = False

    def request_reset(self) -> None:
        """告知server重置"""
        url = CONFIG['base_url'] + 'reset'
        payload = {'pid': self.pid}
        response = self.post_request(url, payload)
        self.win_rate = response.get('win_rate')

    def _heartbeat_loop(self):
        self.status = ClientStatus.IDLE
        self.is_thinking = False
        while self.alive:
            self.send_heartbeat()
            time.sleep(5)

    def send_heartbeat(self) -> None:
        """通过定期发送心跳告知服务端存活，以再服务端保留资源"""
        self.time_stamp = time.time()
        url = CONFIG['base_url'] + 'heartbeat'
        payload = {'pid': self.pid}
        response = self.post_request(url, payload)
        self.win_rate = response.get('win_rate')

    def request_setup(self, state: NDArray, last_action: int, play_to_move: int) -> None:
        """告知server创建推理引擎"""
        url = CONFIG['base_url'] + 'setup'
        payload = {'model_id': self.model_id,
                   'env_class': self.env_class.__name__,
                   'state': state.tolist(),
                   'last_action': last_action,
                   'play_to_move': play_to_move}
        response = self.post_request(url, payload)
        self.pid = response.get('pid')
        self.model_id = response.get('model_id')
        self.status = ClientStatus.DEACTIVATED
        self.is_thinking = False

    def post_request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求的基础方法，获取反馈，处理错误"""
        response = None
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=60  # 添加超时设置
            )

            response.raise_for_status()  # 检查HTTP错误

            return response.json()

        except requests.exceptions.HTTPError as http_err:
            error_msg = f'HTTP Error ({response.status_code if response else "Unknown"}): '
            try:
                error_data = response.json()
                error_msg += str(error_data.get('error', error_data))
            except ValueError:
                error_msg += response.text or str(http_err)
            logger.error(error_msg)
            raise  # 重新抛出异常

        except json.JSONDecodeError as json_err:
            error_msg = f'Invalid JSON response: {str(json_err)}'
            logger.error(error_msg)
            raise requests.exceptions.RequestException(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f'Request failed: {str(e)}'
            logger.error(error_msg)
            raise  # 重新抛出异常

    def reset(self) -> None:
        if self.status != ClientStatus.UNINITIATED:
            self.request_reset()
        super().reset()
        self.status = ClientStatus.UNINITIATED

    @staticmethod
    def get_available_models(env_name: str) -> dict[str, Any]:
        """从服务器获取可用模型列表"""
        url = CONFIG['base_url'] + 'models'
        payload = {'env_name': env_name}
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch models from server: {e}")
            return {'indices': [], 'best_index': -1}

    def shutdown(self) -> None:
        self.alive = False
