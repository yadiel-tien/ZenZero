import threading
import uuid
import time
import traceback
from functools import wraps
from typing import Callable, Any

import numpy as np
from flask import Flask, request, jsonify

from core.player.ai_server import AIServer
from core.network.functions import list_all_indices, read_best_index
from core.utils.logger import get_logger
from services.inference.hub import ServerHub


class PlayApiServer:
    def __init__(self, standalone_hub: bool = True):
        self.app = Flask(__name__)
        self.ai_players: dict[str, AIServer] = {}
        self.clients_live_time: dict[str, float] = {}
        self.lock = threading.Lock()
        self.standalone_hub = standalone_hub
        self.logger = get_logger('play_server')
        
        # 注册路由
        self._setup_routes()
        
        # 启动后台清理线程
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def _require_json(self, f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not request.is_json:
                self.logger.warning(f"Invalid Content-Type: {request.content_type}")
                return jsonify({"error": "Content-Type must be application/json"}), 400
            data = request.get_json(silent=True)
            if data is None:
                self.logger.warning(f"Failed to parse JSON data from {request.remote_addr}")
                return jsonify({"error": "Invalid JSON data"}), 400
            return f(data, *args, **kwargs)
        return wrapper

    def _setup_routes(self):
        @self.app.route('/models', methods=['POST'])
        @self._require_json
        def get_models(data):
            try:
                env_name = data['env_name']
                indices = list_all_indices(env_name)
                best_index = read_best_index(env_name)
                self.logger.info(f"Models requested for {env_name}. Found {len(indices)} versions.")
                return jsonify({'indices': indices, 'best_index': best_index})
            except Exception as e:
                self.logger.error(f"Error listing models: {e}", exc_info=True)
                return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 500

        @self.app.route('/setup', methods=['POST'])
        @self._require_json
        def setup(data):
            try:
                model_id = data['model_id']
                env_name = data['env_class']
                pid = str(uuid.uuid4())
                with self.lock:
                    self.clients_live_time[pid] = time.time()
                    player = AIServer(env_name, model_id, 1000, verbose=False)
                    self.ai_players[pid] = player
                self.logger.info(f"New session created: {pid[:8]}... (Game: {env_name}, Model: {model_id})")
                return jsonify({'pid': pid, 'model_id': player.model_id})
            except Exception as e:
                self.logger.error(f"Failed to setup AI player: {e}", exc_info=True)
                return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 500

        @self.app.route('/update', methods=['POST'])
        @self._require_json
        def update(data):
            try:
                pid = data['pid']
                with self.lock:
                    if pid not in self.ai_players:
                        self.logger.warning(f"Update failed: Session {pid[:8]} not found.")
                        return jsonify({"error": "Player not found"}), 404
                    self.clients_live_time[pid] = time.time()
                    player = self.ai_players[pid]

                state = np.array(data['array'], dtype=np.int32)
                player.update_state(state, data['action'], data['player_to_move'])
                return jsonify({'win_rate': float(player.win_rate)})
            except Exception as e:
                self.logger.error(f"Error during state update for {pid[:8]}: {e}", exc_info=True)
                return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 500

        @self.app.route('/get_action', methods=['POST'])
        @self._require_json
        def get_action(data):
            try:
                pid = data['pid']
                with self.lock:
                    if pid not in self.ai_players:
                        self.logger.warning(f"Action failed: Session {pid[:8]} not found.")
                        return jsonify({"error": "Player not found"}), 404
                    self.clients_live_time[pid] = time.time()
                    player = self.ai_players[pid]
                
                action = player.get_action()
                self.logger.info(f"Session {pid[:8]} moved: {action} (WR: {player.win_rate:.2%})")
                return jsonify({
                    "action": action,
                    'win_rate': float(player.win_rate),
                    "model_id": player.model_id
                })
            except Exception as e:
                self.logger.error(f"Error fetching action for {pid[:8]}: {e}", exc_info=True)
                return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 500

        @self.app.route('/reset', methods=['POST'])
        @self._require_json
        def reset(data):
            try:
                pid = data['pid']
                with self.lock:
                    if pid in self.ai_players:
                        self.ai_players[pid].reset()
                        self.logger.info(f"Session {pid[:8]} reset.")
                        return jsonify({"win_rate": self.ai_players[pid].win_rate})
                return jsonify({"error": "Player not found"}), 404
            except Exception as e:
                self.logger.error(f"Error resetting session {pid[:8]}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/heartbeat', methods=['POST'])
        @self._require_json
        def heartbeat(data):
            try:
                pid = data['pid']
                with self.lock:
                    if pid in self.ai_players:
                        self.clients_live_time[pid] = time.time()
                        return jsonify({'win_rate': float(self.ai_players[pid].win_rate)})
                return jsonify({"error": "Player not found"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _cleanup_loop(self, timeout=60):
        while True:
            time.sleep(timeout)
            now = time.time()
            to_delete = []
            with self.lock:
                for pid, last_beat in self.clients_live_time.items():
                    if now - last_beat > timeout:
                        to_delete.append(pid)
                for pid in to_delete:
                    self.clients_live_time.pop(pid)
                    player = self.ai_players.pop(pid)
                    player.shutdown()
                    self.logger.info(f"Session {pid[:8]} cleaned up due to inactivity.")

    def run(self, host='0.0.0.0', port=5000):
        if self.standalone_hub:
            self.logger.info("Starting background Inference Hub in standalone mode...")
            hub = ServerHub()
            threading.Thread(target=hub.start, daemon=True).start()
            time.sleep(1)
        
        self.logger.info(f"Play API Server running on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)
