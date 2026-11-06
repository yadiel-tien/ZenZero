import os
import threading
import uuid
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

from core.player.ai_server import AIServer
from core.network.functions import list_all_indices, read_best_index
from core.utils.config import net_dir
from core.utils.logger import get_logger
import core.utils.config as app_config
from services.inference.hub import ServerHub

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / 'apps' / 'web'
DOCS_DIR = ROOT_DIR / 'docs'   # 项目文档集（静态 HTML，/docs/ 只读托管）
GRAPHICS_DIR = ROOT_DIR / 'apps' / 'assets' / 'graphics'
SOUND_DIR = ROOT_DIR / 'apps' / 'assets' / 'sound'

# ============ 显存需求估算（/devices 空间判定）============
# 2026-10-30：前端按「实际要加载的模型集合」（1 个 AI 或 2 个 AI、各多大）判断 GPU 空间；
# 不够则设备标记 unavailable（无切换按钮）。估算口径与 Net._fit_device 一致：
# 参数量×4字节×1.3（CUDA context/cuDNN/碎片余量）+ 600MiB 固定预留。
_MODEL_PARAM_CACHE: dict = {}   # checkpoint 路径 -> (mtime, 参数量)


def _model_param_count(path: str):
    """用 meta 设备读取 checkpoint 参数量（不占显存/内存），按 mtime 缓存；失败返回 None。"""
    import torch
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _MODEL_PARAM_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]

    def count_tensors(obj) -> int:
        n = 0
        for v in (obj.values() if isinstance(obj, dict) else []):
            if torch.is_tensor(v):
                n += v.numel()
            elif isinstance(v, dict):
                n += sum(t.numel() for t in v.values() if torch.is_tensor(t))
        return n

    for kw in (dict(map_location='meta', weights_only=True), dict(map_location='meta')):
        try:
            n = count_tensors(torch.load(path, **kw))
            if n:
                _MODEL_PARAM_CACHE[path] = (mtime, n)
                return n
        except Exception:
            continue
    return None


def _vram_need_bytes(env_name: str, model_specs: list, default_best: str | None = None):
    """同一 GPU 上加载指定模型集合的预估显存需求（bytes）；无法估算返回 None（前端保持原显示，不误禁）。"""
    specs = [s for s in model_specs if str(s).strip()]
    if not specs and default_best:
        specs = [default_best]
    if not specs:
        return None
    from services.inference.functions import get_checkpoint_path
    total = 0
    for spec in specs:
        prof, _, idx_s = str(spec).partition(':')
        try:
            idx = int(idx_s)
        except ValueError:
            return None
        n = _model_param_count(get_checkpoint_path(env_name, idx, (prof or None)))
        if not n:
            return None
        total += n
    if not total:
        return None
    return int(total * 4 * 1.3 + 600 * 1024 * 1024)


class PlayApiServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.ai_players: dict[str, AIServer] = {}
        self.clients_live_time: dict[str, float] = {}
        self.lock = threading.Lock()
        self.logger = get_logger('play_server')
        # 进程内推理引擎管理中心，所有AIServer共享
        self.hub = ServerHub()
        
        # 注册路由 + 训练控制台（/admin 页面与 /api/admin/* 接口）
        self._setup_routes()
        from services.admin.routes import bp as admin_bp
        self.app.register_blueprint(admin_bp)
        
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
        # ---------- Web 前端（apps/web）托管 + 资源 + CORS ----------
        @self.app.after_request
        def add_cors(resp):
            # 允许跨机部署时从任意页面（含 file://）访问 API
            resp.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
            resp.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            resp.headers['Access-Control-Max-Age'] = '86400'
            # 缓存策略：页面与代码每次经 ETag/Last-Modified 再验证（改版后 F5 即最新，
            # 未变则 304 仅几百字节）；图片素材低频变动，缓存 7 天
            path = request.path
            if path.startswith('/assets/'):
                resp.headers['Cache-Control'] = 'public, max-age=604800'
            elif request.method == 'GET' and (
                    path in ('/',) or path.startswith(('/js/', '/css/')) or path.endswith('.html')):
                resp.headers['Cache-Control'] = 'no-cache'
            return resp

        @self.app.route('/')
        def web_index():
            return send_from_directory(WEB_DIR, 'index.html')

        @self.app.route('/docs/')
        def web_docs_index():
            # 文档集目录页：五篇静态 HTML，互相用相对链接
            return send_from_directory(DOCS_DIR, 'index.html')

        @self.app.route('/docs/<path:rel>')
        def web_docs(rel):
            return send_from_directory(DOCS_DIR, rel)

        @self.app.route('/css/<path:rel>')
        def web_css(rel):
            return send_from_directory(WEB_DIR / 'css', rel)

        @self.app.route('/js/<path:rel>')
        def web_js(rel):
            return send_from_directory(WEB_DIR / 'js', rel)

        @self.app.route('/assets/graphics/<path:rel>')
        def web_graphics(rel):
            return send_from_directory(GRAPHICS_DIR, rel)

        @self.app.route('/assets/sound/<path:rel>')
        def web_sound(rel):
            return send_from_directory(SOUND_DIR, rel)

        def probe_device_info() -> dict:
            info = {'device': app_config.CONFIG['device'], 'device_note': app_config.DEVICE_NOTE, 'vram': None}
            dev = str(info['device'])
            if dev.startswith('cuda') and torch.cuda.is_available():
                try:
                    free, total = torch.cuda.mem_get_info(torch.device(dev).index or 0)
                    info['vram'] = {'free': round(free / 2 ** 30, 2), 'total': round(total / 2 ** 30, 2)}
                except Exception:
                    pass
            return info

        @self.app.route('/info', methods=['GET'])
        def server_info():
            from core.utils.version import __version__
            return jsonify({**probe_device_info(), 'version': __version__})

        @self.app.route('/devices', methods=['GET'])
        def devices_list():
            from core.utils.config import enumerate_devices
            # 可选 ?env=ChineseChess&models=mini:50,m:12 —— 按「实际要加载的模型集合」判定显存空间
            env = (request.args.get('env') or '').strip() or None
            models_arg = (request.args.get('models') or '').strip()
            need, n_models = None, 0
            if env:
                specs = [p.strip() for p in models_arg.split(',') if p.strip()]
                default_best = None
                if not specs:
                    # 未指定模型（页面加载期）：按默认档位 best_index 估算
                    try:
                        prof = _default_profile(env)
                        cur = (_profile_models(env) or {}).get(prof) or {}
                        best = int(cur.get('best_index', -1))
                        if best >= 0:
                            default_best = f'{prof}:{best}'
                    except Exception:
                        pass
                need = _vram_need_bytes(env, specs, default_best)
                n_models = len(specs) or (1 if default_best else 0)
            devices = [d for d in enumerate_devices() if d['status'] != 'absent']
            if need:
                for d in devices:
                    # 空间不够 -> 直接 unavailable（前端不显示切换按钮）；active 不受影响
                    if str(d['id']).startswith('cuda') and d.get('vram') and d['status'] != 'active':
                        d['fits'] = d['vram']['free'] * 2 ** 30 >= need
                        if not d['fits']:
                            d['status'] = 'unavailable'
                            d['reason'] = (f'显存不足：空闲 {d["vram"]["free"]:.1f}G '
                                           f'< 加载 {n_models} 个模型需 ~{need / 2 ** 30:.1f}G，暂不可切换')
            return jsonify({'current': app_config.CONFIG['device'],
                            'device_note': app_config.DEVICE_NOTE,
                            'devices': devices})

        @self.app.route('/device', methods=['POST'])
        @self._require_json
        def device_switch(data):
            from core.utils.config import resolve_device
            target = str(data.get('device', ''))
            try:
                dev = resolve_device(target)
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 400
            old_dev = app_config.CONFIG['device']
            if dev != old_dev:
                err = self.hub.relocate_all(dev, old_dev)
                if err:
                    self.logger.warning(f'Device switch {old_dev} -> {dev} failed: {err}')
                    return jsonify({'error': f'切换失败：{err}'}), 409
                app_config.CONFIG['device'] = dev
                app_config.DEVICE = dev
                app_config.DEVICE_NOTE = ''
                self.logger.info(f'Inference device switched (manual): {old_dev} -> {dev}')
            return jsonify({'device': dev})

        def _profile_models(env_name: str) -> dict:
            """每个档位各自的模型列表 + 最新文件时间（用来判断"哪一档有刚训出来的新模型"）。"""
            out = {}
            presets = dict(app_config.CONFIG[env_name].get('presets') or {'m': {}})
            try:                                   # 内置 ∪ 自定义命名配置（data/profiles.json）
                presets.update(app_config.available_profiles(env_name))
            except Exception:                      # noqa: BLE001 配置读取失败不拦对弈
                pass
            for prof in presets:
                d = net_dir(env_name, prof)
                if not os.path.isdir(d):
                    continue
                pts = [f for f in os.listdir(d) if f.endswith('.pt')]
                idx = list_all_indices(env_name, prof)
                if not idx:
                    continue
                try:
                    newest = max(os.path.getmtime(os.path.join(d, f)) for f in pts) if pts else 0
                except OSError:
                    newest = 0
                out[prof] = {'indices': sorted(idx), 'best_index': read_best_index(env_name, prof),
                             'newest_ts': newest}
            return out

        def _default_profile(env_name: str) -> str:
            """对弈默认用"最近有产出的档位"——训练在 s 档，面板就该默认开 s 档的新模型。"""
            found = _profile_models(env_name)
            if not found:
                return app_config.get_profile()
            return max(found, key=lambda p: found[p]['newest_ts'])

        @self.app.route('/models', methods=['POST'])
        @self._require_json
        def get_models(data):
            try:
                env_name = data['env_name']
                found = _profile_models(env_name)
                prof = _default_profile(env_name)
                cur = found.get(prof) or {}
                indices = cur.get('indices', [])
                self.logger.info(f"Models requested for {env_name}: profile={prof}, {len(indices)} versions."
                                 f" (profiles: {[(k, len(v['indices'])) for k, v in found.items()]})")
                return jsonify({'indices': indices,
                                'best_index': cur.get('best_index', -1),
                                'profile': prof,
                                'profiles': {p: {'indices': v['indices'],
                                                 'best_index': v['best_index']}
                                             for p, v in found.items()}})
            except Exception as e:
                self.logger.error(f"Error listing models: {e}", exc_info=True)
                return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 500

        @self.app.route('/setup', methods=['POST'])
        @self._require_json
        def setup(data):
            try:
                model_id = data['model_id']
                env_name = data['env_class']
                profile = (data.get('profile') or _default_profile(env_name)).strip().lower()
                # 可选 AI 参数（前端"AI 设置"，2026-10-30）：n_sim 模拟次数 / temp 采样温度；
                # 缺省用按设备档位与纯 argmax（原行为）
                try:
                    n_sim_req = int(data.get('n_sim') or 0)
                except (TypeError, ValueError):
                    n_sim_req = 0
                try:
                    temp_req = float(data.get('temp') or 0)
                except (TypeError, ValueError):
                    temp_req = 0.0
                pid = str(uuid.uuid4())
                with self.lock:
                    self.clients_live_time[pid] = time.time()
                    # CPU 推理单批 ~15s，1000 模拟要 5 分钟；降档保持可玩（GPU 保持原值）
                    n_sim = n_sim_req if 10 <= n_sim_req <= 20000 else (
                        1000 if app_config.CONFIG['device'] != 'cpu' else 60)
                    temp = min(max(temp_req, 0.0), 10.0)
                    player = AIServer(env_name, model_id, n_sim, verbose=False, hub=self.hub,
                                      profile=profile, temperature=temp)  # 用户对比体感：暂随 CONFIG K=64；改回纯搜索加 batch_leaves=1
                    self.ai_players[pid] = player
                device = app_config.CONFIG['device']
                # 模型加载可能刚把设备降级成 CPU（n_sim 档位判定发生在构造前，读到的还是 cuda）：
                # 事后补降档，否则 m 档 CPU 1000 sims ≈ 5 分钟一手，整机 CPU 被打满形同死机
                if n_sim_req <= 0 and str(device) == 'cpu' and player._n_simulation > 60:
                    player._n_simulation = 60
                    n_sim = 60
                note = app_config.DEVICE_NOTE
                if note:
                    self.logger.warning(f'Session {pid[:8]} running on {device}: {note}')
                else:
                    self.logger.info(f"New session created: {pid[:8]}... (Game: {env_name}, Model: {model_id}, "
                                 f"Device: {device}, Simulations: {n_sim}, Temperature: {temp})")
                return jsonify({'pid': pid, 'model_id': player.model_id, 'profile': profile,
                                'device': device, 'device_note': note})
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
                    if getattr(player, '_dead', False):   # 已 reset（退局竞态下的在途请求）
                        return jsonify({"error": "Player not found"}), 404

                state = np.array(data['array'], dtype=np.int32)
                player.update_state(state, data['action'], data['player_to_move'])
                if getattr(player, '_dead', False):   # 跑完的瞬间被退局（reset）
                    return jsonify({"error": "Player not found"}), 404
                return jsonify({'win_rate': float(player.win_rate),
                                'wdl': list(player.wd_probs) if player.wd_probs else None})
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
                    if getattr(player, '_dead', False):
                        return jsonify({"error": "Player not found"}), 404
                
                action = player.get_action()
                self.logger.info(f"Session {pid[:8]} moved: {action} (WR: {player.win_rate:.2%})")
                return jsonify({
                    "action": action,
                    'win_rate': float(player.win_rate),
                    'wdl': list(player.wd_probs) if player.wd_probs else None,
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
                        pl = self.ai_players[pid]
                        return jsonify({"win_rate": pl.win_rate,
                                        "wdl": list(pl.wd_probs) if pl.wd_probs else None})
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

    def run(self, host='0.0.0.0', port=5000, debug: bool = False):
        try:
            d = self.app.test_request_context('/info')
            with d:
                info = self.app.view_functions['server_info']().get_json()
            msg = f"Inference device: {info['device']}"
            if info.get('vram'):
                msg += f" | VRAM free {info['vram']['free']}GiB / {info['vram']['total']}GiB"
            if info.get('device_note'):
                msg += f" | {info['device_note']}"
            self.logger.info(msg)
        except Exception as e:
            self.logger.warning(f'device probe failed: {e}')
        from core.utils.version import full as _full
        self.logger.info(f"ZenZero v{_full()}  服务已启动  http://{host}:{port}/   "
                         f"控制台 /admin   文档 /docs/   （Ctrl-C 停止；训练进程不受影响）")
        self.app.run(host=host, port=port, debug=debug)
