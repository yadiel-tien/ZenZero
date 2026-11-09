"""服务页（Hub / 对弈服务 / 显存）后端回归。

覆盖两条链：
- ServerHub.snapshot()/release_idle()：清单字段齐全、只动 users==0 的引擎；
- /api/admin/services*：CURRENT 缺席时的降级、会话清单、踢会话、释放按钮 409/正常。

对弈服务不写真 PlayApiServer（那要加载 torch 引擎）；用假实例注入 play_api.CURRENT，
routes 只依赖少数几个属性（hub/lock/ai_players/clients_live_time/started_ts/listen）。

运行：.venv/bin/python tests/test_services_tab.py
"""
import os
import queue
import shutil
import sys
import tempfile
import threading
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils.config import CONFIG  # noqa: E402

_backup = None
_tmp = None


def setUp():
    global _backup, _tmp
    _tmp = tempfile.mkdtemp(prefix='zz_svc_')
    _backup = (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'])
    CONFIG['data_dir'] = os.path.join(_tmp, 'data')
    CONFIG['runs_dir'] = os.path.join(_tmp, 'runs')
    CONFIG['log_dir'] = os.path.join(_tmp, 'logs')
    os.makedirs(os.path.join(CONFIG['log_dir'], '_state'), exist_ok=True)


def tearDown():
    global _backup, _tmp
    if _backup:
        CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'] = _backup
        _backup = None
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


class FakeEngine:
    """只带 snapshot()/release_idle() 会用到的最小表面。"""

    def __init__(self, name: str, users_hint: int = 0):
        self.name = name
        self.env_name = 'Gomoku'
        self.profile = 's'
        self.model_index = 7
        self.arch_name = 'resnet_v1'
        self.feature_version = 3
        self.start_time = time.time() - 42
        self.request_queue: queue.Queue = queue.Queue()
        self.dead = False

    def _model_device(self):
        return 'cpu'

    def infer_stats(self):
        return {'req_per_s': 12.3, 'fwd_ms_mean': 1.5, 'batch_recent_mean': 4}

    def shutdown(self):
        self.dead = True


def _mk_hub():
    from services.inference.hub import ServerHub
    # 拉长回收周期：测的是"手动释放"，不是后台 evict 线程自己抢跑
    return ServerHub(evict_interval=10_000)


class FakePlayer:
    def __init__(self, game='Gomoku', model_id=7, engine='Gomoku_7_s'):
        self.env_class = type(game, (), {})   # routes 用 env_class.__name__ 显示棋种
        self.profile = 's'
        self.model_id = model_id
        self._n_simulation = 100
        self.temperature = 0.0
        self._engine_name = engine
        self._dead = False
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1
        self._dead = True


class FakeServer:
    def __init__(self):
        from core.utils.logger import get_logger
        self.logger = get_logger('play_server')   # 真 PlayApiServer 有 logger，routes 会写日志
        self.hub = _mk_hub()
        self.lock = threading.Lock()
        self.ai_players = {}
        self.clients_live_time = {}
        self.started_ts = time.time() - 100
        self.listen = {'host': '127.0.0.1', 'port': 5000}

    def cleanup_stale(self, timeout: float = 60.0):
        now = time.time()
        cleaned = []
        with self.lock:
            for pid, last in list(self.clients_live_time.items()):
                if now - last > timeout:
                    self.clients_live_time.pop(pid, None)
                    p = self.ai_players.pop(pid, None)
                    if p:
                        p.shutdown()
                    cleaned.append(pid)
        return cleaned


def _app():
    from flask import Flask
    from services.admin import routes
    app = Flask(__name__)
    app.register_blueprint(routes.bp)
    return app.test_client()


def test_hub_snapshot_and_release_idle():
    hub = _mk_hub()
    e_idle = FakeEngine('Gomoku_7_s')
    e_busy = FakeEngine('Gomoku_9_s')
    hub.engines = {'Gomoku_7_s': e_idle, 'Gomoku_9_s': e_busy}
    hub.ref_counts = {'Gomoku_7_s': 0, 'Gomoku_9_s': 2}
    snap = {e['name']: e for e in hub.snapshot()}
    assert snap['Gomoku_7_s']['idle'] and snap['Gomoku_7_s']['users'] == 0
    assert not snap['Gomoku_9_s']['idle'] and snap['Gomoku_9_s']['device'] == 'cpu'
    assert snap['Gomoku_9_s']['req_per_s'] == 12.3
    freed = hub.release_idle()
    assert freed == ['Gomoku_7_s'] and e_idle.dead and not e_busy.dead
    assert 'Gomoku_9_s' in hub.engines, '有使用方的引擎绝不能被动'
    assert hub.release_idle() == [], '再点一次没有空闲引擎就是空操作'
    hub.stop_event.set()
    print('   ok  ServerHub.snapshot/release_idle：只动空闲、字段齐全')


def test_services_api():
    from services.inference import play_api
    setUp()
    try:
        c = _app()
        play_api.CURRENT = None
        r = c.get('/api/admin/services').get_json()
        assert r['ok'] and r['server'] is None and r['note'], r
        j = c.post('/api/admin/services/release', json={})
        assert j.status_code == 409, j.status_code

        srv = FakeServer()
        play_api.CURRENT = srv
        e = FakeEngine('Gomoku_7_s')
        srv.hub.engines = {'Gomoku_7_s': e}
        srv.hub.ref_counts = {'Gomoku_7_s': 1}
        p = FakePlayer()
        srv.ai_players = {'abcdef12-3456': p}
        srv.clients_live_time = {'abcdef12-3456': time.time()}
        d = c.get('/api/admin/services').get_json()
        assert d['server']['sessions'][0]['pid'] == 'abcdef12', d['server']
        assert d['server']['sessions'][0]['game'] == 'Gomoku'
        assert d['hub']['engines'][0]['name'] == 'Gomoku_7_s'
        assert d['idle_engines'] == [], '会话还挂着，不算空闲'

        r = c.post('/api/admin/services/release', json={}).get_json()
        assert r['ok'] and r['freed'] == [] and r['busy'] == ['Gomoku_7_s'], r

        # 踢会话 → 引用没了 → 再释放就掉引擎
        k = c.post('/api/admin/services/session_kill', json={'pid': 'abcd'}).get_json()
        assert k['ok'] and k['killed'] == ['abcdef12'], k
        assert p.shutdowns == 1
        srv.hub.ref_counts['Gomoku_7_s'] = 0          # 模拟 player.shutdown 里的 hub.release
        r = c.post('/api/admin/services/release', json={}).get_json()
        assert r['freed'] == ['Gomoku_7_s'] and e.dead, r

        # 超时会话清理
        srv.ai_players = {'stale0000-1': FakePlayer()}
        srv.clients_live_time = {'stale0000-1': time.time() - 3600}
        cl = c.post('/api/admin/services/cleanup', json={}).get_json()
        assert cl['ok'] and cl['cleaned'] == ['stale000'], cl
        assert srv.ai_players == {}

        bad = c.post('/api/admin/services/session_kill', json={'pid': 'ff'})
        assert bad.status_code == 404, bad.status_code
        print('   ok  /api/admin/services*：降级/清单/释放/踢会话/清超时 全链路')
    finally:
        play_api.CURRENT = None
        tearDown()


def main() -> None:
    for fn in (test_hub_snapshot_and_release_idle, test_services_api):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 2 SERVICES-TAB TESTS PASSED')


if __name__ == '__main__':
    main()
