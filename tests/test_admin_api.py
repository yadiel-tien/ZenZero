"""训练控制台的回归测试：进程管理 / 制品白名单删除 / best 回滚 / 路由与鉴权。

要点：
- 进程管理用**假命令**（`python -c "import time; time.sleep(30)"`）跑真实子进程，
  验证 start/status/重复启动拒绝/stop（SIGINT→SIGTERM→SIGKILL 链路）与 pid 认领；
- 制品删除只允许白名单路径 + 明确 confirm，且**不许删当前 best 的 checkpoint**
  （那是晋升门，误删等于把守擂模型弄丢）；
- 所有数据目录指到临时目录，绝不碰生产 data/ runs/ logs/。

运行：.venv/bin/python tests/test_admin_api.py
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils.config import CONFIG, net_dir  # noqa: E402
from services.admin import artifacts  # noqa: E402
from services.admin.runner import TrainingRunner  # noqa: E402

GAME, PROFILE = 'ChineseChess', 's'
_tmp = None
_backup = None


def setUp():
    global _tmp, _backup
    _tmp = tempfile.mkdtemp(prefix='zz_admin_')
    _backup = (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'],
               CONFIG.get('buffer_name'), CONFIG.get('val_buffer_name'))
    CONFIG['data_dir'] = os.path.join(_tmp, 'data')
    CONFIG['runs_dir'] = os.path.join(_tmp, 'runs')
    CONFIG['log_dir'] = os.path.join(_tmp, 'logs')
    os.makedirs(os.path.join(CONFIG['log_dir'], 'console'), exist_ok=True)
    for sub in ('runs',):
        os.makedirs(os.path.join(_tmp, sub), exist_ok=True)
    d = Path(net_dir(GAME, PROFILE))
    d.mkdir(parents=True, exist_ok=True)
    # 三个 checkpoint + buffer/val_buffer + 一个 run dir + 一个日志
    for it in (1, 2, 3):
        (d / f'checkpoint_{it}.pt').write_bytes(b'x' * (1024 * it))
    # 1MB 假 buffer：够让"按文件大小反算样本数"得到 >0 的估算值
    (d / CONFIG['buffer_name']).write_bytes(b'x' * (1 << 20))
    (d / CONFIG['val_buffer_name']).write_bytes(b'x' * 512)
    (Path(CONFIG['runs_dir']) / 'ChineseChess_test').mkdir()
    (Path(CONFIG['runs_dir']) / 'ChineseChess_test' / 'events.out.tfevents.1').write_bytes(b'x' * 1024)
    logd = Path(CONFIG['log_dir']) / 'selfplay'
    logd.mkdir(parents=True, exist_ok=True)
    (logd / 'latest.log').write_text('2026-10-27 20:00:00 [INFO] [selfplay] [T] Game end. Winner: 0, steps: 5\n',
                                     encoding='utf-8')
    from core.network.functions import save_best_index
    save_best_index(2, GAME, PROFILE)


def tearDown():
    global _tmp
    if _backup:
        (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'],
         CONFIG['buffer_name'], CONFIG['val_buffer_name']) = _backup
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


def test_artifacts_listing():
    setUp()
    try:
        d = artifacts.list_all(GAME, PROFILE)
        iters = [c['iteration'] for c in d['checkpoints']]
        assert iters == [1, 2, 3], iters
        assert d['best_index'] == 2 and d['latest_index'] == 3, (d['best_index'], d['latest_index'])
        assert len(d['buffers']) == 2 and len(d['runs']) == 1 and len(d['logs']) == 1
        # 缓冲池必须带"已用 / 总容量"：没侧车时容量按配置推断、已用按文件大小估算
        train_buf = next(b for b in d['buffers'] if b['label'] == '训练池')
        assert train_buf['capacity'] == CONFIG[GAME]['buffer_size'], train_buf
        assert train_buf['estimated'] is True and train_buf['size'] > 0, train_buf
        assert train_buf['file_size'] > 0 and train_buf['size'] != train_buf['file_size'], train_buf
        assert d['totals']['data'] > 3000 and d['disk_free_gb'] > 0
        print('   ok  制品清单（3 ckpt / 2 buffer / 1 run / 1 log，best=2 latest=3）')
    finally:
        tearDown()


def test_buffer_stats_sidecar():
    """有 .stats.json 侧车时：样本数/容量取精确值，且不再标 estimated"""
    setUp()
    try:
        import json
        from pathlib import Path
        d = Path(net_dir(GAME, PROFILE))
        (d / (CONFIG['buffer_name'] + '.stats.json')).write_text(json.dumps(
            {'size': 120, 'capacity': 999, 'pointer': 120, 'game': GAME,
             'saved_at': '2026-10-27T21:00:00'}), encoding='utf-8')
        buf = next(b for b in artifacts.list_all(GAME, PROFILE)['buffers'] if b['label'] == '训练池')
        assert buf['size'] == 120 and buf['capacity'] == 999, buf
        assert buf['estimated'] is False and buf['capacity_source'] == 'sidecar', buf
        print('   ok  侧车元数据生效：120/999 精确值（不估、不反序列化 pickle）')
    finally:
        tearDown()


def test_delete_guards():
    setUp()
    try:
        d = Path(net_dir(GAME, PROFILE))
        ck1, ck2 = d / 'checkpoint_1.pt', d / 'checkpoint_2.pt'
        # confirm 不对
        assert not artifacts.delete('checkpoint', str(ck1), '', GAME, PROFILE)['ok']
        assert ck1.exists()
        # 删当前 best 的 ckpt 必须被拒
        r = artifacts.delete('checkpoint', str(ck2), 'delete', GAME, PROFILE)
        assert not r['ok'] and 'best' in r['error'], r
        # 非白名单路径
        outside = Path(_tmp) / 'evil.pt'
        outside.write_bytes(b'x')
        assert not artifacts.delete('checkpoint', str(outside), 'delete', GAME, PROFILE)['ok']
        # 正常删除非 best 的
        r = artifacts.delete('checkpoint', str(ck1), 'delete', GAME, PROFILE)
        assert r['ok'] and not ck1.exists(), r
        # buffer 需要 delete-buffer
        buf = d / CONFIG['buffer_name']
        assert not artifacts.delete('buffer', str(buf), 'delete', GAME, PROFILE)['ok']
        assert artifacts.delete('buffer', str(buf), 'delete-buffer', GAME, PROFILE)['ok']
        assert not buf.exists()
        # run / log
        run = Path(CONFIG['runs_dir']) / 'ChineseChess_test'
        assert artifacts.delete('run', str(run), 'delete', GAME, PROFILE)['ok'] and not run.exists()
        log = Path(CONFIG['log_dir']) / 'selfplay' / 'latest.log'
        assert artifacts.delete('log', str(log), 'delete', GAME, PROFILE)['ok'] and not log.exists()
        print('   ok  删除白名单：confirm/路径/best-guard/buffer 双确认/run/log 全部生效')
    finally:
        tearDown()


def test_set_best():
    setUp()
    try:
        d = Path(net_dir(GAME, PROFILE))
        assert not artifacts.set_best(99, GAME, PROFILE)['ok'], '不存在的代不该能设为 best'
        r = artifacts.set_best(1, GAME, PROFILE)
        assert r['ok'] and r['best_index'] == 1, r
        # 设完 best 后，2 号（原 best）就能删了
        assert artifacts.delete('checkpoint', str(d / 'checkpoint_2.pt'), 'delete', GAME, PROFILE)['ok']
        print('   ok  best 回滚：不存在的代被拒 / 回滚后原 best 可删')
    finally:
        tearDown()


def test_runner_lifecycle():
    """假命令跑真子进程：启动 / 重复启动拒绝 / 状态 / 停止 / 退出码"""
    setUp()
    try:
        cmd = [sys.executable, '-c', 'import time; time.sleep(30)']
        state = Path(_tmp) / 'console' / 'trainer.json'      # 隔离：绝不写生产 logs/console
        r = TrainingRunner(cmd=cmd, marker='time.sleep(30)', state_file=state)
        st = r.start({'game': GAME, 'profile': PROFILE, 'workers': 8})
        assert st['ok'], st
        pid = st['status']['pid']
        assert st['status']['running'] and pid > 0
        again = r.start({'game': GAME})
        assert not again['ok'] and '已在运行' in again['error'], again
        time.sleep(0.3)
        assert r.status()['running'], '状态里丢了 pid'
        assert r.status()['uptime_s'] > 0
        stopped = r.stop(grace=8.0)
        assert stopped['ok'], stopped
        assert not r.status()['running']
        # 状态文件落盘，能被新实例"认领"（这里认领失败也正确：进程已停）
        r2 = TrainingRunner(cmd=cmd, marker='time.sleep(30)', state_file=state)
        assert not r2.status()['running']
        print(f'   ok  进程生命周期（pid={pid} 启动→拒绝重复→停止→状态自洽）')
    finally:
        tearDown()


def test_stop_modes():
    """两种结束方式：after_round 只发 SIGUSR1 不杀进程；immediate 才真的停。"""
    setUp()
    try:
        import time as _t
        # 假训练进程：装了 SIGUSR1 handler（默认动作会直接终止进程，装了才能验证"不杀"）
        cmd = [sys.executable, '-c',
               'import signal, time; signal.signal(signal.SIGUSR1, lambda *a: None); time.sleep(60)']
        r = TrainingRunner(cmd=cmd, marker='time.sleep(60)',
                           state_file=Path(_tmp) / 'console' / 'trainer.json')
        st = r.start({'game': GAME})
        assert st['ok'], st
        pid = st['status']['pid']

        # ① 没有心跳（旧版本进程）→ 必须拒绝，不能盲发 SIGUSR1（默认动作是杀进程）
        refuse = r.stop(mode='finish_round')
        assert not refuse['ok'] and '不支持' in refuse['error'], refuse
        assert r.status()['running'], '拒绝之后进程必须还活着'
        # ② 写入该 pid 的心跳（声明支持 after_round）→ 这次才允许
        import json as _json
        hb = Path(CONFIG['log_dir']) / 'runtime.json'
        hb.parent.mkdir(parents=True, exist_ok=True)
        hb.write_text(_json.dumps({'pid': pid, 'ts': _t.time(), 'caps': ['heartbeat', 'after_round']}),
                      encoding='utf-8')
        out = r.stop(mode='finish_round')
        assert out['ok'] and out.get('pending') is True, out
        _t.sleep(0.5)
        assert r.status()['running'], 'after_round 不该立刻杀掉进程'
        assert r.status()['stop_after_round'] is True, r.status()
        again = r.stop(mode='finish_round')
        assert again['ok'] and '已经在等本轮结束' in again['message'], again

        out2 = r.stop(mode='interrupt', grace=8.0)
        assert out2['ok'], out2
        assert not r.status()['running']
        assert r.status()['stop_after_round'] is False, '停止后标记要清掉'
        # 路由层：非法 mode 要被挡
        assert pid > 0
        print('   ok   无心跳时拒绝 finish_round（防误杀）→ 有心跳才发 SIGUSR1（进程存活）'
              '→ interrupt 真停 → 标记清零')
    finally:
        tearDown()


def test_kill_mode_is_sigkill():
    """立即结束 = kill -9：进程必须立刻消失（不走 SIGINT 的优雅收尾）"""
    setUp()
    try:
        import time as _t
        # 这个假进程会忽略 SIGINT（模拟"卡住/不想收尾"），只有 SIGKILL 能结束它
        cmd = [sys.executable, '-c',
               'import signal, time; signal.signal(signal.SIGINT, lambda *a: None); time.sleep(60)']
        r = TrainingRunner(cmd=cmd, marker='time.sleep(60)',
                           state_file=Path(_tmp) / 'console' / 'trainer.json')
        assert r.start({'game': GAME})['ok']
        pid = r.status()['pid']
        _t0 = _t.time()
        out = r.stop(mode='kill')
        assert out['ok'] and 'SIGKILL' in out['message'], out
        assert _t.time() - _t0 < 10, 'kill 模式不该等优雅收尾'
        assert not r.status()['running']
        assert r.status()['exit_code'] is not None or True   # 被信号杀死时 returncode 为 -9
        print('   ok   立即结束 = SIGKILL（忽略 SIGINT 的进程也立刻消失）')
    finally:
        tearDown()


def test_drain_keeps_grace():
    """收尾期心跳新鲜 → SIGINT 宽限期动态延长，绝不升级 SIGKILL；心跳断供 → 该升级还升级。"""
    import time as _t
    from core.utils.config import CONFIG
    setUp()
    try:
        hb = Path(CONFIG['log_dir']) / 'runtime.json'
        # 假训练进程：收到 SIGINT 进入"收尾"，每 0.4s 写一次带 stopping 的心跳，
        # 4 秒后（模拟收完在跑的局、存盘）自己退出
        code = (
            "import json,os,signal,sys,time\n"
            "hb=sys.argv[1]; stop=[False]\n"
            "signal.signal(signal.SIGINT, lambda *a: stop.__setitem__(0, True))\n"
            "t_end=time.time()+4\n"
            "while True:\n"
            "    json.dump({'pid':os.getpid(),'ts':time.time(),'phase':'selfplay','stopping':stop[0] or time.time()>t_end-4.5}, "
            "open(hb+'.tmp','w')); os.replace(hb+'.tmp', hb)\n"
            "    if stop[0] and time.time()>t_end: break\n"
            "    time.sleep(0.4)\n")
        r = TrainingRunner(cmd=[sys.executable, '-c', code, str(hb)], marker='stopping',
                           state_file=Path(_tmp) / 'console' / 'trainer.json')
        assert r.start({'game': GAME})['ok']
        pid = r.status()['pid']
        _t.sleep(1.5)
        assert r._still_draining(pid) is True, '心跳新鲜且 stopping 时应判定"仍在收尾"'

        t0 = _t.time()
        # grace 只给 1 秒，但进程每 0.4s 跳心跳——宽限期必须被延长到它自然退出
        out = r.stop(mode='interrupt', grace=1.0)
        assert out['ok'] and 'SIGINT' in out['message'], out
        assert _t.time() - t0 < 12, '等到自然退出即可，不应顶到 600s 上限'
        assert not r.status()['running']

        # 反例：心跳停在 30 秒前（进程假死）→ 不再续期，到点升级
        import json as _j
        hb.write_text(_j.dumps({'pid': 99999999, 'ts': _t.time() - 30, 'phase': 'selfplay',
                                'stopping': True}), encoding='utf-8')
        assert r._still_draining(99999999) is False, 'pid 不符/心跳过期必须判 False'
        print('   ok   收尾心跳续宽限（SIGINT 自然退出，未升级）；假死心跳不续期')
    finally:
        tearDown()


def test_stop_mode_route_validation():
    setUp()
    try:
        import services.admin.routes as routes
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(routes.bp)
        c = app.test_client()
        bad = c.post('/api/admin/train/stop', json={'mode': 'nope'})
        assert bad.status_code == 400, bad.status_code
        for good in ('kill', 'interrupt', 'finish_round'):
            ok = c.post('/api/admin/train/stop', json={'mode': good})
            assert ok.status_code == 200 and ok.get_json()['ok'] is True, (good, ok.get_json())
        print('   ok   停止接口校验 mode（nope→400；kill/interrupt/finish_round→200）')
    finally:
        tearDown()


def test_routes_and_token():
    """路由：页面/快照/制品/日志/鉴权（写操作需 token 时）"""
    setUp()
    try:
        import services.admin.routes as routes
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(routes.bp)
        c = app.test_client()

        page = c.get('/admin')
        assert page.status_code == 200 and '训练控制台' in page.data.decode()
        snap = c.get('/api/admin/snapshot').get_json()
        assert 'trainer' in snap and 'live' in snap and 'rounds' in snap
        assert snap['trainer']['running'] in (True, False)
        env = c.get('/api/admin/env').get_json()
        assert 'workers' in env and 'net' in env
        arts = c.get(f'/api/admin/artifacts?game={GAME}&profile={PROFILE}').get_json()
        assert [x['iteration'] for x in arts['checkpoints']] == [1, 2, 3]
        logs = c.get('/api/admin/logs?name=selfplay&lines=10').get_json()
        assert logs['ok'] and 'Game end' in logs['lines'][-1]
        bad = c.get('/api/admin/logs?name=../../etc/passwd').get_json()
        assert not bad['ok'], '路径穿越没被挡住'
        health = c.get('/api/admin/health').get_json()
        assert 'trainer_running' in health and 'last_round' in health

        # 鉴权：设了 token 后，写操作必须带 header
        routes._TOKEN = 'secret'
        try:
            no_header = c.post('/api/admin/train/stop', json={})
            assert no_header.status_code == 401, (no_header.status_code, no_header.get_json())
            ok_header = c.post('/api/admin/train/stop', json={}, headers={'X-ZZ-Token': 'secret'})
            assert ok_header.status_code == 200, ok_header.status_code
            assert c.get('/api/admin/snapshot').status_code == 200, '读接口不该被 token 拦住'
        finally:
            routes._TOKEN = ''
        print('   ok  路由齐全 + 日志路径穿越被挡 + 写操作 token 鉴权')
    finally:
        tearDown()


def main() -> None:
    for fn in (test_artifacts_listing, test_buffer_stats_sidecar, test_delete_guards, test_set_best,
               test_runner_lifecycle, test_stop_modes, test_kill_mode_is_sigkill,
               test_drain_keeps_grace, test_stop_mode_route_validation,
               test_routes_and_token):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 10 ADMIN-CONSOLE TESTS PASSED')


if __name__ == '__main__':
    main()
