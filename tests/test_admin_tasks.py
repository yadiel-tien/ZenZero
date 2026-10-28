"""TaskRunner（控制台「任务」页的后端）回归。

覆盖：注册表校验、启动/停止生命周期、收养（running 判定基于 pid+cmdline 而非内存对象）、
参数白名单（未知任务 / 非法整数 / 注入字符必须被拒）、once 任务退出后可读到日志。
运行：.venv/bin/python tests/test_admin_tasks.py
"""
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.admin.tasks import TaskRunner  # noqa: E402

_tmp = None


def setUp():
    global _tmp
    _tmp = tempfile.mkdtemp(prefix='zz_tasks_')


def tearDown():
    global _tmp
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


def _registry():
    """假任务：long=demo（可收养），once=bye（秒退留退出码）。绝不启动真模块。"""
    return {
        'demo': {'argv': [sys.executable, '-c',
                          'import time\nfor _ in range(200): time.sleep(0.1)'],
                 'marker': 'time.sleep(0.1)', 'kind': 'long', 'desc': '长跑假任务',
                 'opts': {'port': {'flag': '--port', 'type': 'int', 'default': 12345}}},
        'bye': {'argv': [sys.executable, '-c', 'print("hello task"); raise SystemExit(3)'],
                'marker': 'SystemExit(3)', 'kind': 'once', 'desc': '秒退假任务', 'opts': {}},
    }


def _runner():
    return TaskRunner(state_file=os.path.join(_tmp, 'tasks.json'), registry=_registry(),
                      log_root=os.path.join(_tmp, 'logs'))


def test_lifecycle():
    setUp()
    try:
        r = _runner()
        st = r.status()['tasks']
        assert {x['name'] for x in st} == {'demo', 'bye'}
        assert all(not x['running'] for x in st)

        res = r.start('demo')
        assert res['ok'], res
        pid = res['status']['pid']
        assert r.status()['tasks'][0]['running'] or res['status']['running']

        # 重复启动必须被拒
        again = r.start('demo')
        assert not again['ok'] and '已在运行' in again['error'], again

        stopped = r.stop('demo')
        assert stopped['ok'], stopped
        time.sleep(0.2)
        row = next(x for x in r.status()['tasks'] if x['name'] == 'demo')
        assert not row['running'], row

        # once 任务：退出码可见、日志可读
        r.start('bye')
        time.sleep(0.8)
        row = next(x for x in r.status()['tasks'] if x['name'] == 'bye')
        assert not row['running'] and row['exit_code'] == 3, row
        log = r.log('bye')
        assert log['ok'] and any('hello task' in l for l in log['lines']), log
        print('   ok   启动/重复拒绝/停止/退出码/日志 全链路')
    finally:
        tearDown()


def test_validation_and_unknown():
    setUp()
    try:
        r = _runner()
        assert not r.start('nope')['ok'], '未知任务必须拒绝'
        bad = r.start('demo', {'port': 'abc; rm -rf /'})
        assert not bad['ok'] and '整数' in bad['error'], bad
        r2 = _runner()
        noflag = r2._build_cmd('demo', {'port': ''})       # 空值=用默认，不传 flag
        assert noflag[1] is None and '--port' not in ' '.join(noflag[0]), noflag
        print('   ok   未知任务/非法参数被挡，空参数回落默认')
    finally:
        tearDown()


def test_adopt_after_restart():
    """模拟服务重启：新实例读状态文件后，须凭 pid+cmdline 收养仍存活的任务。"""
    setUp()
    try:
        r = _runner()
        res = r.start('demo')
        assert res['ok'], res
        pid = res['status']['pid']

        r2 = TaskRunner(state_file=os.path.join(_tmp, 'tasks.json'), registry=_registry())
        row = next(x for x in r2.status()['tasks'] if x['name'] == 'demo')
        assert row['running'] and row['pid'] == pid, row
        assert r2.stop('demo')['ok']

        # 进程没了：再"重启"一次，必须判 not running（cmdline 校验兜底）
        time.sleep(0.3)
        r3 = TaskRunner(state_file=os.path.join(_tmp, 'tasks.json'), registry=_registry())
        row = next(x for x in r3.status()['tasks'] if x['name'] == 'demo')
        assert not row['running'], row
        print('   ok   收养：活着认领、死了翻牌，全凭 pid + /proc cmdline')
    finally:
        tearDown()


def test_routes_smoke():
    """HTTP 面：/api/admin/tasks* 与注入后的单例联动（测试用临时 state_file，不触生产）。"""
    setUp()
    try:
        import services.admin.routes as routes
        from flask import Flask
        routes._tasks = TaskRunner(state_file=os.path.join(_tmp, 'tasks.json'), registry=_registry())
        app = Flask(__name__)
        app.register_blueprint(routes.bp)
        c = app.test_client()
        try:
            lst = c.get('/api/admin/tasks')
            assert lst.status_code == 200, lst.status_code
            names = {t['name'] for t in lst.get_json()['tasks']}
            assert names == {'demo', 'bye'}, names
            bad = c.post('/api/admin/tasks/start', json={'name': 'nope'})
            assert bad.status_code == 400, bad.status_code
            go = c.post('/api/admin/tasks/start', json={'name': 'demo'})
            assert go.status_code == 200 and go.get_json()['ok'], go.get_json()
            lg = c.get('/api/admin/tasks/log?name=demo')
            assert lg.status_code == 200 and lg.get_json()['ok'], lg.get_json()
            assert c.post('/api/admin/tasks/stop', json={'name': 'demo'}).get_json()['ok']
        finally:
            routes._tasks = None
        print('   ok   路由：列表/未知400/启动/日志/停止（单例已复位）')
    finally:
        tearDown()


def main() -> None:
    for fn in (test_lifecycle, test_validation_and_unknown, test_adopt_after_restart, test_routes_smoke):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 4 TASK-RUNNER TESTS PASSED')


if __name__ == '__main__':
    main()
