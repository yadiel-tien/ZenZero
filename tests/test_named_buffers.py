"""命名数据仓与训练起点回归（core/utils/buffers.py + replay 落点 + 表单旋钮）。

运行：.venv/bin/python tests/test_named_buffers.py
"""
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils import buffers                              # noqa: E402
from core.utils.config import CONFIG                         # noqa: E402

_tmp = None
_data_backup = None


def setUp():
    global _tmp, _data_backup
    _tmp = tempfile.mkdtemp(prefix='zz_buf_')
    _data_backup = CONFIG['data_dir']
    CONFIG['data_dir'] = os.path.join(_tmp, 'data')
    os.makedirs(os.path.join(CONFIG['data_dir'], 'ChineseChess'), exist_ok=True)


def tearDown():
    global _tmp
    CONFIG.pop('buffer_dir', None)
    if _data_backup:
        CONFIG['data_dir'] = _data_backup
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


def test_resolve_start_from_matrix():
    """起点决策纯函数矩阵（④）。"""
    exists = lambda i: i in (24, 34, 48)                    # noqa: E731
    compat = lambda i: i == -1 or i >= 34                   # 34 起是 epoch2（ noqa: E731）
    ok, note, err = buffers.resolve_start_from('auto', 48, 48, exists, compat)
    assert ok == 48 and not err and '续训' in note
    ok, note, err = buffers.resolve_start_from('auto', 24, 24, exists, compat)
    assert ok == -1 and '新网络' in note                     # best 不兼容表示版本 → 新网络
    ok, note, err = buffers.resolve_start_from('fresh', 48, 48, exists, compat)
    assert ok == -1 and '49' in note and '从头训练' in note    # 随机初始化，代数接续
    ok, note, err = buffers.resolve_start_from('34', 48, 48, exists, compat)
    assert ok == 34 and '分叉' in note and 'max' not in note and '49' in note
    bad = buffers.resolve_start_from('99', 48, 48, exists, compat)
    assert bad[2] and '不存在' in bad[2]
    bad = buffers.resolve_start_from('24', 48, 48, exists, compat)
    assert bad[2] and '不兼容' in bad[2]
    bad = buffers.resolve_start_from('huh', 48, 48, exists, compat)
    assert bad[2] and '未知起点' in bad[2]
    print('   ok   起点矩阵：auto/fresh/分叉/不存在/不兼容/乱值')


def test_registry_and_gates():
    """命名仓：创建校验、注册表扫描（default@ + named）、表示版本闸门、原位仓拒写 meta。"""
    setUp()
    try:
        assert not buffers.create_named('ChineseChess', 'Bad Name')[0]
        assert not buffers.create_named('ChineseChess', 'x' * 40)[0]
        ok, _, meta = buffers.create_named('ChineseChess', 'warm128', note='s 热身共享')
        assert ok and meta['feature_version'] == 2
        ok2, msg2, _ = buffers.create_named('ChineseChess', 'warm128')
        assert not ok2 and '已存在' in msg2

        lst = {b['ref']: b for b in buffers.list_buffers('ChineseChess')}
        assert 'warm128' in lst and 'default@m' in lst and 'default@s' in lst
        assert lst['warm128']['note'] == 's 热身共享'

        d, err, _ = buffers.resolve_for_train('warm128', 'ChineseChess')
        assert d and not err and str(d).endswith(os.path.join('buffers', 'warm128'))
        d, err, _ = buffers.resolve_for_train('', 'ChineseChess')
        assert d is None and not err, '空引用=原位默认，不注入 buffer_dir'
        d, err, _ = buffers.resolve_for_train('default@s', 'ChineseChess')
        assert d and str(d).endswith(os.path.join('ChineseChess', 's'))

        # 表示版本闸门：改 meta 成 v1 → 拒
        mp = os.path.join(CONFIG['data_dir'], 'ChineseChess', 'buffers', 'warm128', 'meta.json')
        m = json.load(open(mp, encoding='utf-8')); m['feature_version'] = 1
        json.dump(m, open(mp, 'w', encoding='utf-8'))
        d, err, _ = buffers.resolve_for_train('warm128', 'ChineseChess')
        assert d is None and '表示版本' in err

        # update_meta：命名仓写入；default 原位仓跳过（不产 meta.json）
        buffers.update_meta('ChineseChess', d if d else os.path.dirname(mp),
                            size=123, source={'model': 's', 'iteration': 48})
        mm = json.load(open(os.path.join(CONFIG['data_dir'], 'ChineseChess', 'buffers',
                                         'warm128', 'meta.json'), encoding='utf-8'))
        assert mm['size'] == 123 and mm['source']['iteration'] == 48
        mdir = os.path.join(CONFIG['data_dir'], 'ChineseChess')     # default@m 原位
        buffers.update_meta('ChineseChess', mdir, size=9)
        assert not os.path.exists(os.path.join(mdir, 'meta.json')), '原位仓不该写 meta.json'
        print('   ok   注册表/闸门/元数据回写（named 写、default 跳）')
    finally:
        tearDown()


def test_replay_buffer_dir_roundtrip():
    """replay 落点：CONFIG['buffer_dir'] 注入后主/验仓成对写入命名目录，读回一致；不设则原行为。"""
    setUp()
    try:
        from core.utils.replay import ReplayBuffer
        bd = os.path.join(CONFIG['data_dir'], 'ChineseChess', 'buffers', 'rt')
        os.makedirs(bd, exist_ok=True)
        CONFIG['buffer_dir'] = bd
        buf = ReplayBuffer(capacity=64, batch_size=8, game='ChineseChess')
        import numpy as np
        shp = CONFIG['ChineseChess']['tensor_shape']
        for _ in range(20):
            buf.add((np.ones(shp, dtype=np.float32), np.ones(buf.policy_buffer.shape[1],
                                                              dtype=np.float32) * 0.01, 0.5))
        buf.save('ChineseChess')
        assert os.path.exists(os.path.join(bd, CONFIG['buffer_name'])), '主仓应落命名目录'
        assert os.path.exists(os.path.join(bd, CONFIG['buffer_name'] + '.stats.json'))
        b2 = ReplayBuffer(capacity=128, batch_size=8, game='ChineseChess')
        b2.load('ChineseChess')
        assert b2.size == 20, b2.size
        # 不注入 = 原位（net_dir(game)）——路径构造回归
        CONFIG.pop('buffer_dir')
        from core.utils.config import net_dir as _nd
        assert ReplayBuffer(64, 8, 'ChineseChess')._path('ChineseChess') == \
            os.path.join(_nd('ChineseChess'), CONFIG['buffer_name'])
        print('   ok   replay 落点：命名目录往返一致，缺省回原位')
    finally:
        tearDown()


def test_auto_pool_numbering():
    """新建经验池默认递增编号：空名字 → pool1、pool2……已有 pool5 则下一个 pool6。"""
    setUp()
    try:
        from core.utils.buffers import create_named, next_pool_name
        ok1, _, m1 = create_named('ChineseChess', '')
        assert ok1 and m1['name'] == 'pool1', m1
        ok2, _, m2 = create_named('ChineseChess', '   ')
        assert ok2 and m2['name'] == 'pool2', m2
        create_named('ChineseChess', 'pool5')
        assert next_pool_name('ChineseChess') == 'pool6'
        ok3, msg3, _ = create_named('ChineseChess', 'pool1')
        assert not ok3 and '已存在' in msg3
        print('   ok   自动编号：pool1/pool2 顺序、max+1、撞名仍拒')
    finally:
        tearDown()


def test_routes_and_knobs():
    """HTTP 面 + 训练旋钮白名单校验。"""
    setUp()
    try:
        import services.admin.routes as routes
        from flask import Flask
        from services.admin.runner import _validate
        app = Flask(__name__)
        app.register_blueprint(routes.bp)
        c = app.test_client()
        lst = c.get('/api/admin/buffers?game=ChineseChess')
        assert lst.status_code == 200
        refs = {b['ref'] for b in lst.get_json()['buffers']}
        assert any(r.startswith('default@') for r in refs), refs
        cr = c.post('/api/admin/buffers/create', json={'name': 'api1', 'game': 'ChineseChess'})
        assert cr.status_code == 200 and cr.get_json()['ok']
        assert 'api1' in {b['ref'] for b in c.get('/api/admin/buffers?game=ChineseChess')
                          .get_json()['buffers']}
        bad = c.post('/api/admin/buffers/create', json={'name': 'bad name!'})
        assert bad.status_code == 400, '带空格/符号仍必须被拒'
        ab = c.post('/api/admin/buffers/create', json={'name': 'AB'})
        assert ab.status_code == 200 and ab.get_json()['buffer']['name'] == 'ab', '显式大写名宽容转小写'

        o, e = _validate({'buffer': 'api1', 'start_from': '34'})
        assert not e and o['buffer'] == 'api1' and o['start_from'] == '34', (o, e)
        o, e = _validate({'buffer': 'default@s', 'start_from': 'fresh'})
        assert not e, e
        o, e = _validate({'start_from': 'yolo'})
        assert e and '起点' in e or (e and 'start_from' in e), e
        o, e = _validate({'buffer': '../evil'})
        assert e, '非法仓名必须被表单层挡住'
        print('   ok   路由与旋钮：清单/建仓/坏名拒、env 映射、值校验')
    finally:
        tearDown()


def main() -> None:
    for fn in (test_resolve_start_from_matrix, test_registry_and_gates,
               test_replay_buffer_dir_roundtrip, test_auto_pool_numbering, test_routes_and_knobs):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 5 NAMED-BUFFER/START tests PASSED')


if __name__ == '__main__':
    main()
