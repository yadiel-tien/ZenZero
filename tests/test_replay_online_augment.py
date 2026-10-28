"""在线增强 ReplayBuffer + 验证集 的回归测试。

核心命题：
- add 只存一份原始样本（capacity = 不同局面数，不再被 ×N 复制稀释）；
- get_batch(augment=True) 每条样本随机施加一个对称群成员，
  结果必在 {transform(raw, g)} 的轨道里且 state/policy 用同一个 g（配对正确）；
- get_batch(augment=False) 与仓内原始数据逐位一致（验证集口径）；
- 群成员分布在大数下接近均匀；
- 训练/验证双池文件互不串门；
- fit 端到端：新配平公式跑通 + Val/* 不炸。
"""
import json
import os
import shutil
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'
# 这套测试全程五子棋（2x15x15）；仓库默认活动棋种是象棋，显式钉住，
# 否则 ReplayBuffer/TrainServer 会按 26 面象棋形状建仓、整组红
os.environ['ZZ_GAME'] = 'Gomoku'

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from core.utils.config import CONFIG  # noqa: E402
from core.utils.mirror import apply_symmetry, mirror_board_policy, mirror_action_policy  # noqa: E402
from core.utils.replay import ReplayBuffer  # noqa: E402
from core.env.gomoku import Gomoku  # noqa: E402
from core.env.chess import ChineseChess  # noqa: E402

_tmp = None


def setUp():
    global _tmp
    _tmp = tempfile.mkdtemp(prefix='zz_aug_')
    os.makedirs(os.path.join(_tmp, 'Gomoku'), exist_ok=True)
    os.makedirs(os.path.join(_tmp, 'ChineseChess'), exist_ok=True)
    CONFIG['data_dir'] = _tmp
    CONFIG['runs_dir'] = os.path.join(_tmp, 'runs')   # 隔离：绝不共享仓库 runs/（并发进程互删会炸）
    CONFIG['log_dir'] = os.path.join(_tmp, 'logs')    # 同理：sidecar/日志绝不能写进生产 logs/
    os.makedirs(CONFIG['log_dir'], exist_ok=True)


def tearDown():
    CONFIG['data_dir'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)


def _gomoku_orbit(state, pi):
    return [(apply_symmetry(state.copy(), g), mirror_board_policy(pi.copy(), g, state.shape))
            for g in Gomoku.symmetry_indices()]


def _orbit_match(sample, orbit):
    st, pi = sample
    return any(np.allclose(st, s) and np.allclose(pi, p) for s, p in orbit)


def test_add_stores_single_raw():
    setUp()
    try:
        rng = np.random.default_rng(1)
        buf = ReplayBuffer(100, 8, game='Gomoku')
        raw = (rng.random((2, 15, 15)).astype('float32'),
               (rng.random(225) / 225).astype('float32'), 0.4)
        buf.add(raw)
        assert buf.size == 1, f"写入增强时代会存 8 份，现在是 {buf.size}"
        # 未增强读取 = 逐位等于原始
        buf2 = ReplayBuffer(100, 1, game='Gomoku')
        buf2.append(*raw)
        s, p, v = buf2.get_batch(augment=False)
        assert np.allclose(s[0], raw[0]) and np.allclose(p[0], raw[1]) and v[0] == 0.4
    finally:
        tearDown()


def test_online_augment_orbit_and_pairing():
    """batch 里每条样本必须落在"某条原始样本的对称轨道"上，且 state/policy 同组配对"""
    setUp()
    try:
        rng = np.random.default_rng(2)
        buf = ReplayBuffer(64, 32, game='Gomoku')
        raws = [(rng.random((2, 15, 15)).astype('float32'),
                 (rng.dirichlet(np.ones(225))).astype('float32'), 0.1)
                for _ in range(40)]
        orbits = [_gomoku_orbit(s, p) for s, p, _ in raws]
        for r in raws:
            buf.add(r)
        seen_groups = set()
        for _ in range(6):
            st, pi, _v = buf.get_batch()
            for b in range(st.shape[0]):
                matched = False
                for ob in orbits:
                    for g_idx, (gs, gp) in enumerate(ob):
                        if np.allclose(st[b], gs) and np.allclose(pi[b], gp):
                            seen_groups.add(g_idx)
                            matched = True
                            break
                    if matched:
                        break
                assert matched, '样本脱离轨道 → transform 配对错误'
        # 192 次抽样对 8 个群成员的覆盖（缺席概率 ~3e-11/群，稳定不闪断）
        assert len(seen_groups) == len(Gomoku.symmetry_indices()), f'群覆盖不全: {seen_groups}'
    finally:
        tearDown()


def test_chess_online_augment_matches_reference():
    setUp()
    try:
        ChineseChess.init_class_dicts()
        rng = np.random.default_rng(3)
        buf = ReplayBuffer(64, 32, game='ChineseChess')
        lr, ud = ChineseChess.mirror_lr_actions, ChineseChess.mirror_ud_actions
        raws, orbits = [], []
        for _ in range(40):
            raw = (rng.random((26, 10, 9)).astype('float32'),
                   (rng.random(2086) / 2086).astype('float32'), -0.2)
            raws.append(raw)
            fid = apply_symmetry(raw[0], 4)
            fpi = mirror_action_policy(raw[1].copy(), 4, lr, ud)
            orbits.append([(raw[0], raw[1]), (fid, fpi)])   # {恒等, 左右镜像}
            buf.add(raw)
        assert buf.size == 40
        seen_id = seen_flip = False
        for _ in range(4):
            st, pi, _v = buf.get_batch()
            for b in range(st.shape[0]):
                ok = False
                for gi, orb in enumerate(orbits):
                    for g_idx, (gs, gp) in enumerate(orb):
                        if np.allclose(st[b], gs) and np.allclose(pi[b], gp):
                            ok = True
                            if g_idx == 0:
                                seen_id = True
                            else:
                                seen_flip = True
                            break
                    if ok:
                        break
                assert ok, '象棋在线增强必须恰为某原始样本的 {恒等, 左右镜像} 且配对'
        assert seen_id and seen_flip, '两个群成员都应出现'
    finally:
        tearDown()


def test_val_file_separation():
    setUp()
    try:
        rng = np.random.default_rng(4)
        tr = ReplayBuffer(64, 8, game='Gomoku')
        va = ReplayBuffer(64, 8, game='Gomoku', file_name=CONFIG['val_buffer_name'])
        for _ in range(40):
            tr.add((rng.random((2, 15, 15)).astype('float32'),
                    (rng.random(225) / 225).astype('float32'), 0.0))
            va.add((rng.random((2, 15, 15)).astype('float32'),
                    (rng.random(225) / 225).astype('float32'), 0.0))
        tr.save(); va.save()
        d = os.path.join(_tmp, 'Gomoku')
        # 训练池/验证池各带一份 .stats.json 侧车（监控面板据此显示"已用/总容量"，
        # 免得为两个数字去反序列化几 GB 的 pickle）
        assert {'buffer.pkl', 'buffer.pkl.stats.json',
                'val_buffer.pkl', 'val_buffer.pkl.stats.json'} == set(os.listdir(d)), os.listdir(d)
        stats = json.loads(open(os.path.join(d, 'buffer.pkl.stats.json'), encoding='utf-8').read())
        assert stats['size'] == 40 and stats['capacity'] == 64, stats
        assert stats['game'] == 'Gomoku' and 'saved_at' in stats, stats
        tr2 = ReplayBuffer(64, 8, game='Gomoku'); tr2.load(); assert tr2.size == 40
        va2 = ReplayBuffer(64, 8, game='Gomoku', file_name=CONFIG['val_buffer_name']); va2.load()
        assert va2.size == 40
    finally:
        tearDown()


def test_profile_subdir_autocreate_and_checkpoint():
    """档位子目录首次写入必须自建：s 档 buffer/checkpoint 落在 data/<Game>/s/，
    目录不存在也要能落盘（实测 s 档首跑 FileNotFoundError 的回归测试）"""
    import logging
    setUp()
    logging.disable(logging.CRITICAL)
    from core.utils.config import set_profile
    from services.inference.train_server import TrainServer
    srv = None
    set_profile('s')
    try:
        rng = np.random.default_rng(9)
        buf = ReplayBuffer(2048, 8, game='Gomoku')
        for _ in range(32):
            buf.add((rng.random((2, 15, 15)).astype('float32'),
                     np.ones(225, dtype='float32') / 225, 0.0))
        sub = os.path.join(_tmp, 'Gomoku', 's')
        assert not os.path.exists(sub), '前置：s 子目录必须不存在'
        buf.save()   # 旧代码在此 FileNotFoundError
        assert os.path.exists(os.path.join(sub, 'buffer.pkl'))
        srv = TrainServer(-1, 'Gomoku', buffer=buf)
        srv.save_checkpoint(99995)   # checkpoint 写入同样要走子目录自建
        assert os.path.exists(os.path.join(sub, 'checkpoint_99995.pt'))
        # 读回验证：s 档 best_index 与 buffer 各自独立
        from core.network import functions as nf
        nf.save_best_index(99995, 'Gomoku', 's')
        assert nf.read_best_index('Gomoku', 's') == 99995
        assert not os.path.exists(os.path.join(_tmp, 'Gomoku', 'best_index.pkl')), 's 档不得写进 m 档根目录'
    finally:
        set_profile('m')
        if srv is not None:
            srv.shutdown()
        logging.disable(logging.NOTSET)
        tearDown()


def test_fit_with_val_end_to_end():
    """五子小仓端到端：新配平公式步数 = passes×size/batch；Val/* 评估回路走通"""
    import time
    import logging
    logging.disable(logging.CRITICAL)
    from services.inference.train_server import TrainServer
    setUp()
    backup = CONFIG['training_passes_per_buffer']
    ck = os.path.join(_tmp, 'Gomoku', 'checkpoint_99996.pt')
    try:
        CONFIG['training_passes_per_buffer'] = 1
        srv = TrainServer(-1, 'Gomoku', buffer=ReplayBuffer(2100, 2048, game='Gomoku'))
        val = ReplayBuffer(2100, 2048, game='Gomoku', file_name=CONFIG['val_buffer_name'])
        rng = np.random.default_rng(5)
        for i in range(2048):
            pi = np.zeros(225, dtype='float32'); pi[i % 225] = 1.0
            srv.buffer.append(rng.random((2, 15, 15)).astype('float32') / 9, pi,
                              float(rng.uniform(-1, 1)))
            pi2 = np.zeros(225, dtype='float32'); pi2[(i * 7) % 225] = 1.0
            val.append(rng.random((2, 15, 15)).astype('float32') / 9, pi2,
                       float(rng.uniform(-1, 1)))
        expect_steps = max(1, int(2048 * 1 // 2048))
        t0 = time.time()
        done = srv.fit(n_collected_samples=2048, iteration=99996,
                       reload_buffer=False, val_buffer=val)
        assert done == 'Fit done!' and os.path.exists(ck)
        # 验证集必须同时记 CE 与"可学部分"（KL / 目标熵）：CE 有硬地板，单看它会误判"学不动"
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import glob as _glob
        ev = _glob.glob(os.path.join(CONFIG['runs_dir'], '*', 'events.out.tfevents.*'))
        assert ev, '没写出 TensorBoard 事件'
        acc = EventAccumulator(os.path.dirname(max(ev, key=os.path.getmtime)),
                               size_guidance={'scalars': 0})
        acc.Reload()
        tags = set(acc.Tags()['scalars'])
        for tag in ('Val/policy', 'Val/policy_kl', 'Val/policy_entropy'):
            assert tag in tags, f'缺 {tag}: {sorted(tags)}'
        # sidecar：面板读它，不再从几千行日志里刨 Val 行（会被挤出 tail 窗口）
        side_path = os.path.join(CONFIG['log_dir'], 'fit', 'val_latest.json')
        assert os.path.exists(side_path), f'没写留出集 sidecar: {side_path}'
        side = json.load(open(side_path, encoding='utf-8'))
        for key in ('policy', 'value', 'kl', 'entropy', 'n_samples', 'step'):
            assert side.get(key) is not None, f'sidecar 缺 {key}: {side}'
        assert side['kl'] >= 0 and side['entropy'] >= 0, side
        assert side['n_samples'] == val.size, (side['n_samples'], val.size)
        # 确定性：同一份留出集 + 同一个模型，两次评估必须给出完全相同的数
        # （旧实现每轮随机抽 4 批，Val 曲线自带 ±0.01 采样噪声，看不出真实变化）
        r1 = (side['policy'], side['value'], side['kl'], side['entropy'])
        srv.evaluate_val(val, iteration=99996)
        r2 = json.load(open(side_path, encoding='utf-8'))
        r2t = (r2['policy'], r2['value'], r2['kl'], r2['entropy'])
        assert r2t == r1, ('留出集评估不确定', r1, r2t)
        assert srv.total_steps_trained == expect_steps, (srv.total_steps_trained, expect_steps)
        print(f'   fit+val ok ({time.time() - t0:.1f}s, steps={expect_steps})')
    finally:
        CONFIG['training_passes_per_buffer'] = backup
        if os.path.exists(ck):
            os.remove(ck)
        srv.shutdown() if 'srv' in dir() else None
        tearDown()


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, f in tests:
        print(f'== {name}')
        f()
        print('   ok')
    print(f'\nALL {len(tests)} ONLINE-AUGMENT TESTS PASSED')


if __name__ == '__main__':
    main()
