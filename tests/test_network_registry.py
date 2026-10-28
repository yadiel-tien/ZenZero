"""架构注册表 + checkpoint envelope + profile 档位 的回归测试（S1/S2）。

核心验收（RFC 定稿原话）：
- "新增一种架构 = 往 archs/ 放一个文件 + 注册一行，engine/train_server 零改动"
  → 本测试在运行时注册一个哑元架构，走完 build → save envelope → engine.load_model
    → convert 布局 全链路，证明引擎侧没有任何架构硬编码。
- "老 checkpoint 零迁移" → 无 arch 字段的 v1 档按 config 形状兜底为 resnet_v1。
- profile：m=历史根路径（不搬家）；s/l 子目录隔离；preset 禁止覆写棋规/表示级键。

运行：.venv/bin/python tests/test_network_registry.py
"""
import os
import shutil
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from core.network import registry  # noqa: E402
from core.network.network import BaseNet  # noqa: E402  (触发存量注册)
from core.utils.config import CONFIG, net_dir, set_profile  # noqa: E402

TINY = {'arch': 'resnet_v1', 'in_channels': 4, 'n_filters': 8, 'n_res_blocks': 1,
        'n_cells': 20, 'n_actions': 30, 'use_se': False, 'n_policy_filters': 2,
        'n_value_filters': 2, 'n_value_hidden_channels': 4, 'feature_version': 9}
TINY_BIGAME_SHAPED = dict(TINY, in_channels=26)  # 形状不变，供 dummy 复用


def _tmp_data_dir():
    d = tempfile.mkdtemp(prefix='zz_registry_')
    CONFIG['data_dir'] = d
    return d


# ---------------------------------------------------------------- registry

def test_registry_has_stock_archs():
    assert 'resnet_v1' in registry.known()
    assert 'bigame_v1' in registry.known()


def test_build_and_forward():
    m = registry.build('resnet_v1', TINY, eval_model=True, device='cpu')
    assert m.arch_name == 'resnet_v1' and m.feature_version == 9
    m.eval()
    with torch.no_grad():
        p, v = m(torch.zeros(2, 4, 4, 5))
    assert tuple(p.shape) == (2, 30) and tuple(v.shape) == (2,)


def test_unknown_arch_raises():
    try:
        registry.build('no_such_arch_v9', TINY, device='cpu')
        raise AssertionError('unknown arch must raise')
    except ValueError as e:
        assert 'resnet_v1' in str(e)   # 报错要告知可用架构


def test_new_arch_zero_touch_e2e():
    """RFC 验收：模拟"往 archs/ 放一个新文件"——运行时注册哑元架构，
    经 envelope 保存 → BaseNet.load_from_checkpoint 分发 → 前向，全程 engine 无感知"""

    class DummyNet(nn.Module):
        arch_name = None

        def __init__(self, config, eval_model=True, device='cpu'):
            super().__init__()
            self.config = config
            self.feature_version = int(config.get('feature_version', 1))
            self.fc = nn.Linear(config['in_channels'], config['n_actions'])
            self.initialize = lambda: None

        def forward(self, x):
            f = x.mean(dim=(2, 3))
            return self.fc(f), torch.tanh(f.sum(dim=1, keepdim=True)).squeeze(1)

    registry.register('dummy_v1')(DummyNet)
    try:
        cfg = dict(TINY, feature_version=2)
        model = registry.build('dummy_v1', cfg, device='cpu')
        d = tempfile.mkdtemp(prefix='zz_dummy_')
        p = os.path.join(d, 'checkpoint_0.pt')
        torch.save({'model': model.state_dict(), 'config': cfg, 'arch': 'dummy_v1',
                    'feature_version': 2}, p)
        got, ok = BaseNet.load_from_checkpoint(p, eval_model=True)
        assert ok and isinstance(got, DummyNet) and got.arch_name == 'dummy_v1'
        assert got.feature_version == 2
        m2, ok2 = registry.load_model(p)
        assert ok2 and isinstance(m2, DummyNet)
    finally:
        registry._REGISTRY.pop('dummy_v1', None)
        shutil.rmtree(d, ignore_errors=True)


def test_legacy_envelope_fallback():
    """v1 老档（无 arch/feature_version 字段）：按 config 形状兜底 resnet_v1，fv=1"""
    m = registry.build('resnet_v1', {k: v for k, v in TINY.items() if k != 'feature_version' and k != 'arch'},
                       device='cpu')
    d = tempfile.mkdtemp(prefix='zz_legacy_')
    p = os.path.join(d, 'checkpoint_1.pt')
    torch.save({'model': m.state_dict(),
                'config': {k: v for k, v in TINY.items() if k not in ('feature_version', 'arch')}}, p)
    got, ok = registry.load_model(p)
    assert ok and got.arch_name == 'resnet_v1'
    assert got.feature_version == 1, 'v1 档缺 feature_version 必须推断为 1，不能冒充 2'
    shutil.rmtree(d, ignore_errors=True)


def test_missing_checkpoint_fresh():
    m, ok = registry.load_model(os.path.join(tempfile.gettempdir(), 'zz_no_such_ckpt.pt'))
    assert m is None and not ok   # registry 不知道游戏，缺档交调用方 fresh


# ---------------------------------------------------------------- profile

def test_profile_scale_merge_and_restore():
    from core.utils import config as C
    base = dict(C.CONFIG['ChineseChess']['default_net'])
    try:
        set_profile('s')
        dn = C.CONFIG['ChineseChess']['default_net']
        assert dn['n_filters'] == 128 and dn['n_res_blocks'] == 6
        assert dn['in_channels'] == base['in_channels'], 'preset 不得触碰表示级'
    finally:
        set_profile('m')
    assert C.CONFIG['ChineseChess']['default_net']['n_filters'] == base['n_filters']


def test_profile_object_identity_stable():
    """default_net 必须原地更新（同一 dict 对象）——Net 的默认参数按引用绑定，换对象会脱钩"""
    from core.utils import config as C
    before = C.CONFIG['ChineseChess']['default_net']
    set_profile('s')
    assert C.CONFIG['ChineseChess']['default_net'] is before
    set_profile('m')
    assert C.CONFIG['ChineseChess']['default_net'] is before


def test_profile_forbidden_override_guard():
    from core.utils import config as C
    C.CONFIG['Gomoku']['presets']['bad'] = {'in_channels': 777}
    try:
        set_profile('bad')
        raise AssertionError('forbidden preset key must raise')
    except ValueError:
        pass
    finally:
        C.CONFIG['Gomoku']['presets'].pop('bad')
        set_profile('m')


def test_profile_paths_isolated():
    """m 档=根路径零迁移；s 档=子目录；best_index/checkpoint 各自独立"""
    d = _tmp_data_dir()
    try:
        from core.network import functions as nf
        nf.save_best_index(111, 'ChineseChess', 'm')
        nf.save_best_index(222, 'ChineseChess', 's')
        assert os.path.exists(os.path.join(d, 'ChineseChess', 'best_index.pkl')), 'm 必须仍写根目录'
        assert os.path.exists(os.path.join(d, 'ChineseChess', 's', 'best_index.pkl'))
        assert nf.read_best_index('ChineseChess', 'm') == 111
        assert nf.read_best_index('ChineseChess', 's') == 222
        assert nf.list_all_indices('ChineseChess', 's') == []   # 空目录不炸
        # checkpoint 路径
        from services.inference.functions import get_checkpoint_path, get_model_name
        assert get_checkpoint_path('ChineseChess', 5, 'm').endswith('ChineseChess/checkpoint_5.pt')
        assert get_checkpoint_path('ChineseChess', 5, 'l').endswith('ChineseChess/l/checkpoint_5.pt')
        assert get_model_name('ChineseChess', 5, 'm') == 'ChineseChess_5'
        assert get_model_name('ChineseChess', 5, 'l') == 'ChineseChess_l_5'
    finally:
        CONFIG['data_dir'] = str(os.path.join(os.path.dirname(__file__), '..', 'data'))
        shutil.rmtree(d, ignore_errors=True)


def test_named_custom_profile():
    """自定义命名配置：保存→出现在档位全集→set_profile 生效→删除；含校验矩阵与内置保护。"""
    import json
    import os as _os
    from core.utils.config import (available_profiles, delete_custom_profile, net_config_for,
                                   profiles_file, save_custom_profile, set_profile, CONFIG as C)
    old_profile = C['game_name'] and __import__('core.utils.config', fromlist=['PROFILE']).PROFILE
    d = _tmp_data_dir()
    try:
        # —— 校验矩阵：坏名字/坏键/坏范围/空配置/撞内置，全部拒 ——
        assert not save_custom_profile('Bad Name', {'n_filters': 64})[0]
        assert not save_custom_profile('ok1', {'n_actions': 999})[0]          # 棋规层禁入
        assert not save_custom_profile('ok1', {'feature_version': 3})[0]      # 表示层禁入
        assert not save_custom_profile('ok1', {'n_filters': 99999})[0]        # 超上限
        assert not save_custom_profile('ok1', {})[0]                          # 空
        assert not save_custom_profile('s', {'n_filters': 32})[0]             # 撞内置
        assert not _os.path.exists(profiles_file()), '失败不应留下文件'

        # —— 正常路径 ——
        ok, msg = save_custom_profile('mini64', {'n_filters': 64, 'n_res_blocks': 3,
                                                 'n_value_hidden_channels': 32, 'use_se': False})
        assert ok, msg
        assert json.load(open(profiles_file()))['mini64']['n_filters'] == 64
        aps = available_profiles('ChineseChess')
        assert 'mini64' in aps and 's' in aps and 'm' in aps
        cfg = net_config_for('mini64', 'ChineseChess')
        assert cfg['n_filters'] == 64 and cfg['n_res_blocks'] == 3
        assert cfg['n_actions'] == 2086 and cfg['in_channels'] == 26          # 未覆写的层照旧
        assert set_profile('mini64') == 'mini64'
        assert __import__('core.utils.config', fromlist=['settings']).settings['default_net']['n_filters'] == 64

        ok, msg = delete_custom_profile('mini64')
        assert ok and 'mini64' not in available_profiles('ChineseChess')
        assert not delete_custom_profile('m')[0], '内置档位不可删'
    finally:
        set_profile(old_profile or 'm')
        C['data_dir'] = _os.path.join(_os.path.dirname(__file__), '..', 'data')


def main():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, f in fns:
        print(f'== {name}')
        f()
        print('   ok')
    print(f'\nALL {len(fns)} REGISTRY/PROFILE TESTS PASSED')


if __name__ == '__main__':
    main()
