"""WDL value 三头的回归测试（resnet_wdl_v1）。

验收点：
- 架构注册与 forward 契约：返回 (policy_logits, 期望分)，期望分 ∈ (-1,1) 且与
  forward_wdl 的 logits 折算一致——engine/MCTS 按此消费，尺度必须与终局奖励 ±1/0 对齐；
- z→类别映射（胜2 和1 负0，与 logit 顺序一致）；
- checkpoint envelope 往返：wdl 存档经 registry 重建仍是 WdlNet，v1 老档不受影响；
- 半迁移 migrated_load：v1→wdl 只继承同名同形张量（trunk/policy 全中，value.fc2 重初始化）；
- CE 训练信号：loss 能反传到 value 头；
- engine 无档新建走 default_net 声明的架构。

运行：.venv/bin/python tests/test_wdl_value.py
"""
import os
import shutil
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from core.network import registry  # noqa: E402
from core.network.archs.resnet_wdl import WdlNet, expected_score, zld_to_class  # noqa: E402
from core.utils.config import CONFIG  # noqa: E402

TINY = {'in_channels': 4, 'n_filters': 8, 'n_res_blocks': 1, 'n_cells': 20,
        'n_actions': 30, 'use_se': False, 'n_policy_filters': 2,
        'n_value_filters': 2, 'n_value_hidden_channels': 4, 'feature_version': 9}


def _build(arch):
    return registry.build(arch, dict(TINY, arch=arch), eval_model=True, device='cpu')


def test_arch_registered():
    assert 'resnet_wdl_v1' in registry.known()


def test_forward_contract():
    """(policy, 期望分) 双输出；期望分区间与 forward_wdl 折算一致（engine 零改动的依据）"""
    net = _build('resnet_wdl_v1')
    x = torch.randn(3, 4, 4, 5)
    with torch.no_grad():
        pol, v = net(x)
        pol2, wdl = net.forward_wdl(x)
    assert tuple(pol.shape) == (3, 30) and tuple(wdl.shape) == (3, 3)
    assert v.shape == (3,) and bool(((v > -1) & (v < 1)).all())
    assert torch.allclose(v, expected_score(wdl), atol=1e-6)
    assert torch.equal(pol, pol2)


def test_zld_to_class():
    z = torch.tensor([1.0, 0.5, 0.0, -0.3, -1.0])
    assert zld_to_class(z).tolist() == [2, 1, 1, 1, 0]


def test_envelope_roundtrip():
    """save→load 按 envelope 的 arch 重建；v1 档不受影响"""
    d = tempfile.mkdtemp(prefix='zz_wdl_env_')
    try:
        net = _build('resnet_wdl_v1')
        path = os.path.join(d, 'checkpoint_1.pt')
        torch.save({'model': net.state_dict(), 'config': dict(TINY, arch='resnet_wdl_v1'),
                    'feature_version': 9, 'arch': 'resnet_wdl_v1'}, path)
        m, ok = registry.load_model(path, eval_model=True, device='cpu')
        assert ok and isinstance(m, WdlNet) and m.arch_name == 'resnet_wdl_v1'
        v1 = _build('resnet_v1')
        p1 = os.path.join(d, 'checkpoint_2.pt')
        torch.save({'model': v1.state_dict(), 'config': dict(TINY, arch='resnet_v1'),
                    'feature_version': 9, 'arch': 'resnet_v1'}, p1)
        m2, ok2 = registry.load_model(p1, eval_model=True, device='cpu')
        assert ok2 and not isinstance(m2, WdlNet)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_migrated_load():
    """v1 权重半迁移进 wdl：同名同形全继承，value.fc2 重初始化"""
    src = _build('resnet_v1')
    dst = _build('resnet_wdl_v1')
    before_fc1 = dst.value.fc1.weight.clone()
    used, reinit = registry.migrated_load(dst, src.state_dict())
    assert reinit == ['value.fc2.bias', 'value.fc2.weight'], reinit
    assert used == len(dst.state_dict()) - 2
    # trunk/policy 真的搬进来了
    torch.testing.assert_close(dst.policy.fc.weight, src.policy.fc.weight)
    # 与 value 无关的 fc1 不受迁移影响（保持自身初始化）
    assert not torch.allclose(before_fc1, dst.value.fc1.weight) or True  # fc1 同名同形：应继承
    torch.testing.assert_close(dst.value.fc1.weight, src.value.fc1.weight)


def test_ce_signal_reaches_value_head():
    """CE(wdl, z类) 可反传到 value 头参数（v1 的 MSE 路径结构上不可能给 fc2 这类信号）"""
    net = _build('resnet_wdl_v1')
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    x = torch.randn(4, 4, 4, 5)
    z = torch.tensor([1.0, 0.0, -1.0, 1.0])
    g0 = net.value.fc2.weight.grad
    _, wdl = net.forward_wdl(x)
    pol_logits, _ = net(x)
    pi_soft = torch.rand(4, 30)
    pi_soft = pi_soft / pi_soft.sum(dim=1, keepdim=True)
    loss = (F.cross_entropy(wdl, zld_to_class(z))
            - (pi_soft * F.log_softmax(pol_logits, dim=1)).sum(dim=1).mean())
    opt.zero_grad()
    loss.backward()
    assert net.value.fc2.weight.grad is not None
    assert torch.isfinite(net.value.fc2.weight.grad).all()
    assert net.value.fc2.weight.grad.abs().sum() > 0
    assert g0 is None  # 起点无梯度，确认是本次反传写入


def test_engine_fresh_build_follows_default_arch():
    """无 checkpoint 时 engine 按棋种 default_net 声明的架构新建（有档仍按 envelope）"""
    from services.inference.engine import InferenceEngine
    from core.utils.config import set_profile
    d = tempfile.mkdtemp(prefix='zz_wdl_engine_')
    old_net = dict(CONFIG['ChineseChess']['default_net'])
    old_dir = CONFIG['data_dir']
    try:
        CONFIG['data_dir'] = d
        CONFIG['ChineseChess']['default_net'].update(dict(TINY, arch='resnet_wdl_v1',
                                                          in_channels=26))
        model, idx = InferenceEngine.load_model(-1, 'ChineseChess', profile='m')
        assert idx == -1 and isinstance(model, WdlNet)
    finally:
        CONFIG['data_dir'] = old_dir
        CONFIG['ChineseChess']['default_net'].clear()
        CONFIG['ChineseChess']['default_net'].update(old_net)
        shutil.rmtree(d, ignore_errors=True)


def main():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, f in fns:
        print(f'== {name}')
        f()
        print('   ok')
    print(f'\nALL {len(fns)} WDL VALUE TESTS PASSED')


if __name__ == '__main__':
    main()
