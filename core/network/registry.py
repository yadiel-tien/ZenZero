"""网络架构注册表 + checkpoint envelope 读取（S1）。

设计（对应 RFC 定稿）：
- 架构是"注册出来的"，不是"改出来的"：新增一种网络 = 在 core/network/archs/ 放一个
  文件、@register('xxx_v1') 标注、被 archs/__init__ 自动导入——engine / train_server /
  hub 全部零改动。
- checkpoint 自描述（envelope）：'arch'（架构名）+ 'feature_version'（棋盘表示版本）
  写进 ckpt 顶层；缺字段的 v1 老档按 config 形状兜底推断（in_channels/n_res_blocks →
  resnet_v1），因此 data/ 下所有历史 checkpoint 不需要迁移。
- 架构、表示版本、棋种（env）是三个正交维度：本模块只管第一个，feature_version
  沿用 model.feature_version 约定（env.convert_to_network 按它选布局）。

已知约定（契约，勿破坏）：
    cls(config: dict, eval_model: bool, device: str|torch.device) -> module
    module.feature_version: int      # BaseNet.__init__ 从 config 读取
    module.arch_name: str            # register 装饰器写入
"""
from __future__ import annotations

import os

import torch.nn as nn

# 架构名 -> 类。register() 是唯一写入口。
_REGISTRY: dict[str, type[nn.Module]] = {}

# v1 时代没有 arch 字段，全部 checkpoint 都是这个类
DEFAULT_ARCH = 'resnet_v1'


def register(name: str):
    """把网络类注册进表。可作装饰器：@register('resnet_v1')"""

    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f'arch name collision: {name}')
        _REGISTRY[name] = cls
        cls.arch_name = name
        return cls

    return deco


def known() -> dict[str, str]:
    """已注册架构 -> 类名，供日志/报错/前端展示"""
    return {k: v.__name__ for k, v in _REGISTRY.items()}


def get(arch: str) -> type[nn.Module]:
    cls = _REGISTRY.get(arch)
    if cls is None:
        raise ValueError(f"Unknown arch '{arch}'. Registered: {sorted(_REGISTRY)}")
    return cls


def infer_arch(config) -> str:
    """老 checkpoint（无 arch 字段）按 config 形状兜底识别架构"""
    if isinstance(config, dict):
        if 'game1' in config and 'game2' in config:
            return 'bigame_v1'
        if 'n_res_blocks' in config:
            return DEFAULT_ARCH
    return DEFAULT_ARCH


def build(arch: str, config: dict, eval_model: bool = True, device=None) -> nn.Module:
    """按注册表实例化。device=None 时不传，用类自身的默认设备"""
    cls = get(arch)
    kwargs = {} if device is None else {'device': device}
    return cls(config, eval_model=eval_model, **kwargs)


def migrated_load(model: nn.Module, state_dict: dict) -> tuple[int, list[str]]:
    """跨架构半迁移加载：只搬**同名且同形**的参数张量，其余保持新初始化
    （典型场景：resnet_v1 → resnet_wdl_v1，trunk/policy 继承，value.fc2 重初始化）。
    :return (继承的张量数, 重初始化的参数名)"""
    tgt = model.state_dict()
    usable = {k: v for k, v in state_dict.items() if k in tgt and v.shape == tgt[k].shape}
    model.load_state_dict(usable, strict=False)
    return len(usable), sorted(set(tgt) - set(usable))


def load_model(path: str, eval_model: bool = True, device=None):
    """读 envelope 并重建模型。

    :return: (model, success)。文件缺失返回 (None, False)，由调用方决定
             fresh-build 的 arch/config（registry 不知道游戏，给不了默认配置）。
             文件存在但损坏照旧抛异常。
    """
    if not os.path.exists(path):
        return None, False
    import torch
    checkpoint = torch.load(path, map_location='cpu')  # 先落 CPU 再迁移，支持跨设备加载
    config = checkpoint['config']
    arch = checkpoint.get('arch') or infer_arch(config)
    model = build(arch, config, eval_model=eval_model, device=device)
    # feature_version：顶层优先于 config 内同名字段（版本 2 起双写）
    if checkpoint.get('feature_version') is not None:
        model.feature_version = int(checkpoint['feature_version'])
    model.load_state_dict(checkpoint['model'])
    return model, True
