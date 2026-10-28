"""架构包：新增网络架构 = 在本目录放一个文件，import registry 并 @register('your_arch_v1')。

本 __init__ 自动导入所有同级模块（各模块自行完成注册），engine/train_server 零改动。
注意：本包由 core/network/network.py 末尾导入——网络模块此时已初始化完毕，
     所以这里的模块可以放心 `from ..network import BaseNet, ResBlock...` 复用组件；
     反过来 network.py 绝不能 import 本包之外的 archs 内容（循环）。
"""
import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    if not _m.name.startswith('_'):
        importlib.import_module(f'{__name__}.{_m.name}')
