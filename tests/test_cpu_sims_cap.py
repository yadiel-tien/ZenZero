"""CPU 降级训练容错回归（2026-11-07 复盘）。

背景：引擎显存不足会自动降 CPU（合法路径，服务纯 CPU 机/小显存机），但模拟数
不跟着降——784 sims/手在 CPU 上是每手 30~60 秒，推理请求堵满队列 → 300s 超时
雪崩 → 整局炸穿（今晨 m8 run 死于此）。修复=cap_sims 按引擎实际设备封顶。

这里验证的是判定函数本身（不碰 GPU/真引擎）：CPU 封顶、GPU 原样、缺属性不误伤。
运行：.venv/bin/python tests/test_cpu_sims_cap.py
"""
import os
import sys

os.environ.setdefault('ZZ_DEVICE', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils.config import CONFIG, settings  # noqa: E402


class _Eng:
    def __init__(self, dev):
        self._ac_dev = dev


def test_cpu_caps_to_config():
    from services.train.selfplay import cap_sims
    cap = int(settings.get('cpu_sims_cap'))
    assert cap > 0, 'cpu_sims_cap 必须存在于配置'
    assert cap_sims(784, _Eng('cpu')) == cap
    assert cap_sims(784, _Eng('cuda:0')) == 784          # GPU 原样不封顶
    assert cap_sims(50, _Eng('cpu')) == 50                # 本就低于帽不动
    assert cap_sims(300, _Eng('cpu')) == cap              # 守擂同样封顶
    assert cap_sims(784, object()) == 784                 # 无设备属性 → 宁不误伤
    import torch
    assert cap_sims(784, _Eng(torch.device('cpu'))) == cap  # torch.device 对象形态
    print('   ok  cap_sims：CPU 封顶 / GPU 原样 / 边界不误伤（帽=%d，两棋种各自可配）' % cap)


def test_both_games_have_cap_key():
    for game in ('ChineseChess', 'Gomoku'):
        assert 'cpu_sims_cap' in CONFIG[game], f'{game} 缺 cpu_sims_cap'
    print('   ok  两棋种配置段都有 cpu_sims_cap')


if __name__ == '__main__':
    test_cpu_caps_to_config()
    test_both_games_have_cap_key()
    print('\nALL CPU-SIMS-CAP TESTS PASSED')
