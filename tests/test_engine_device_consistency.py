"""回归测试：引擎输入张量必须跟随本引擎模型参数所在设备，而非全局 CONFIG['device']。

复现的线上崩溃（engine.py F.conv2d）：
    RuntimeError: Expected all tensors to be on the same device,
    but got weight is on cuda:0, different from other tensors on cpu
成因：其他模型加载失败时 BaseNet._degrade_cpu 把全局 CONFIG['device'] 改写为
'cpu'，而本引擎权重仍驻留 GPU（或反之）；旧版 _inference_loop 按全局配置搬运
输入，于是输入与权重分裂在两个设备。

测试中模型与输入都固定在 CPU，用非法的 CONFIG['device']='cuda:0' 模拟全局配置
污染：修复前引擎线程会尝试 .to('cuda:0') 而崩溃/请求超时；修复后完全忽略全局值。
（CUDA_VISIBLE_DEVICES 置空，确保测试永不触碰被其他进程占用的 GPU。）
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 必须在 import torch 前
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.network.network import Net  # noqa: E402
from core.utils.config import CONFIG  # noqa: E402
from services.inference.engine import InferenceEngine, _ModelAccess  # noqa: E402
from services.inference.request import QueueRequest  # noqa: E402

CFG = {'in_channels': 2, 'n_filters': 8, 'n_res_blocks': 1, 'use_se': False,
       'n_policy_filters': 2, 'n_value_filters': 2, 'n_value_hidden_channels': 4,
       'n_cells': 9, 'n_actions': 16}


def make_engine() -> InferenceEngine:
    model = Net(CFG, eval_model=True, device='cpu')
    InferenceEngine.load_model = classmethod(
        lambda cls, i, e, eval_mode=True: (model, 0))
    eng = InferenceEngine(0, 'Gomoku')
    eng.start()
    return eng


def infer(eng: InferenceEngine, state: np.ndarray, timeout: float = 30.0):
    ev = threading.Event()
    req = QueueRequest(state, ev)
    eng.request_queue.put(req)
    assert ev.wait(timeout), '推理请求超时：引擎线程可能因设备不一致崩溃'
    return req.policy, req.value


def test_model_access():
    """_ModelAccess：读者并发；写者等读者排空，等待期间新读者排在写者后（写者优先）"""
    acc = _ModelAccess()

    def reader(events: list):
        with acc.shared():
            events.append(('in', threading.get_ident()))
            time.sleep(0.15)
            events.append(('out', threading.get_ident()))

    ev: list = []
    ths = [threading.Thread(target=reader, args=(ev,)) for _ in range(4)]
    t0 = time.perf_counter()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    span = time.perf_counter() - t0
    assert span < 0.45, f'4 个读者疑似串行执行（{span:.2f}s），shared 锁未并发'
    assert sum(1 for k, _ in ev if k == 'in') == 4

    order: list = []

    def slow_reader():
        with acc.shared():
            order.append('r1-in')
            time.sleep(0.2)
            order.append('r1-out')

    def writer():
        with acc.exclusive():
            order.append('w-in')
            time.sleep(0.05)
            order.append('w-out')

    def late_reader():
        time.sleep(0.05)
        with acc.shared():
            order.append('r2-in')

    ths = [threading.Thread(target=f) for f in (slow_reader, writer, late_reader)]
    ths[0].start()
    time.sleep(0.02)
    ths[1].start()
    time.sleep(0.02)
    ths[2].start()
    for th in ths:
        th.join()
    assert order == ['r1-in', 'r1-out', 'w-in', 'w-out', 'r2-in'], order
    print('PASS 4: _ModelAccess 读者并发 + 写者优先')


def test_concurrent_requests():
    """模拟 selfplay 多局并发：16 个 MCTS 线程同时经引擎请求推理，全部正确返回"""
    eng = make_engine()
    assert eng._infer_workers >= 1
    n_games, n_moves = 16, 4
    errors: list = []
    done = [0]
    lock = threading.Lock()

    def game():
        rng = np.random.default_rng(threading.get_ident() & 0xFFFF)
        try:
            for _ in range(n_moves):
                policy, value = infer(eng, rng.random((2, 3, 3)).astype(np.float32),
                                      timeout=60)
                assert policy.shape == (CFG['n_actions'],)
                assert np.isfinite(value)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            with lock:
                done[0] += 1

    t0 = time.perf_counter()
    ths = [threading.Thread(target=game) for _ in range(n_games)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=120)
    rps = n_games * n_moves / (time.perf_counter() - t0)
    assert not errors, errors
    assert done[0] == n_games
    print(f'PASS 5: {n_games} 并发局 × {n_moves} 请求全部返回，引擎吞吐 {rps:.0f} req/s')
    eng.shutdown()


def main() -> None:
    eng = make_engine()
    state = np.random.default_rng(0).random((2, 3, 3)).astype(np.float32)

    # 1) 全局配置污染：模型在 cpu，CONFIG 指向不存在的 cuda:0
    real_dev = CONFIG['device']
    CONFIG['device'] = 'cuda:0'
    try:
        policy, value = infer(eng, state)
        assert policy.shape == (CFG['n_actions'],), policy.shape
        assert np.isfinite(policy).all() and np.isclose(policy.sum(), 1.0)
        assert np.isfinite(value)
        print('PASS 1: 输入跟随模型参数设备，忽略被污染的全局 CONFIG')

        # 2) relocate 失败回滚后：模型仍在 cpu 且 backend 配置已还原，仍能出结果
        try:
            eng.relocate('cuda:0')
            raise AssertionError('relocate 到不存在的设备应当抛异常')
        except RuntimeError:
            pass
        assert eng._ac_dev == 'cpu' and eng._pin is False, \
            f'回滚后 backend 未还原: {_as(eng)}'
        policy, _ = infer(eng, state)
        assert policy.shape == (CFG['n_actions'],)
        print('PASS 2: relocate 失败回滚后引擎仍正常服务')
    finally:
        CONFIG['device'] = real_dev

    # 3) 正常路径：relocate 同设备为 no-op，不破坏推理
    eng.relocate('cpu')
    assert eng._ac_dev == 'cpu' and eng._pin is False
    policy, _ = infer(eng, state)
    assert policy.shape == (CFG['n_actions'],)
    print('PASS 3: 同设备 relocate 幂等，推理正常')

    eng.shutdown()

    test_model_access()
    test_concurrent_requests()
    print('ALL OK')


def _as(eng):
    return (eng._ac_dev, eng._pin)


if __name__ == '__main__':
    main()
