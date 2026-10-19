"""引擎级端到端 bench：真实 Gomoku 网络，MCTS 式阻塞请求。
对比 worker 数对 多局并发吞吐 和 单局延迟 的影响。"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading
import time  # noqa: E402

import numpy as np  # noqa: E402

from core.network.network import Net  # noqa: E402
from core.utils.config import CONFIG  # noqa: E402
from services.inference.engine import InferenceEngine  # noqa: E402
from services.inference.request import QueueRequest  # noqa: E402

WORKERS = int(os.environ.get('ZZ_INFER_WORKERS', '8'))

torch_threads_note = ''
cfg = CONFIG['Gomoku']['default_net']
model = Net(cfg, eval_model=True, device='cpu')
InferenceEngine.load_model = classmethod(lambda cls, i, e, eval_mode=True: (model, 0))
eng = InferenceEngine(0, 'Gomoku')
eng.start()
state = np.random.default_rng(0).random((2, 15, 15), dtype=np.float32)


def one():
    ev = threading.Event()
    r = QueueRequest(state, event=ev)
    eng.request_queue.put(r)
    assert ev.wait(60), 'request timed out'


def measure(n_games: int, seconds: float) -> float:
    stop = time.perf_counter() + seconds
    cnt = [0]
    lock = threading.Lock()

    def game():
        local = 0
        while time.perf_counter() < stop:
            one()
            local += 1
        with lock:
            cnt[0] += local

    ths = [threading.Thread(target=game) for _ in range(n_games)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return cnt[0] / seconds


print(f'workers={WORKERS} threads_single={eng._threads_single} '
      f'threads_multi={eng._threads_multi} split={eng._split_batches}')
g1 = measure(1, 5)
print(f'单局 (人机对战 sim/s): {g1:6.1f}')
g4 = measure(4, 5)
print(f'4 局并发 sim/s      : {g4:6.1f}')
g8 = measure(8, 5)
print(f'8 局并发 sim/s      : {g8:6.1f}')
g16 = measure(16, 5)
print(f'16 局并发 sim/s     : {g16:6.1f}')
eng.shutdown()
