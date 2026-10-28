"""env 侧 CPU 成本的上限实验：convert_to_network 值不值得 C 化？

做法：同一份 MCTS 代码 + 同一个"零延迟批处理引擎"（隔离掉 GPU），在**同样的并发压力**下
（默认 32 线程 × K=12）跑三档，比较聚合 sims/s：

  real    : 仓库里的真 convert_to_network（35 个 numpy 调用）
  zero    : 返回常量数组（成本归零）→ 这是任何优化都够不到的**上限**
  copy    : 返回常量数组的 .copy()（仍要分配 26×10×9 float32 并写满 2340 个数）
            → 这是 C 版**不可省**的代价，更接近真实落点

用法：.venv/bin/python tests/drills/env_cost_ceiling.py [线程数] [K] [sims] [moves]
注意：classmethod 从类上取到的已是绑定方法，classmethod(that) 会双绑 cls —— 必须走 __dict__。
"""
import os
import queue
import statistics
import sys
import threading
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np  # noqa: E402

import core.mcts.deepMcts as dm  # noqa: E402
from core.env.chess import ChineseChess  # noqa: E402

POL = np.ones(ChineseChess.n_actions, dtype=np.float32) / ChineseChess.n_actions
REAL_FUNC = ChineseChess.__dict__['convert_to_network'].__func__
CACHED = np.zeros((26, 10, 9), dtype=np.float32)


def _stub(cls, state, current_player, version=None):
    return CACHED


def _stub_copy(cls, state, current_player, version=None):
    return CACHED.copy()


class InstantEngine:
    """零延迟批处理引擎：只测 MCTS 自身的 CPU 成本，不让"GPU"成为变量"""

    def __init__(self):
        self.request_queue: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                first = self.request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            while True:
                try:
                    batch.append(self.request_queue.get_nowait())
                except queue.Empty:
                    break
            for r in batch:
                event = r.fill(POL, 0.0)
                if event is not None:
                    event.set()


def run(n_threads: int, k: int, sims: int, moves: int) -> float:
    eng = InstantEngine()
    out = [0.0] * n_threads
    errors: list = []

    def worker(i: int):
        try:
            env = ChineseChess()
            mcts = dm.NeuronMCTS.make_queue_mcts(infer_q=eng.request_queue, env_class=ChineseChess,
                                                state=env.state, last_action=-1, player_to_move=0,
                                                is_self_play=True, planes_version=2)
            t0 = time.perf_counter()
            for _ in range(moves):
                mcts.run_pipelined(sims, batch_leaves=k)
                mcts.apply_action(int(np.argmax(mcts.root.child_n)))
            out[i] = time.perf_counter() - t0
            mcts.shutdown()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    eng.stop.set()
    assert not errors, errors[:3]
    return n_threads * moves * sims / wall


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    moves = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    arms = {'real': REAL_FUNC, 'zero': _stub, 'copy': _stub_copy}
    got = {name: [] for name in arms}
    try:
        for _ in range(4):
            for name, fn in arms.items():
                ChineseChess.convert_to_network = classmethod(fn)
                got[name].append(run(n, k, sims, moves))
    finally:
        ChineseChess.convert_to_network = classmethod(REAL_FUNC)

    base = statistics.median(got['real'])
    print(f'{n} 线程 × K={k}，每次 {sims} 模拟 × {moves} 手')
    for name, label in (('real', '真 convert       '), ('zero', '零成本桩（上限） '),
                        ('copy', '数组+写满（下界）')):
        m = statistics.median(got[name])
        print(f'  {label}: {m:9.0f} sims/s   ({(m / base - 1) * 100:+5.0f}%)   '
              f'样本 {[round(x) for x in got[name]]}')


if __name__ == '__main__':
    main()
