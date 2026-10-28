"""流水线气泡复现/对照（不需要 GPU）：同一个"假引擎"下，对比两种 MCTS 驱动方式的
引擎利用率。

线上实测（GPU）：每批 window 5.17ms 等请求、fwd 2.90ms 真算 → 占空比 34%。
怀疑根因是 run(K) 的同步屏障："发 K 条 → 等齐 K 条 → 再发一批"，等齐期间队列被吃空。
本脚本用 CPU + 一个模拟前向延迟的批处理引擎，把同一条 MCTS 代码跑两遍：

  barrier     : mcts.run(120, batch_leaves=8)         老路径（等齐 K 条）
  pipelined   : mcts.run_pipelined(120, batch_leaves=8) 新路径（回一条补一条）

只看相对量：同样的 CPU、同样的"GPU"，谁把引擎喂得更满、谁跑完同样的模拟数更快。

运行：.venv/bin/python tests/drills/pipeline_starvation_replica.py
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
from services.train.selfplay import run_mcts  # noqa: E402

N_ACTIONS = ChineseChess.n_actions
POLICY = np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS


class FakeBatchEngine:
    """模拟 GPU 引擎的批处理线程：收批 → 固定 fwd 延迟 → 回填。

    与真引擎的关键行为一致：队列里有多少就抓多少（最多 max_batch），队列空就等，
    抓完一批就"算" fwd_ms，然后把结果按 fill() 回填（流水线靠 lane、同步靠 event）。
    """

    def __init__(self, fwd_ms: float = 2.0, max_batch: int = 256, window_ms: float = 0.5):
        self.request_queue: queue.Queue = queue.Queue()
        self.fwd_ms = fwd_ms
        self.max_batch = max_batch
        self.window = window_ms / 1e3
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.fwd_total = 0.0
        self.wall = 0.0
        self.batches = 0
        self.reqs = 0
        self.batch_sizes: list[int] = []
        self._lf = threading.Lock()

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                first = self.request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            t_start = time.perf_counter()
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.request_queue.get(timeout=self.window))
                except queue.Empty:
                    break
            t0 = time.perf_counter()
            time.sleep(self.fwd_ms / 1e3)
            for r in batch:
                event = r.fill(POLICY, 0.0)
                if event is not None:
                    event.set()
            dt = time.perf_counter() - t0
            self.wall += time.perf_counter() - t_start
            self.fwd_total += dt
            with self._lf:
                self.batches += 1
                self.reqs += len(batch)
                self.batch_sizes.append(len(batch))

    def shutdown(self):
        self.stop.set()
        self.thread.join(timeout=5)


def _worker(engine, n_moves, n_sims, batch_leaves, pipelined, out, idx):
    env = ChineseChess()
    mcts = dm.NeuronMCTS.make_queue_mcts(infer_q=engine.request_queue, env_class=ChineseChess,
                                        state=env.state, last_action=env.last_action,
                                        player_to_move=env.player_to_move,
                                        is_self_play=True, planes_version=2)
    rng = np.random.default_rng(idx)
    t0 = time.perf_counter()
    for _ in range(n_moves):
        if pipelined:
            mcts.run_pipelined(n_sims, batch_leaves=batch_leaves)
        else:
            mcts.run(n_sims, batch_leaves=batch_leaves)
        pi = mcts.get_pi(1.0)
        pi = pi / pi.sum()   # get_pi 是 float32，41 个动作的和可能偏差 1e-7，
                             # numpy.random.choice 的 atol 只有 1.5e-8 → 会偶发
                             # "Probabilities do not sum to 1"
        action = int(rng.choice(len(pi), p=pi))
        env.step(action)
        mcts.apply_action(action)
        if env.terminated:
            break
    out[idx] = time.perf_counter() - t0
    mcts.shutdown()


def run_case(name: str, workers: int, n_moves: int, n_sims: int, batch_leaves: int,
             pipelined: bool) -> dict:
    eng = FakeBatchEngine(fwd_ms=2.9, max_batch=256, window_ms=2.0)
    eng.start()
    out = [0.0] * workers
    ths = [threading.Thread(target=_worker,
                            args=(eng, n_moves, n_sims, batch_leaves, pipelined, out, i))
           for i in range(workers)]
    t0 = time.perf_counter()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    eng.shutdown()

    sims = workers * n_moves * n_sims
    duty = eng.fwd_total / max(eng.wall, 1e-9)
    res = {'case': name, 'wall': wall, 'sims': sims, 'sims_per_s': sims / wall,
           'req_per_s': eng.reqs / wall,
           'batch_mean': statistics.mean(eng.batch_sizes) if eng.batch_sizes else 0,
           'engine_duty': duty, 'batches': eng.batches}
    return res


def main() -> None:
    # 参数对齐线上量级：64 局 × K=8 = 512 在飞，fwd 2.9ms、max_batch 256、窗口 2ms
    workers, n_moves, n_sims, k = 64, 8, 160, 8
    rows = []
    for name, pipelined in (('barrier (run)', False), ('pipelined (run_pipelined)', True)):
        r = run_case(name, workers, n_moves, n_sims, k, pipelined)
        rows.append(r)
        print(f"[{name:26s}] {r['wall']:5.1f}s  {r['req_per_s']:7.0f} req/s  "
              f"均批 {r['batch_mean']:6.1f}  引擎占空比 {r['engine_duty'] * 100:5.1f}%  "
              f"模拟数 {r['sims']}")
    b, p = rows
    print(f"\n流水线 vs 同步屏障：吞吐 {p['req_per_s'] / b['req_per_s']:.2f}x，"
          f"占空比 {b['engine_duty'] * 100:.0f}% → {p['engine_duty'] * 100:.0f}%，"
          f"均批 {b['batch_mean']:.0f} → {p['batch_mean']:.0f}")


if __name__ == '__main__':
    main()
