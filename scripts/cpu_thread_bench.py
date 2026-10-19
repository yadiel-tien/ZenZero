"""Gomoku 真实网络 CPU 扩展性基准：torch 线程数 × batch × bf16，外加并发 worker 吞吐。
用 CUDA_VISIBLE_DEVICES='' 保证绝不触碰 GPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.network.network import Net
from core.utils.config import CONFIG

torch.manual_seed(0)
cfg = CONFIG['Gomoku']['default_net']
net = Net(cfg, eval_model=True, device='cpu').eval()
print(f'net params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M, logical cpus: {os.cpu_count()}')

x1 = torch.randn(1, 2, 15, 15)
x8 = torch.randn(8, 2, 15, 15)
x32 = torch.randn(32, 2, 15, 15)


def bench(fn, x, iters=15, warmup=4):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def run(threads, use_bf16, x, iters):
    torch.set_num_threads(threads)
    if use_bf16:
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            return bench(lambda inp: net(inp), x, iters=iters)
    return bench(lambda inp: net(inp), x, iters=iters)


print('\n== single-request latency (ms, lower=better) ==')
print(f"{'threads':>8} | {'b1 fp32':>8} {'b1 bf16':>8} | {'b8 fp32':>8} {'b8 bf16':>8} | {'b32 fp32':>9} {'b32 bf16':>9}")
for t in (4, 8, 12, 16, 24, 32):
    row = [run(t, False, x1, 25), run(t, True, x1, 25),
           run(t, False, x8, 12), run(t, True, x8, 12),
           run(t, False, x32, 5), run(t, True, x32, 5)]
    print(f'{t:>8} |' + ''.join(f'{v:>9.1f} ' for v in row))

print('\n== concurrent batch=1 worker throughput (req/s, higher=better): simulates N parallel MCTS games ==')
for t, w in ((32, 1), (16, 2), (8, 4), (4, 8), (16, 1), (8, 2), (8, 1)):
    torch.set_num_threads(t)
    stop = threading.Event()
    cnt = [0]
    lock = threading.Lock()

    def worker():
        local = 0
        with torch.no_grad():
            while not stop.is_set():
                net(x1)
                local += 1
        with lock:
            cnt[0] += local

    ths = [threading.Thread(target=worker) for _ in range(w)]
    for th in ths:
        th.start()
    time.sleep(2.5)
    stop.set()
    for th in ths:
        th.join()
    rps = cnt[0] / 2.5
    print(f'threads={t:>2} workers={w} -> {rps:6.1f} req/s')
