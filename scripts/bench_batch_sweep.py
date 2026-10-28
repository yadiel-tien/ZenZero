"""批量大小扫描：实测推理路径 t(b) 曲线，回答"批多大最划算"。

为什么需要：selfplay 吞吐 ≈ 批大小 / 每批耗时。每批耗时 = 固定开销 + 每样本边际成本，
两者拆开后才能判断该把 max_batch_size 调到多少、固定开销还值不值得抠。
本脚本对同一份网络（当前 profile 的形状）跑三组测量：

  A. model-only     —— 只算 forward（GPU 纯算力+kernel 发射）
  B. engine-path    —— 照抄 engine._pre_infer_loop/_inference_loop 的完整路径
                        （np.stack → from_numpy → pin_memory → to(dev) → autocast → softmax → cpu）
  C. engine-nopin   —— 同上但去掉 pin_memory（P2 待验证的假设：pin 是纯亏的额外拷贝）

用法（需要 GPU 设备节点，容器/沙箱里跑不了）：
    .venv/bin/python -m scripts.bench_batch_sweep            # 默认 profile
    ZZ_PROFILE=s .venv/bin/python -m scripts.bench_batch_sweep --batches 1,8,32,64,128,256

输出：终端表格 + logs/perf/batch_sweep.json（供后续对照）。
注意：会和正在跑的训练抢 GPU —— 只跑 1~2 分钟，显存 <1GiB。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import threading
import time

import numpy as np
import torch

from core.utils.config import CONFIG, settings, get_profile
import core.network.network  # noqa: F401  触发 @register，注册表不导入就是空的
from core.network.registry import build, DEFAULT_ARCH


def make_model(cfg: dict, device: str) -> torch.nn.Module:
    arch = cfg.get('arch', DEFAULT_ARCH)
    model = build(arch, cfg, eval_model=True)
    model = model.to(device).eval()
    return model


def timeit(fn, iters: int, warmup: int = 3, repeats: int = 5) -> float:
    """单次调用的墙钟毫秒 = 多次重复里的最小值。

    不在每次调用后 sync（那会把每次 ~50-100µs 的同步开销算进结果），改成连续跑
    iters 次只 sync 一次；再重复 repeats 轮取最小值——GPU 与正在跑的训练共享，
    只有最小值才接近"独占时"的真实成本。"""
    for _ in range(warmup):
        fn()
    best = float('inf')
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e3)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default='1,2,4,8,16,32,64,128,256,512')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(',') if x.strip()]

    device = str(CONFIG['device'])
    if not device.startswith('cuda'):
        device = 'cuda:0' if torch.cuda.is_available() else device
    cfg = dict(settings['default_net'])
    model = make_model(cfg, device)
    shape = (cfg['in_channels'], 10, 9) if CONFIG['game_name'] == 'ChineseChess' else (cfg['in_channels'], 15, 15)
    n_actions = cfg['n_actions']
    dev_type = torch.device(device).type
    ac = (lambda: torch.amp.autocast(dev_type)) if dev_type in ('cuda', 'xpu') else contextlib.nullcontext
    print(f"profile={get_profile()} device={device} arch={cfg.get('arch', DEFAULT_ARCH)} "
          f"{cfg['n_filters']}x{cfg['n_res_blocks']} planes={shape[0]} x {shape[1]}x{shape[2]} "
          f"actions={n_actions}")
    print(f"{'b':>5} {'A model-only':>13} {'B path(pin)':>12} {'C path(nopin)':>13} "
          f"{'D full-stack':>13} {'D us/sample':>12} {'D req/s':>10}")

    rows = []
    for b in batches:
        x_np = np.random.rand(b, *shape).astype(np.float32)
        x_t = torch.from_numpy(x_np).to(device)

        def run_model(x_t=x_t):
            with torch.inference_mode(), ac():
                logits, values = model(x_t)
            return logits

        def run_path(pin: bool, x_np=x_np):
            t = torch.from_numpy(x_np)
            if pin and torch.cuda.is_available():
                t = t.pin_memory()
            with torch.inference_mode():
                t = t.to(device, dtype=torch.float32, non_blocking=pin)
                with ac():
                    logits, values = model(t)
                probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
                vals = values.float().cpu().numpy()
            return probs, vals

        # D: 完整引擎路径 = 逐请求 np.stack + tensor 搬运 + forward + 逐条投递
        states = [np.random.rand(*shape).astype(np.float32) for _ in range(b)]
        events = [threading.Event() for _ in range(b)]

        def run_full(b=b):
            batch = np.stack(states)
            t = torch.from_numpy(batch).to(device, dtype=torch.float32)
            with torch.inference_mode(), ac():
                logits, values = model(t)
            probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
            vals = values.float().cpu().numpy()
            for i, ev in enumerate(events):
                ev.set()
            return probs, vals

        iters = max(5, min(args.iters, int(3000 / max(b, 1))))
        t_model = timeit(run_model, iters)
        t_path = timeit(lambda: run_path(True), iters)
        t_nopin = timeit(lambda: run_path(False), iters)
        t_full = timeit(run_full, iters)
        rows.append({'batch': b, 'model_ms': round(t_model, 3), 'path_ms': round(t_path, 3),
                     'nopin_ms': round(t_nopin, 3), 'full_ms': round(t_full, 3),
                     'full_req_per_s': round(b / t_full * 1e3),
                     'path_us_per_sample': round(t_path / b * 1e3, 1),
                     'nopin_us_per_sample': round(t_nopin / b * 1e3, 1),
                     'nopin_req_per_s': round(b / t_nopin * 1e3)})
        r = rows[-1]
        print(f"{b:>5} {t_model:>12.2f}ms {t_path:>11.2f}ms {t_nopin:>12.2f}ms "
              f"{t_full:>12.2f}ms {t_full / b * 1e3:>10.1f}us {r['full_req_per_s']:>9.0f}/s")

    # 线性拟合 t = a + c*b（用 engine-nopin 的曲线）
    bs = np.array([r['batch'] for r in rows], dtype=np.float64)
    ts = np.array([r['full_ms'] for r in rows], dtype=np.float64)
    a_mat = np.vstack([np.ones_like(bs), bs]).T
    (a, c), *_ = np.linalg.lstsq(a_mat, ts, rcond=None)
    print(f"\n拟合(engine-nopin)：t(b) = {a:.2f}ms + {c * 1e3:.1f}us x b "
          f"→ 边际 {1 / c:.0f} 样本/s，固定开销占 b=1 的 {a / (a + c) * 100:.0f}%")
    for bb in (16, 32, 64, 128, 256, 512):
        t = a + c * bb
        print(f"  b={bb:>4}: {t:6.2f}ms/batch → {bb / t * 1e3:7.0f} req/s")

    out = args.out or os.path.join(CONFIG['log_dir'], 'perf', 'batch_sweep.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fp:
        json.dump({'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'profile': get_profile(),
                   'device': device, 'net': cfg, 'rows': rows,
                   'fit': {'fixed_ms': round(a, 3), 'per_sample_ms': round(c, 4)}},
                  fp, ensure_ascii=False, indent=2)
    print(f"\n写入 {out}")


if __name__ == '__main__':
    main()
