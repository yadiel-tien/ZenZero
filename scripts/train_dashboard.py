"""训练监控面板入口。

用法:
    .venv/bin/python -m scripts.train_dashboard                # 0.0.0.0:8099
    .venv/bin/python -m scripts.train_dashboard --port 9000
    .venv/bin/python -m scripts.train_dashboard --selftest     # 冒烟自检，不起常驻服务

面板包含：每轮吞吐/批大小/每模拟耗时曲线、Loss 与验证集、评估胜率、
CPU/GPU/内存实时占用与历史、当前阶段进度、线程崩溃记录（logs/crash）。
只读，不影响正在跑的训练。
"""
import argparse

from services.monitor.app import _selftest, serve


def main() -> None:
    ap = argparse.ArgumentParser(description='ZenZero 训练监控面板')
    ap.add_argument('--host', default='0.0.0.0', help='监听地址（默认 0.0.0.0，可从别的机器打开）')
    ap.add_argument('--port', type=int, default=8099, help='监听端口（默认 8099）')
    ap.add_argument('--selftest', action='store_true', help='只跑冒烟自检，不起常驻服务')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    serve(args.host, args.port)


if __name__ == '__main__':
    main()
