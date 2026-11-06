"""ZenZero 主入口 —— 一个命令拉起整套服务。

用法（默认绑定本机回环 127.0.0.1:5000）：
    python main.py
局域网开放 / 换端口：
    python main.py --host 0.0.0.0 --port 8080

提供（均由本进程承载，控制路径全部走 Web）：
    http://<host>:<port>/            对弈（人机 / 双人 / 观战，含模型选择）
    http://<host>:<port>/admin       训练控制台（监控 · 启停训练 · 制品 · 日志 · 任务）
    http://<host>:<port>/docs/       项目文档集（六篇）

scripts/ 下的其余功能经控制台"任务"页启停：
    scripts.train_server（训练页）、scripts.piskvork_server、scripts.arena、
    scripts.bench_batch_sweep、scripts.build_cython（任务页）。
scripts.cli_play 仍保留为纯终端对战（不依赖本服务）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:                       # 允许从任意工作目录启动
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                      # config 的相对路径以仓库根为基准


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='ZenZero', description='ZenZero 一体化入口（对弈 + 控制台 + 文档）')
    ap.add_argument('--host', default='127.0.0.1',
                    help='绑定地址（默认本机回环；局域网开放用 0.0.0.0）')
    ap.add_argument('--port', type=int, default=5000, help='端口（默认 5000）')
    ap.add_argument('--debug', action='store_true', help='Flask debug（仅开发用）')
    ap.add_argument('--version', action='version', version=f'%(prog)s {__import__("core.utils.version", fromlist=[""]).full()}',
                    help='显示版本号（含 git 构建号）')
    args = ap.parse_args(argv)

    from services.inference.play_api import PlayApiServer      # 延后到 path/chdir 之后再 import

    server = PlayApiServer()
    if not args.host.startswith('127.') and args.host != 'localhost':
        server.logger.warning(f'绑定 {args.host}：局域网内任何人都可访问控制台（含启停训练）。'
                              f'建议设 ZZ_ADMIN_TOKEN 为写操作加口令。')
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
