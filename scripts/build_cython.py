"""Cython 扩展的安全构建：暂存目录编译 → 原子替换。

为什么不能直接 `setup.py build_ext --inplace`：它内部是 shutil.copyfile，
**原地截断重写**同一个 .so 文件。已经在跑的 Python 进程把旧 .so mmap 进了地址空间，
文件被截断/改写后它的映射就指向了垃圾页 —— 几分钟后某次函数调用直接吃 SIGBUS/SIGSEGV，
而且 Python 层什么都记不下（10/27 20:45 就因此打死了正在跑的训练：
.so 20:45:34 被重写，进程 20:50:56 后消失，日志里零 traceback）。

本脚本：先 build 到临时目录，再用 os.replace 原子改名覆盖目标——
旧 inode 对已在跑的进程继续有效，新进程才拿到新版本。

用法：
    .venv/bin/python scripts/build_cython.py            # 构建 chess_cython
    .venv/bin/python scripts/build_cython.py --check    # 只检查现有 .so 能否导入且含新符号
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, 'core', 'env')
SO_GLOB = 'chess_cython.*.so'


def build() -> int:
    # 暂存目录必须与目标**同一个文件系统**，否则 os.replace 会 EXDEV（/tmp 是 tmpfs，
    # 跨设备不能 rename）。放在 core/env 下的点目录里，天然同盘。
    tmp = tempfile.mkdtemp(prefix='.build_stage_', dir=ENV_DIR)
    try:
        cmd = [sys.executable, 'setup.py', 'build_ext',
               '--build-temp', os.path.join(tmp, 'temp'),
               '--build-lib', os.path.join(tmp, 'lib')]
        print(f'[build] {" ".join(cmd)}  (cwd={ENV_DIR})')
        proc = subprocess.run(cmd, cwd=ENV_DIR)
        if proc.returncode != 0:
            print('[build] 编译失败，目标 .so 未改动（跑着的进程不受影响）')
            return proc.returncode
        built = glob.glob(os.path.join(tmp, 'lib', SO_GLOB))
        if not built:
            print('[build] 没找到产物，放弃')
            return 1
        src = built[0]
        dst = os.path.join(ENV_DIR, os.path.basename(src))
        os.replace(src, dst)          # 原子改名：旧 inode 对运行中的进程继续有效
        print(f'[build] 原子替换完成: {dst} ({os.path.getsize(dst)} bytes)')
        print('[build] 提示：正在跑的训练进程仍使用旧代码，需重启才会加载新 .so')
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check() -> int:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        mod = importlib.import_module('core.env.chess_cython')
    except Exception as e:                                   # noqa: BLE001
        print(f'[check] 导入失败: {type(e).__name__}: {e}')
        return 1
    names = sorted(n for n in dir(mod) if not n.startswith('_'))
    print(f'[check] 导入成功: {getattr(mod, "__file__", "?")}')
    print(f'[check] 导出符号: {", ".join(names)}')
    missing = [n for n in ('get_valid_actions', 'is_check', 'convert_v2') if n not in names]
    if missing:
        print(f'[check] 缺符号: {missing}（转换会自动降级到 Python，但会慢 ~30 倍）')
        return 2
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description='Cython 扩展安全构建（暂存 + 原子替换）')
    ap.add_argument('--check', action='store_true', help='只检查现有 .so')
    args = ap.parse_args()
    sys.exit(check() if args.check else build())


if __name__ == '__main__':
    main()
