"""线程崩溃可观测性回归：异常必须进日志，不能只在终端一闪。

两次事故的形状一样——线程里的异常只打到 stderr，日志文件干干净净：
  · 10/27 20:32 引擎投递线程 `deque mutated during iteration` → 整轮静默卡死
  · 10/27 20:49 用户报"evaluate 报错"，但 logs/ 里没有任何非 INFO 行
这里钉住两条兜底：
  1. `install_thread_excepthook()` —— 任何线程的未捕获异常写 logs/crash/latest.log（带栈）
  2. `InferenceEngine._spawn()`   —— 引擎线程崩溃写 logs/inference 并置 stop_event（快速停）

运行：.venv/bin/python tests/test_crash_logging.py
"""
import logging
import os
import shutil
import sys
import tempfile
import threading
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from core.utils.config import CONFIG  # noqa: E402
from core.utils.logger import install_thread_excepthook  # noqa: E402
from core.network.network import Net  # noqa: E402
from services.inference.engine import InferenceEngine  # noqa: E402

CFG = {'in_channels': 2, 'n_filters': 8, 'n_res_blocks': 1, 'use_se': False,
       'n_policy_filters': 2, 'n_value_filters': 2, 'n_value_hidden_channels': 4,
       'n_cells': 9, 'n_actions': 16}

_tmp = None


def setUp():
    """日志目录隔离到临时目录：绝不往生产 logs/ 里写测试崩溃栈"""
    global _tmp
    _tmp = tempfile.mkdtemp(prefix='zz_crash_')
    CONFIG['log_dir'] = _tmp


def tearDown():
    global _tmp
    CONFIG['log_dir'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


def _read(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as fp:
            return fp.read()
    except OSError:
        return ''


def test_excepthook_writes_crash_log():
    setUp()
    old = threading.excepthook
    try:
        install_thread_excepthook('crash')

        def boom():
            raise RuntimeError('excepthook boom 探针')

        t = threading.Thread(target=boom, name='boom-thread')
        t.start()
        t.join(timeout=10)
        time.sleep(0.2)                       # 等文件 handler flush
        text = _read(os.path.join(_tmp, 'crash', 'latest.log'))
        assert 'excepthook boom 探针' in text, f'未捕获异常没进日志:\n{text[-500:]}'
        assert 'Traceback' in text and 'RuntimeError' in text, text[-500:]
        assert 'boom-thread' in text, '没记录线程名，定位不到是谁死的'
        print('   ok  未捕获线程异常 → logs/crash（含栈与线程名）')
    finally:
        threading.excepthook = old
        tearDown()


def test_engine_thread_crash_is_logged_and_stops():
    """引擎线程崩溃：栈进 logs/inference + stop_event 置位（不留下半死的引擎）"""
    setUp()
    model = Net(CFG, eval_model=True, device='cpu')
    InferenceEngine.load_model = classmethod(
        lambda cls, i, e, eval_mode=True, profile=None: (model, 0))
    eng = InferenceEngine(0, 'Gomoku')
    eng.start()
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    eng.logger.addHandler(handler)
    try:
        crashed = eng._spawn('probe', lambda: 1 / 0)     # 人为让引擎线程崩
        crashed.join(timeout=10)
        assert eng.stop_event.is_set(), '引擎线程崩溃后没有置 stop_event（会留下半死引擎）'
        msgs = [r.getMessage() for r in records]
        assert any('probe' in m and '崩溃' in m for m in msgs), f'引擎崩溃没记日志: {msgs}'
        # 栈也要落到文件里（logger.exception 带 exc_info）
        time.sleep(0.2)
        text = _read(os.path.join(_tmp, 'inference', 'latest.log'))
        assert 'ZeroDivisionError' in text, f'引擎崩溃栈没进文件:\n{text[-400:]}'
        print('   ok  引擎线程崩溃 → logs/inference 带栈 + stop_event 置位')
    finally:
        eng.logger.removeHandler(handler)
        eng.stop_event.set()
        try:
            eng.shutdown()
        except Exception:                     # noqa: BLE001 已崩引擎的清理失败不影响断言
            pass
        tearDown()


def main() -> None:
    # 注意：不能在 setUp() 之前 get_logger('crash')——handler 会绑到生产 logs/ 目录上，
    # 之后 get_logger 因为"已有 handler"直接返回，测试数据就写进生产日志了（踩过）。
    for fn in (test_excepthook_writes_crash_log, test_engine_thread_crash_is_logged_and_stops):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 2 CRASH-LOGGING TESTS PASSED')


if __name__ == '__main__':
    main()
