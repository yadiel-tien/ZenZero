"""优雅停止：HTTP/信号停止时"取消没开跑的、把手上这波收完、存盘、退出"。

为什么值得单测：从面板点"停止训练"是日常操作。旧实现只在轮边界检查信号，中途停止
只能 SIGTERM 硬杀——已经算出来的几百局样本全丢（buffer 只在 self_play 结尾存一次）。
这里钉住：
  1. 置位 stop_event 后，**未开始**的 futures 被 cancel、不再起新局；
  2. **已经在跑的局不丢**：它们的样本照常入池，且 buffer.pkl 真的落盘；
  3. 已停止时 run() 直接返回，不进 fit、不进评估（半成品模型不该进 checkpoint、更不该去对擂）。

运行：.venv/bin/python tests/test_graceful_stop.py
"""
import os
import shutil
import sys
import tempfile
from concurrent.futures import Future

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from core.utils.config import CONFIG, settings  # noqa: E402
from core.utils.replay import ReplayBuffer  # noqa: E402

_tmp = None
_backup = None


class _Payload:
    """一局的返回值：(samples, winner, final_state, mcts_stat)"""

    def __init__(self, shape, n_samples=2):
        self.samples = [(np.zeros(shape, dtype=np.float32),
                         np.zeros(settings['default_net']['n_actions'], dtype=np.float32) + 1e-3,
                         0.5) for _ in range(n_samples)]


class _CountingFuture(Future):
    """result() 时触发回调——用来在第 N 局后置位停止信号（模拟"用户点了停止"）"""

    def __init__(self, payload, cb):
        super().__init__()
        self._payload = payload
        self._cb = cb

    def result(self, timeout=None):
        if self.cancelled():
            raise RuntimeError('cancelled')
        self._cb()
        return self._payload


class _FakePool:
    """前 n_done 个 future 已完成（=已在跑的局），其余 pending（=还没开跑的局，可 cancel）"""

    def __init__(self, n_total, n_done, payload, on_result):
        self.futures = []
        for i in range(n_total):
            f = _CountingFuture(payload, on_result)
            if i < n_done:
                f.set_result(payload)        # set_result 后固定，cancel() 返回 False
            self.futures.append(f)
        self._next = 0

    def submit(self, fn, *args, **kwargs):
        f = self.futures[self._next]
        self._next += 1
        return f


class _StubServer:
    """SelfPlayManager 需要的最小面：共享 buffer + 推理统计"""

    def __init__(self, buffer):
        self.buffer = buffer
        self.request_queue = None
        self.feature_version = 2
        self.total_steps_trained = 0
        self.writer = None

    def infer_stats(self):
        return {}


def setUp():
    global _tmp, _backup
    _tmp = tempfile.mkdtemp(prefix='zz_stop_')
    _backup = (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'], settings['buffer_size'])
    CONFIG['data_dir'] = os.path.join(_tmp, 'data')
    CONFIG['runs_dir'] = os.path.join(_tmp, 'runs')
    CONFIG['log_dir'] = os.path.join(_tmp, 'logs')
    settings['buffer_size'] = 64          # 小仓：别为测试分配几 GB


def tearDown():
    global _tmp
    if _backup:
        CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'], settings['buffer_size'] = _backup
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


def _make_manager(n_games: int, n_done: int, stop_after: int | None):
    from services.train.selfplay import SelfPlayManager
    buf = ReplayBuffer(settings['buffer_size'], 8, game=CONFIG['game_name'])
    mgr = SelfPlayManager(n_workers=4, train_server=_StubServer(buf))
    payload = _Payload((26, 10, 9))
    counter = {'n': 0}

    def on_result():
        counter['n'] += 1
        if stop_after is not None and counter['n'] >= stop_after:
            mgr.stop_event.set()

    mgr.pool = _FakePool(n_games, n_done,
                         (payload.samples, 0, np.zeros((7, 10, 9), np.int32),
                          {'select_s': 0.0, 'wait_s': 0.0, 'sims': 400}), on_result)
    return mgr, payload


def test_stop_cancels_pending_keeps_inflight():
    setUp()
    try:
        # 50 局提下去：6 局已在跑（已完成），44 局还没开跑
        mgr, payload = _make_manager(n_games=50, n_done=6, stop_after=3)
        done_samples = mgr.self_play(iteration=1, n_games=50, round_index=0)
        games = mgr._last_sp['games']
        assert mgr.stop_event.is_set()
        # 在跑的 6 局全收，没开跑的 44 局被取消
        assert games == 6, f'应保住 6 局在跑的数据，实际 {games} 局'
        cancelled = sum(1 for f in mgr.pool.futures if f.cancelled())
        assert cancelled == 44, f'应取消 44 局，实际 {cancelled}'
        assert done_samples == 6 * len(payload.samples), done_samples
        # 留出集是按样本随机 2% 划走的（random.random() < holdout），所以断言必须看"两池合计"，
        # 只盯 buffer.size 会以 ~22% 的概率偶发失败（12 个样本里有点进验证池就变成 11）。
        total = mgr.buffer.size + mgr.val_buffer.size
        assert total == 6 * len(payload.samples), (mgr.buffer.size, mgr.val_buffer.size)
        from core.utils.config import net_dir
        saved = os.path.join(net_dir(CONFIG['game_name']), mgr.buffer.file_name)
        assert os.path.exists(saved), f'buffer 没存盘: {saved}'
        print(f'   ok  停止后：在跑 {games} 局样本入池并存盘，未开始 44 局取消')
    finally:
        tearDown()


def test_run_returns_when_stopped():
    setUp()
    try:
        mgr, _payload = _make_manager(n_games=50, n_done=0, stop_after=None)
        mgr.stop_event.set()
        called = {'fit': 0, 'eval': 0}
        mgr.train_server.fit = lambda *a, **k: called.__setitem__('fit', called['fit'] + 1)
        mgr.evaluation = lambda *a, **k: called.__setitem__('eval', called['eval'] + 1)
        mgr.run(n_games=50)
        assert called == {'fit': 0, 'eval': 0}, f'已停止还进了 fit/eval: {called}'
        print('   ok  stop_event 已置位时 run() 直接返回（不 fit、不评估）')
    finally:
        tearDown()


def main() -> None:
    for fn in (test_stop_cancels_pending_keeps_inflight, test_run_returns_when_stopped):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 2 GRACEFUL-STOP TESTS PASSED')


if __name__ == '__main__':
    main()
