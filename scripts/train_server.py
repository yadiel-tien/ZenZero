import logging
import os
import signal
import sys

from core.utils.functions import register_sigint
from core.utils.logger import install_thread_excepthook
from core.utils.config import game_name
from services.inference.train_server import TrainServer
from services.train.selfplay import SelfPlayManager
from core.network.functions import read_best_index, feature_version_compatible
from core.utils.logger import get_logger


def main():
    """统一训练入口：推理引擎 + selfplay + 模型训练运行在同一进程中。

    MCTS 通过队列直接请求推理，训练命令为直接方法调用，无需socket。
    启动命令: python -X gil=0 -m scripts.train_server
    (首次forward会惰性导入triton并试图重开GIL，-X gil=0 强制保持GIL关闭，
    保证100个MCTS线程真并行；triton仅用于内核编译，风险可接受)
    """
    # 任何线程的未捕获异常都落 logs/crash（否则只在终端一闪，日志里查不到）
    install_thread_excepthook()
    logger = get_logger('train')
    from core.utils.config import get_profile, settings as game_settings
    from core.network.registry import known as known_archs
    logger.info(f"Active profile: '{get_profile()}', archs registered: {known_archs()}, "
                f"net: {game_settings['default_net'].get('arch')} "
                f"{game_settings['default_net']['n_filters']}x{game_settings['default_net']['n_res_blocks']}, "
                f"feature_version: {game_settings.get('feature_version', 1)}")
    # ── 信号处理器必须在任何重活之前注册 ──
    # TrainServer 构造要装载 GB 级仓与权重；handler 晚注册的话，启动窗口里收到的 SIGINT
    # 走 Python 默认动作 = KeyboardInterrupt 打断主线程（10/28 演练实锤：stop 落在
    # buffer.load 里，进程以 -2 退出），SIGUSR1/SIGTERM 的默认动作同样是直接杀进程。
    # handler 经 state 取 manager：启动阶段（manager 未出生）收到停止 = 立刻体面退出，
    # 此时没有任何值得保存的东西。
    state = {'manager': None}
    server = None

    def _bounded_shutdown(timeout: float = 20.0) -> None:
        """收尾限时：另起线程做清理，主流程到点就走——清理不完整的代价远小于进程杀不掉
        （10/28 10:01 那次：数据 10:01:25 已存盘，收尾却吊死，10:02:37 被外部 SIGKILL，
        状态从此误导人）。"""
        import threading

        def _do():
            m = state['manager']
            if m is not None:
                m.shutdown()
            if server is not None:
                server.shutdown()

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            logger.warning(f'收尾 {timeout:.0f}s 未完成（有线程卡在驱动/队列），不再等待，直接退出')

    def graceful_stop(sig, frame):
        """第一次 SIGINT（HTTP"优雅停止"同义）：打完手上这波局、存盘、退出；再按一次硬退。"""
        m = state['manager']
        if m is None:
            logger.warning('收到停止请求（启动阶段，尚无对局与数据）：直接退出')
            os._exit(0)
        if m.stop_event.is_set():
            logger.warning('再次收到中断：强制退出')
            _bounded_shutdown(timeout=5.0)
            os._exit(1)
        logger.info('收到停止请求：打完手上这波局就收尾（再按一次强制退出）')
        m.stop_event.set()

    def stop_after_round(sig, frame):
        """SIGUSR1：本轮（自博弈→fit→评估→perf）全做完再退；启动阶段收到视同立即退。"""
        m = state['manager']
        if m is None:
            logger.warning('收到"等本轮结束"请求（启动阶段，无轮可等）：直接退出')
            os._exit(0)
        m.stop_after_round = True
        logger.info('收到"等本轮结束"请求：完成当前这一轮后退出（想立刻停就再发 SIGINT）')

    def hard_stop(sig, frame):
        """SIGTERM：尽力收尾 + 限时，然后硬退。"""
        _bounded_shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, graceful_stop)          # 立即结束
    signal.signal(signal.SIGUSR1, stop_after_round)      # 等本轮结束
    signal.signal(signal.SIGTERM, hard_stop)

    from core.utils import buffers as bufmod
    from core.utils.config import CONFIG, get_profile
    from services.inference.functions import get_checkpoint_path
    import os as _os

    # ── 仓实体（⑤）：ZZ_BUFFER=<名字|default@模型>，空=当前模型原位仓（原行为零变化）──
    buf_ref = (_os.environ.get('ZZ_BUFFER') or '').strip()
    bdir, berr, _bm = bufmod.resolve_for_train(buf_ref, game_name) if buf_ref else (None, '', {})
    if berr:
        logger.error(f'训练拒绝启动（仓闸门）：{berr}')
        logging.shutdown()
        os._exit(2)
    if bdir is not None:
        CONFIG['buffer_dir'] = str(bdir)

    # ── 训练起点（④）：ZZ_START_FROM=auto|fresh|<代数N>，默认 auto=续最强 ──
    best_idx = read_best_index()
    latest_idx = __import__('core.network.functions', fromlist=['x']).read_latest_index()
    model_idx, note, serr = bufmod.resolve_start_from(
        _os.environ.get('ZZ_START_FROM') or 'auto', best_idx, latest_idx,
        exists_fn=lambda i: _os.path.exists(get_checkpoint_path(game_name, i)),
        compat_fn=lambda i: i == -1 or feature_version_compatible(i))
    if serr:
        logger.error(f'训练拒绝启动（起点闸门）：{serr}')
        logging.shutdown()
        os._exit(2)
    logger.info(f'起点：{note} | 仓：{buf_ref or f"default@{get_profile()}（原位）"}'
                + (f' → {bdir}' if bdir else ''))
    # 进程内推理引擎(同时承担训练)
    server = TrainServer(model_idx, game_name)
    server.start()

    # selfplay管理器，直接使用上面的推理引擎
    from core.utils.config import CONFIG
    manager = SelfPlayManager(n_workers=int(CONFIG['selfplay_workers']), train_server=server)
    state['manager'] = manager          # 信号处理器从此刻起走"优雅停止"正路

    try:
        # 每轮 selfplay → fit → 评估，循环。局数远大于线程数：线程池滚动补位，
        # 尾段不再是"一波同时开跑的 50 局只剩几局"（见 CONFIG['selfplay_games_per_round']）
        manager.run(int(CONFIG['selfplay_games_per_round']))
    finally:
        _bounded_shutdown()
    # 正常跑完（含优雅停止）：数据都已存盘、引擎已关，直接硬退，
    # 不让 torch/CUDA/atexit 的退役卡住进程（否则外部只能 SIGKILL，状态显示也误导）。
    # 注意：抛异常的路径不会走到这里，traceback 照常打印。
    logging.shutdown()
    os._exit(0)


if __name__ == '__main__':
    main()
