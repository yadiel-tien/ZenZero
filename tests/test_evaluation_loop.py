"""评估环（`SelfPlayManager.evaluation`）的端到端回归。

为什么单独建这个文件：这个环**从来没有被真正跑过**——
演示脚本里 `best_index == -1` 走冷启动分支直接 return，smoke 里也是冷启动，
于是 `evaluation()` 主体一直是测试盲区。10/27 21:42 就因此放过了一个真回归：
改 `self_play` 的 import 时把 `as_completed` 删了，`evaluation()` 仍在用它 →
提交 100 局后立刻 `NameError`，`finally` 先把两个评测引擎 shutdown 掉，
32 个池线程卡在等推理（300s 超时），主线程堵在 `pool.shutdown()`，训练"看着卡死"。

这里用微型网 + 真 checkpoint 把整条链路跑通：
  两个引擎建起来并启动 → 布局评估 → 至少打完一局 → 返回结构化结果
  → 两个引擎的线程在 finally 里全部回收。
运行：.venv/bin/python tests/test_evaluation_loop.py
"""
import faulthandler
import logging
import os
import shutil
import sys
import tempfile
import threading

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'
os.environ['ZZ_GAME'] = 'Gomoku'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.utils.config import CONFIG, settings  # noqa: E402

_tmp = None
_backup = None


def setUp():
    global _tmp, _backup
    _tmp = tempfile.mkdtemp(prefix='zz_eval_')
    _backup = (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'],
               dict(settings['default_net']), settings['buffer_size'], settings['max_iters'])
    CONFIG['data_dir'] = _tmp
    CONFIG['runs_dir'] = os.path.join(_tmp, 'runs')
    CONFIG['log_dir'] = os.path.join(_tmp, 'logs')
    os.makedirs(os.path.join(_tmp, 'Gomoku'), exist_ok=True)
    settings['default_net'].clear()
    settings['default_net'].update({
        'arch': 'resnet_v1', 'in_channels': 2, 'n_filters': 8, 'n_res_blocks': 1,
        'n_cells': 225, 'n_actions': 225, 'use_se': False, 'n_policy_filters': 2,
        'n_value_filters': 1, 'n_value_hidden_channels': 16, 'feature_version': 1})
    settings['buffer_size'] = 512
    settings['max_iters'] = 1


def tearDown():
    global _tmp
    if _backup:
        (CONFIG['data_dir'], CONFIG['runs_dir'], CONFIG['log_dir'],
         net, settings['buffer_size'], settings['max_iters']) = _backup
        settings['default_net'].clear()
        settings['default_net'].update(net)
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


def test_evaluation_round_trip():
    setUp()
    srv = None
    try:
        from services.inference.train_server import TrainServer
        from services.train.selfplay import SelfPlayManager
        import services.train.selfplay as sp_mod
        sp_mod.ramp_simulations = lambda iteration, sims=None: 32   # 只验链路，不追求棋力

        srv = TrainServer(-1, 'Gomoku')
        srv.start()
        srv.save_checkpoint(1)                            # 守擂
        srv.save_checkpoint(2)                            # 挑战者
        mgr = SelfPlayManager(n_workers=1, train_server=srv)
        mgr.best_index = 1                                # 关键：避开冷启动分支

        before = {t.name for t in threading.enumerate()}
        # 看门狗：任何回归导致的卡死都在 240s 处炸出全部线程栈，而不是无限挂着
        faulthandler.dump_traceback_later(240, exit=True)
        try:
            out = mgr.evaluation(iteration=2, n_games=1)
        finally:
            faulthandler.cancel_dump_traceback_later()

        for key in ('games', 'win', 'lose', 'draw', 'win_rate', 'promoted', 'wall'):
            assert key in out, f'evaluation 返回值缺 {key}: {out}'
        assert out['games'] >= 1, f'一局都没打完: {out}'
        assert abs(out['win'] + out['lose'] + out['draw'] - out['games']) <= 1, out
        assert 0.0 <= out['win_rate'] <= 1.0, out
        # 两个评测引擎的线程必须随 finally 一起回收（漏掉就会留半死引擎/常驻线程）
        leftover = {t.name for t in threading.enumerate() if 'Gomoku_2' in t.name or 'Gomoku_1' in t.name}
        assert not leftover, f'评测引擎线程没退干净: {leftover}'
        new_threads = {t.name for t in threading.enumerate()} - before
        print(f"   ok  评估环跑通：{out['games']} 局，胜{out['win']}负{out['lose']}平{out['draw']}，"
              f"胜率{out['win_rate']:.0%}，用时{out['wall']:.1f}s，引擎线程已回收"
              f"（新增常驻线程 {sorted(new_threads) or '无'}）")
    finally:
        try:
            if srv is not None:
                srv.shutdown()
        except Exception:                                  # noqa: BLE001 清理失败不影响断言
            pass
        tearDown()


def test_self_play_loop_names_exist():
    """静态兜底：self_play 与 evaluation 用到的并发原语都必须在模块命名空间里。

    这条专门针对"改 import 删掉了别处还在用的名字"这类回归（as_completed 就被这样删过）。
    """
    import services.train.selfplay as sp_mod
    for name in ('as_completed', 'wait', 'FIRST_COMPLETED', 'ThreadPoolExecutor'):
        assert hasattr(sp_mod, name), f'selfplay 模块缺少 {name}（import 被改坏了？）'
    src = open(sp_mod.__file__, encoding='utf-8').read()
    for used in ('as_completed(', 'wait(', 'FIRST_COMPLETED'):
        assert used.split('(')[0] in src, used
    print('   ok  selfplay 的并发原语齐全（as_completed / wait / FIRST_COMPLETED）')


def main() -> None:
    logging.disable(logging.CRITICAL)
    for fn in (test_self_play_loop_names_exist, test_evaluation_round_trip):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 2 EVALUATION-LOOP TESTS PASSED')


if __name__ == '__main__':
    main()
