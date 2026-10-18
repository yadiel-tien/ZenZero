from core.utils.functions import register_sigint
from core.utils.config import game_name
from services.inference.train_server import TrainServer
from services.train.selfplay import SelfPlayManager
from core.network.functions import read_best_index


def main():
    """统一训练入口：推理引擎 + selfplay + 模型训练运行在同一进程中。

    MCTS 通过队列直接请求推理，训练命令为直接方法调用，无需socket。
    启动命令: python -X gil=0 -m scripts.train_server
    (首次forward会惰性导入triton并试图重开GIL，-X gil=0 强制保持GIL关闭，
    保证100个MCTS线程真并行；triton仅用于内核编译，风险可接受)
    """
    model_idx = read_best_index()
    # 进程内推理引擎(同时承担训练)
    server = TrainServer(model_idx, game_name)
    server.start()

    # selfplay管理器，直接使用上面的推理引擎
    manager = SelfPlayManager(n_workers=100, train_server=server)

    def shutdown_all():
        manager.shutdown()
        server.shutdown()

    register_sigint(shutdown_all)
    try:
        # 每轮selfplay 100局 -> fit -> 评估，循环
        manager.run(100)
    finally:
        shutdown_all()


if __name__ == '__main__':
    main()
