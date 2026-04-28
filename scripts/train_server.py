from utils.functions import register_sigint
from utils.config import game_name
import time
from inference.train_server import TrainServer
from network.functions import read_best_index, read_latest_index, save_best_index


def main():
    # 启动推理服务

    # model_idx = read_latest_index()
    model_idx = read_best_index()
    server = TrainServer(model_idx, game_name, 200)
    register_sigint(server.shutdown)
    server.start()
    while not server.stop_event.is_set():
        time.sleep(1)  # 阻塞主线程，避免退出
    server.shutdown()


if __name__ == '__main__':
    main()