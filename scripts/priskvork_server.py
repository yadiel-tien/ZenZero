import socket
import traceback
import time

from core.env.gomoku import Gomoku
from core.network.functions import read_best_index
from core.player.ai_server import AIServer
from services.inference.hub import ServerHub


class GomokuSession:
    def __init__(self, conn: socket.socket, model_id: int, n_simulation: int, hub: ServerHub) -> None:
        self.conn = conn
        self.f_reader = conn.makefile('r', encoding='utf-8')
        self.player = AIServer('Gomoku', model_id=model_id, n_simulation=n_simulation, verbose=False, hub=hub)
        self.env = Gomoku()
        self.is_running = True

    def run(self) -> None:
        print(f'GomokuSession {self.player.description} started.')
        while self.is_running:
            line = self.f_reader.readline()
            if not line:
                break
            line = line.strip()
            print(line)
            if not line:
                continue

            try:
                response = self.dispatch(line)
                if response:
                    self.send_response(response)
            except Exception as e:
                print(f"[Warn] Error processing cmd '{line}': {e}")
                traceback.print_exc()

    def send_response(self, response: str) -> None:
        try:
            self.conn.sendall(f'{response}\n'.encode('utf-8'))
        except OSError:
            self.is_running = False

    def dispatch(self, line: str) -> str | None:
        """解析指令，进行相应操作"""
        parts = line.split()
        cmd = parts[0].upper()

        if cmd == 'START':
            return self.handle_start(parts)
        elif cmd == 'TURN':
            return self.handle_turn(parts)
        elif cmd == 'BEGIN':
            return self.handle_begin()
        elif cmd == 'BOARD':
            return self.handle_board()
        elif cmd == 'INFO':
            return None
        elif cmd == 'ABOUT':
            return 'name="ZenZero", version="1.0", country="CN"'
        elif cmd == 'SWAP2BOARD':
            return 'Error Swap2 rule was not supported'
        elif cmd == 'RESTART':
            return self.handle_restart()
        elif cmd == 'END':
            print("[System] END command received.")
            self.is_running = False
            return None
        else:
            print(f"[Warn] Unknown command: {cmd}")
            return None

    def handle_start(self, parts: list[str]) -> None | str:
        """获取棋盘尺寸，初始化成功回复OK"""
        if len(parts) < 2:
            return "ERROR missing size"
        size = int(parts[1])
        if size != 15:
            return "ERROR Unsupported board size"

        self.player.reset()
        self.env.reset()
        return "OK"

    def handle_turn(self, parts: list[str]) -> None | str:
        """解析对手行棋，并给出我方行棋"""
        # 解析对手行棋，并执行
        x, y = map(int, parts[1].split(','))
        self.env.step(self.env.move2action((y, x)))

        # 我方行棋
        return self.get_ai_move()

    def handle_begin(self) -> None | str:
        """我方先手，给出第一步棋"""
        self.player.reset()
        self.env.reset()
        return self.get_ai_move()

    def handle_board(self):
        """根据指令摆棋"""
        self.player.reset()
        self.env.reset()

        while True:
            # 注意：这里读取必须小心，如果中间断网了需要抛出异常给外层
            sub_line = self.f_reader.readline()
            if not sub_line:
                raise ConnectionError("Connection lost inside BOARD")

            sub_line = sub_line.strip()
            if sub_line.upper() == 'DONE':
                break

            # 解析每一行棋子
            try:
                b_parts = sub_line.split(',')
                if len(b_parts) == 3:
                    bx, by, _ = int(b_parts[0]), int(b_parts[1]), int(b_parts[2])
                    self.env.step(self.env.move2action((by, bx)))
            except ValueError:
                print(f"[Warn] Skipped bad board line: {sub_line}")

            # 摆完后思考
        return self.get_ai_move()

    def handle_restart(self) -> None | str:
        self.player.reset()
        self.env.reset()
        return "OK"

    def get_ai_move(self) -> str | None:
        """根据当前局面行棋，update函数适配ui，需放入循环体"""
        if self.player.pending_action == -1:
            self.player.update(self.env.state, self.env.last_action, self.env.player_to_move)

        action = self.player.pending_action
        row, col = self.env.action2move(action)
        self.env.step(action)
        self.player.pending_action = -1
        return f'{col},{row}'

    def shutdown(self) -> None:
        self.is_running = False
        self.f_reader.close()
        self.conn.close()
        self.player.shutdown()


def run_server(host='0.0.0.0', port=12345) -> None:
    # 进程内推理引擎管理中心，多个会话共享引擎
    hub = ServerHub()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 允许端口复用，防止重启时报错 "Address already in use"
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"=================================================")
        print(f" AI Server running on {host}:{port}")
        print(f" Waiting for Piskvork (Windows) to connect...")
        print(f"=================================================")

        while True:
            try:
                session = None
                conn, addr = s.accept()
                print(f"[Net] New connection from {addr}")
                best_idx = read_best_index('Gomoku')
                session = GomokuSession(conn, model_id=best_idx, n_simulation=1000, hub=hub)
                session.run()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Error] Error processing cmd '{addr}': {e}")
            finally:
                if session:
                    print(f"[Net] Connection closed. Waiting for next match...")
                    session.shutdown()
    print(f"[Net] Gomoku server was shutdown.")


if __name__ == "__main__":
    """适用于Priskvork对战协议，具体协议内容见https://plastovicka.github.io/protocl2en.htm"""
    run_server()
