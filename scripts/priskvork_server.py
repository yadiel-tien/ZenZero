import socket
import socketserver
import traceback

from core.env.gomoku import Gomoku
from core.network.functions import read_best_index
from core.player.ai_server import AIServer
from core.utils.logger import get_logger
from services.inference.hub import ServerHub

logger = get_logger('piskvork_server')

# 进程内推理引擎管理中心，多个会话共享引擎 (run_server中初始化)
hub: ServerHub | None = None


class GomokuSession:
    def __init__(self, conn: socket.socket, model_id: int, n_simulation: int, hub: ServerHub) -> None:
        self.conn = conn
        self.f_reader = conn.makefile('r', encoding='utf-8')
        self.player = AIServer('Gomoku', model_id=model_id, n_simulation=n_simulation, verbose=False, hub=hub)
        self.env = Gomoku()
        self.is_running = True
        self.timeout_turn = 5000  # 默认5秒
        self.timeout_match = 0
        self.time_left = 0

    def run(self) -> None:
        logger.info(f'GomokuSession {self.player.description} started.')
        while self.is_running:
            try:
                line = self.f_reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                logger.debug(f"Recv: {line}")
                response = self.dispatch(line)
                if response:
                    logger.debug(f"Send: {response}")
                    self.send_response(response)
            except (ConnectionError, socket.error):
                break
            except Exception as e:
                logger.error(f"Error processing cmd: {e}")
                traceback.print_exc()

    def send_response(self, response: str) -> None:
        try:
            self.conn.sendall(f'{response}\n'.encode('utf-8'))
        except OSError:
            self.is_running = False

    def dispatch(self, line: str) -> str | None:
        parts = line.split()
        if not parts:
            return None

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
            self.handle_info(parts)
            return None
        elif cmd == 'ABOUT':
            return 'name="ZenZero", version="1.1", country="CN"'
        elif cmd == 'SWAP2BOARD':
            return 'Error Swap2 rule was not supported'
        elif cmd == 'RESTART':
            return self.handle_restart()
        elif cmd == 'END':
            logger.info("END command received.")
            self.is_running = False
            return None
        else:
            logger.warning(f"Unknown command: {cmd}")
            return None

    def handle_info(self, parts: list[str]) -> None:
        """处理信息指令，如超时时间等"""
        if len(parts) < 3:
            return
        key = parts[1].lower()
        try:
            val = int(parts[2])
            if key == 'timeout_turn':
                self.timeout_turn = val
            elif key == 'timeout_match':
                self.timeout_match = val
            elif key == 'time_left':
                self.time_left = val
        except ValueError:
            pass

    def handle_start(self, parts: list[str]) -> str:
        if len(parts) < 2:
            return "ERROR missing size"
        try:
            size = int(parts[1])
            if size != 15:
                # 理论上支持其他尺寸，但当前环境配置固定为15
                return "ERROR Unsupported board size (only 15x15)"
        except ValueError:
            return "ERROR invalid size"

        self.player.reset()
        self.env.reset()
        return "OK"

    def handle_turn(self, parts: list[str]) -> str | None:
        if len(parts) < 2:
            return "ERROR missing coordinates"
        try:
            coords = parts[1].split(',')
            x, y = int(coords[0]), int(coords[1])
            action = self.env.move2action((y, x))
            self.env.step(action)
        except (ValueError, IndexError):
            return "ERROR invalid coordinates"

        return self.get_ai_move()

    def handle_begin(self) -> str | None:
        self.player.reset()
        self.env.reset()
        return self.get_ai_move()

    def handle_board(self) -> str | None:
        self.player.reset()
        self.env.reset()

        while True:
            sub_line = self.f_reader.readline()
            if not sub_line:
                raise ConnectionError("Connection lost inside BOARD")

            sub_line = sub_line.strip()
            if sub_line.upper() == 'DONE':
                break

            try:
                b_parts = sub_line.split(',')
                if len(b_parts) >= 3:
                    bx, by, _ = int(b_parts[0]), int(b_parts[1]), int(b_parts[2])
                    self.env.step(self.env.move2action((by, bx)))
            except (ValueError, IndexError):
                logger.warning(f"Skipped bad board line: {sub_line}")

        return self.get_ai_move()

    def handle_restart(self) -> str:
        self.player.reset()
        self.env.reset()
        return "OK"

    def get_ai_move(self) -> str | None:
        # 动态调整模拟次数：简单策略，根据剩余时间或回合限制
        # 这里假设 1000 次模拟在高性能机器上约 1-2 秒
        # 实际应根据硬件测试得出系数
        if self.timeout_turn > 0:
            # 留出 200ms 的网络和系统开销缓冲
            target_time = max(100, self.timeout_turn - 200)
            # 假设每秒能跑 800 次模拟 (保守估计)
            n_sim = (target_time * 800) // 1000
            n_sim = max(100, min(n_sim, 5000))  # 限制在 100 到 5000 之间
            self.player._n_simulation = n_sim

        self.player.update_state(self.env.state, self.env.last_action, self.env.player_to_move)

        action = self.player.pending_action
        if action == -1:
            # 兜底：随机走子
            import random
            action = random.choice(self.env.get_valid_actions(self.env.state, self.env.player_to_move))

        row, col = self.env.action2move(action)
        self.env.step(action)
        self.player.pending_action = -1
        return f'{col},{row}'

    def shutdown(self) -> None:
        self.is_running = False
        self.f_reader.close()
        self.player.shutdown()


class PiskvorkTCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        addr = self.client_address
        logger.info(f"New connection from {addr}")
        session = None
        try:
            best_idx = read_best_index('Gomoku')
            # 默认初始模拟次数设为 800，实际会按INFO timeout_turn动态调整
            session = GomokuSession(self.request, model_id=best_idx, n_simulation=800, hub=hub)
            session.run()
        except Exception as e:
            logger.error(f"Error handling connection from {addr}: {e}")
        finally:
            if session:
                session.shutdown()
            logger.info(f"Connection closed for {addr}")


class PiskvorkServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True  # 允许端口复用，防止重启时报错 "Address already in use"
    daemon_threads = True       # 会话线程随主进程退出


def run_server(host='0.0.0.0', port=12345) -> None:
    global hub
    # 进程内推理引擎管理中心，多个并发会话共享引擎
    hub = ServerHub()
    with PiskvorkServer((host, port), PiskvorkTCPHandler) as server:
        logger.info(f"Piskvork AI Server running on {host}:{port}")
        logger.info("Waiting for Piskvork (Windows) to connect...")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    hub.shutdown()
    logger.info("Server shutdown.")


if __name__ == "__main__":
    """适用于Piskvork对战协议，具体协议内容见https://plastovicka.github.io/protocl2en.htm"""
    run_server()
