import argparse
import sys
from core.env.functions import get_class
from core.network.functions import read_best_index, list_all_indices
from core.player.human import Human
from core.player.ai_server import AIServer
from core.utils.config import CONFIG
from core.utils.logger import get_logger

def select_game():
    """交互式选择游戏"""
    games = ["ChineseChess", "Gomoku"]
    print("\n=== ZenZero CLI Battle ===")
    for i, g in enumerate(games):
        print(f"{i+1}. {g}")
    
    try:
        choice = int(input("\nSelect game (number): ")) - 1
        if 0 <= choice < len(games):
            return games[choice]
    except ValueError:
        pass
    print("Invalid choice, defaulting to ChineseChess")
    return "ChineseChess"

def select_model(game_name):
    """交互式选择模型"""
    indices = list_all_indices(game_name)
    best_idx = read_best_index(game_name)
    
    print(f"\nAvailable models for {game_name}:")
    print(f"0. Strongest AI (v{best_idx})")
    for i, idx in enumerate(indices):
        print(f"{i+1}. AI Version {idx}")
    
    try:
        choice = int(input("\nSelect model (number): "))
        if choice == 0:
            return best_idx
        if 1 <= choice <= len(indices):
            return indices[choice-1]
    except ValueError:
        pass
    return best_idx

if __name__ == "__main__":
    logger = get_logger('cli_play')
    
    parser = argparse.ArgumentParser(description="ZenZero CLI Play")
    parser.add_argument('--game', type=str, help="Game name (ChineseChess/Gomoku)")
    parser.add_argument('--model', type=int, help="Model iteration index")
    parser.add_argument('--sim', type=int, default=800, help="MCTS simulation count")
    args = parser.parse_args()

    # 1. 确定游戏
    game_name = args.game if args.game else select_game()
    
    # 2. 确定模型
    model_idx = args.model if args.model is not None else select_model(game_name)
    
    logger.info(f"Starting {game_name} with Model {model_idx} (Simulations: {args.sim})")
    
    try:
        env = get_class(game_name)()
        # AIServer 会自动尝试连接到已启动的 play_server/hub
        with AIServer(game_name, model_idx, n_simulation=args.sim) as ai:
            # 命令行交互循环
            # Human(game_name) 会处理终端输入
            result = env.run((Human(game_name), ai))
        
        env.render()
        print(f"\nGame Over! Winner: {result}")
        
    except ConnectionError:
        logger.error("Failed to connect to Play Server. Please run 'python -m scripts.play_server' first!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
