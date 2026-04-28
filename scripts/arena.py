import threading

from utils.arena import Arena
from utils.config import game_name

if __name__ == '__main__':
    arena = Arena(game_name)
    # thread = threading.Thread(target=arena.scheduled_show, args=(30,), daemon=True)
    # thread.start()
    # arena.run_gauntlet()
    arena.show_rank(100,True)
    # arena.show_rank(10,False)
