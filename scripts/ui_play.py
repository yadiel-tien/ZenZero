import sys

from utils.types import EnvName
import pygame

from ui.chess import ChineseChessUI
from ui.gomoku import GomokuUI
from utils.config import settings, CONFIG
from player.human import Human
from player.ai_client import AIClient

game_name: EnvName = CONFIG['game_name']


class Game:
    def __init__(self, model1_idx: int, model2_idx: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(settings['screen_size'])
        pygame.display.set_caption(game_name)
        if model1_idx == -1:
            self.players = [Human(game_name), Human(game_name)]
        elif model2_idx == -1:
            self.players = [Human(game_name), AIClient(model1_idx, game_name)]
        else:
            self.players = [AIClient(model1_idx, game_name), AIClient(model2_idx, game_name)]
        if game_name == 'Gomoku':
            self.board = GomokuUI(self.players)
        else:
            self.board = ChineseChessUI(players=self.players)

    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.board.handle_input(event)

            self.board.update()
            self.board.draw()
            pygame.display.update()

    def shutdown(self):
        for player in self.board.players:
            if hasattr(player, 'shutdown'):
                player.shutdown()


if __name__ == '__main__':
    # 解析参数，如果参数个数设置模式为人人，人机，机机对战。
    index1, index2 = 97, -1
    if len(sys.argv) > 1:
        index1 = int(sys.argv[1])
    if len(sys.argv) > 2:
        index2 = int(sys.argv[2])

    game = Game(index1, index2)
    game.play()
    game.shutdown()
    pygame.quit()
    sys.exit()
