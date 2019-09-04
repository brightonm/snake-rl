import pygame
import sys
from DeepQNetwork import DQNAgent
from src.Config import Config
from src.Game import Game
from src.Apple import Apple
from src.Snake import Snake

def main():
    display = pygame.display.set_mode((Config['game']['width'], Config['game']['height']))
    pygame.display.set_caption(Config['game']['caption'])
    rl_agent = None
    if len(sys.argv) > 1 and sys.argv[1] == "-rl":
        rl_agent = DQNAgent()
        game = Game(display, rl_agent)
        apple = Apple(game.display)
        snake = Snake(game.display)
        game.rl_loop(snake, apple)
        print(game.scores)
    else:
        game = Game(display, rl_agent)
        game.loop()


main()
