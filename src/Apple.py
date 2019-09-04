import random
import pygame
from src.Config import Config

class Apple:
    def __init__(self, display):
        self.x_pos = 0
        self.y_pos = 0

        self.display = display

        self.randomize()

    def randomize(self):
        height = Config['game']['height']
        width = Config['game']['width']
        bumper = Config['game']['bumper_size']

        max_x = (width - 2*bumper - Config['snake']['width'])
        max_y = (height - 2*bumper - Config['snake']['height'])

        self.x_pos = random.randint(bumper*2, max_x)
        self.y_pos = random.randint(bumper*2, max_y)

    def draw(self):
        return pygame.draw.rect(
            self.display,
            Config['colors']['red'],
            [
                self.x_pos,
                self.y_pos,
                Config['apple']['height'],
                Config['apple']['width']
            ]
        )
