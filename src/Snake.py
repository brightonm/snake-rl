import pygame
import numpy as np

from src.Config import Config

class Snake:
    def __init__(self, display):
        self.x_pos = Config['game']['width'] / 2
        self.y_pos = Config['game']['height'] / 2
        self.display = display
        self.body = []
        self.max_size = 0
        # Useful for the DQN agent
        self.x_change = 0
        self.y_change = 0
        self.is_dead = False
        self.eaten = False

    def draw(self):
        # This function will be called for each frame from our loop function in the game class
        return pygame.draw.rect(
            self.display,
            Config['colors']['green'],
            [
                self.x_pos,
                self.y_pos,
                Config['snake']['width'],
                Config['snake']['height']
            ]
        )

    def move(self, x_change, y_change):
        self.body.append((self.x_pos, self.y_pos))
        self.x_pos += x_change
        self.y_pos += y_change
        self.x_change = x_change
        self.y_change = y_change

        if len(self.body) > self.max_size:
            del(self.body[0])

        # Handling collision
        bumper_x = Config['game']['width'] - Config['game']['bumper_size']
        bumper_y = Config['game']['height'] - Config['game']['bumper_size']


        if (
            self.x_pos < Config['game']['bumper_size'] or
            self.y_pos < Config['game']['bumper_size'] or
            self.x_pos + Config['snake']['width'] > bumper_x or
            self.y_pos + Config['snake']['height'] > bumper_y or
            (self.x_pos, self.y_pos) in self.body
        ) :
            self.is_dead = True

    def rl_move(self, action):
        if self.eaten :
            self.eaten = False
        GOLEFT = pygame.USEREVENT+1 #25
        go_left = pygame.event.Event(GOLEFT, message="Go left")
        GOUP = pygame.USEREVENT+2 #26
        go_up = pygame.event.Event(GOUP, message="Go up")
        GORIGHT = pygame.USEREVENT+3 #27
        go_right = pygame.event.Event(GORIGHT, message="Go right")
        GODOWN = pygame.USEREVENT+4 #28
        go_down = pygame.event.Event(GODOWN, message="Go down")

        speed = Config['snake']['speed']
        prev_pos = [self.x_change, self.y_change]

        if np.array_equal(action, [1, 0, 0]):
            # keep going in the same direction
            if prev_pos == [-speed, 0]:
                pygame.event.post(go_left)
            elif prev_pos == [speed, 0]:
                pygame.event.post(go_right)
            elif prev_pos == [0, speed]:
                pygame.event.post(go_down)
            elif prev_pos == [0, -speed]:
                pygame.event.post(go_up)

        elif np.array_equal(action, [0, 1, 0]):
            # turn right
            if prev_pos == [-speed, 0]:
                pygame.event.post(go_up)
            elif prev_pos == [speed, 0]:
                pygame.event.post(go_down)
            elif prev_pos == [0, speed]:
                pygame.event.post(go_left)
            elif prev_pos == [0, -speed]:
                pygame.event.post(go_right)

        elif np.array_equal(action, [0, 0, 1]):
            # turn left
            if prev_pos == [-speed, 0]:
                pygame.event.post(go_down)
            elif prev_pos == [speed, 0]:
                pygame.event.post(go_up)
            elif prev_pos == [0, speed]:
                pygame.event.post(go_right)
            elif prev_pos == [0, -speed]:
                pygame.event.post(go_left)

    def eat(self):
        self.max_size += 1
        self.eaten = True

    def clean(self):
        self.x_pos = Config['game']['width'] / 2
        self.y_pos = Config['game']['height'] / 2
        self.body = []
        self.max_size = 0
        # Useful for the DQN agent
        self.x_change = 0
        self.y_change = 0


    def draw_body(self):
        for item in self.body:
            pygame.draw.rect(
                self.display,
                Config['colors']['green'],
                [
                    item[0],
                    item[1],
                    Config['snake']['width'],
                    Config['snake']['height']
                ]
            )
