import pygame
from random import randint
import numpy as np
from keras.utils import to_categorical
from src.Snake import Snake
from src.Config import Config
from src.Apple import Apple

class Game:
    def __init__(self, display, rl_agent):
        self.display = display
        self.score = 0
        self.rl_agent = rl_agent
        self.counter_games = 0
        self.scores = []

    def rl_initialize_game(self, snake, apple):
        state_init1 = self.rl_agent.get_state(snake, apple)
        action = [1, 0, 0]
        # snake.rl_move(action)
        GOLEFT = pygame.USEREVENT+1 #25
        my_event = pygame.event.Event(GOLEFT, message="Go left")
        pygame.event.post(my_event)

        state_init2 = self.rl_agent.get_state(snake, apple)
        reward = self.rl_agent.set_reward(snake, snake.is_dead)
        self.rl_agent.remember(state_init1, action, reward, state_init2, snake.is_dead)
        self.rl_agent.replay_new(self.rl_agent.memory)

    def rl_loop(self, snake, apple):
        clock = pygame.time.Clock()
        self.score = 0
        x_change = 0
        y_change = 0
        snake.is_dead = False
        self.rl_initialize_game(snake, apple)
        # Perform first move
        while self.counter_games < 75:
            # This code below will be called for each frame
            # Iterate through our user input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == 25: # GO LEFT
                        x_change = -Config['snake']['speed']
                        y_change = 0
                        snake.move(x_change, y_change)
                if event.type == 26: # GO UP
                        x_change = 0
                        y_change = -Config['snake']['speed']
                        snake.move(x_change, y_change)

                if event.type == 27: # GO RIGHT
                        x_change = Config['snake']['speed']
                        y_change = 0
                        snake.move(x_change, y_change)
                if event.type == 28: # GO DOWN
                        x_change = 0
                        y_change = Config['snake']['speed']
                        snake.move(x_change, y_change)



            #get old_state
            state_old = self.rl_agent.get_state(snake, apple)

            # predict action based on the old state
            prediction = self.rl_agent.model.predict(state_old.reshape((1, 11)))
            final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            #perform new move and get new state
            snake.rl_move(final_move)

            self.display.fill(Config['colors']['green'])

            pygame.draw.rect(
                self.display,
                Config['colors']['black'],
                [
                    Config['game']['bumper_size'],
                    Config['game']['bumper_size'],
                    Config['game']['width'] - Config['game']['bumper_size']*2,
                    Config['game']['height'] - Config['game']['bumper_size']*2
                ]
            )

            apple_rect = apple.draw()
            snake_rect = snake.draw()
            snake.draw_body()


            #Eating an apple
            if apple_rect.colliderect(snake_rect):
                apple.randomize()
                snake.eat()
                self.score += 1


            state_new = self.rl_agent.get_state(snake, apple)

            # set new reward for the new state
            reward = self.rl_agent.set_reward(snake, snake.is_dead)


            #train short memory base on the new action and state
            self.rl_agent.train_short_memory(state_old, final_move, reward, state_new, snake.is_dead)

            # store the new data into a long term memory
            self.rl_agent.remember(state_old, final_move, reward, state_new, snake.is_dead)

            # Handling collision
            bumper_x = Config['game']['width'] - Config['game']['bumper_size']
            bumper_y = Config['game']['height'] - Config['game']['bumper_size']

            if (
                snake.x_pos < Config['game']['bumper_size'] or
                snake.y_pos < Config['game']['bumper_size'] or
                snake.x_pos + Config['snake']['width'] > bumper_x or
                snake.y_pos + Config['snake']['height'] > bumper_y or
                (snake.x_pos, snake.y_pos) in snake.body
            ):
                self.display = pygame.display.set_mode((Config['game']['width'], Config['game']['height']))
                pygame.display.set_caption(Config['game']['caption'])
                self.rl_agent.replay_new(self.rl_agent.memory)
                self.counter_games += 1
                self.scores.append(self.score)
                apple.randomize()
                snake.clean()
                self.rl_loop(snake, apple)


            ## Handling score and title
            pygame.font.init()
            font = pygame.font.Font('./assets/Now-Regular.otf', 28)
            score_text = 'Score: {} | Game: {}'.format(self.score, self.counter_games)
            score = font.render(score_text, True, Config['colors']['white'])
            title = font.render('Anaconda', True, Config['colors']['white'])

            title_rect = title.get_rect(
                center=(
                    Config['game']['width'] / 2,
                    Config['game']['bumper_size'] / 2
                    )
                )

            score_rect = score.get_rect(
                center=(
                    Config['game']['width'] / 2,
                    Config['game']['height'] - Config['game']['bumper_size'] / 2
                    )
                )

            self.display.blit(score, score_rect)
            self.display.blit(title, title_rect)


            pygame.display.update()

            #Slows the loop iteration rate down to the rate of the game.
            #It accepts a number of ticks per second, so we have set our game timer to 30 frames per second.
            clock.tick(Config['game']['fps'])

    def loop(self):
        clock = pygame.time.Clock()
        self.score = 0
        apple = Apple(self.display)
        snake = Snake(self.display)
        x_change = 0
        y_change = 0
        while True:
            # This code below will be called for each frame
            # Iterate through our user input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x_change = -Config['snake']['speed']
                        y_change = 0
                    elif event.key == pygame.K_RIGHT:
                        x_change = Config['snake']['speed']
                        y_change = 0
                    elif event.key == pygame.K_UP:
                        x_change = 0
                        y_change = -Config['snake']['speed']
                    elif event.key == pygame.K_DOWN:
                        x_change = 0
                        y_change = Config['snake']['speed']

            self.display.fill(Config['colors']['green'])

            pygame.draw.rect(
                self.display,
                Config['colors']['black'],
                [
                    Config['game']['bumper_size'],
                    Config['game']['bumper_size'],
                    Config['game']['width'] - Config['game']['bumper_size']*2,
                    Config['game']['height'] - Config['game']['bumper_size']*2
                ]
            )

            apple_rect = apple.draw()
            snake.move(x_change, y_change)
            snake_rect = snake.draw()
            snake.draw_body()

            #Eating an apple
            if apple_rect.colliderect(snake_rect):
                apple.randomize()
                snake.eat()
                self.score += 1

            # Handling collision
            bumper_x = Config['game']['width'] - Config['game']['bumper_size']
            bumper_y = Config['game']['height'] - Config['game']['bumper_size']

            if (
                snake.x_pos < Config['game']['bumper_size'] or
                snake.y_pos < Config['game']['bumper_size'] or
                snake.x_pos + Config['snake']['width'] > bumper_x or
                snake.y_pos + Config['snake']['height'] > bumper_y or
                (snake.x_pos, snake.y_pos) in snake.body
            ):
                new_display = pygame.display.set_mode((Config['game']['width'], Config['game']['height']))
                pygame.display.set_caption(Config['game']['caption'])
                new_game = Game(new_display, self.rl_agent)
                new_game.loop()


            ## Handling score and title
            pygame.font.init()
            font = pygame.font.Font('./assets/Now-Regular.otf', 28)
            score_text = 'Score: {}'.format(self.score)
            score = font.render(score_text, True, Config['colors']['white'])
            title = font.render('Anaconda', True, Config['colors']['white'])

            title_rect = title.get_rect(
                center=(
                    Config['game']['width'] / 2,
                    Config['game']['bumper_size'] / 2
                )
            )

            score_rect = score.get_rect(
                center=(
                    Config['game']['width'] / 2,
                    Config['game']['height'] - Config['game']['bumper_size'] / 2
                )
            )

            self.display.blit(score, score_rect)
            self.display.blit(title, title_rect)

            pygame.display.update()
            #Slows the loop iteration rate down to the rate of the game.
            #It accepts a number of ticks per second, so we have set our game timer to 30 frames per second.
            clock.tick(Config['game']['fps'])
