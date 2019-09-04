# AI bot learns how to play Snake using Reinforcement Learning
The goal of this project is to develop an AI Bot able to learn how to play the Snake game from scratch. 
* Implemented the Snake game from scratch using pygame library
* Implemented a Deep Reinforcement Learning algorithm using Keras
* Visualized  how the Deep Q-Learning algorithm learns how to play snake, scoring up to 20 points and showing a solid strategy after only 10 minutes of training.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites 
You will need `python3` installed and `pygame`, `Keras` librairies installed. 

### Running
To play the game run the following command 
```
python3 SnakeGame.py
```
To see the AI bot learning how to play the game, run the following command
```
python3 SnakeGame.py -rl
```
The deep neural network can be customized within the `network` function of the agent located in the file `DeepQNetwork.py`.

