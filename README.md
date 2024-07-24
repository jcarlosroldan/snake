# Snake

This simple repo trains a snake game using a deep reinforcement learning. All other repos I saw were either using feature engineering (i.e. passing variables such as distance to the food or whether it was to the left or to the right), we not successful on larger boards, and most were very complicated.

## How does it work

When you run the `run.py` file, it will:

1. Create an agent that is just a bunch of fully connected layers, GELU activations and normalization, by default with 1.3M parameters.
1. Initialize an environment of 15x15 cells, where everything is a wall except for the 5x5 center, where the snake can move and the food can appear.
1. Train the agent using reinforcement learning with a simple DQN algorithm, where the agent will try to maximize the reward of eating the food and minimize the penalty of dying.
1. Iterate over the previous two steps with increasing empty space in the board, until the agent is able to solve the full board.

Everything is defined at using pytorch with no other dependencies.

## What's it useful for

Some ideas of things you can do with this repo:
* Use it as a starting point to learn about reinforcement learning.
* Just run it to generate a perfect snake AI.
* Adjust the hyperparameters (hardcoded at `snake/train.py:run`), specially reducing the agent size. With just 5 hidden layers of 27 neurons each, the network is still able to play perfect games up to 7x7 boards (and that's under 10K parameters).
* Adapt it to other games board games such as Pacman or Sokoban.
* Plot hidden layer activations to understand how the agent thinks.