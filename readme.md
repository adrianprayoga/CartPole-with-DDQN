# Gym (CartPole) Exploration with Double Deep Q Network

In this notebook, we use Deep Q Learning and Double Deep Q Learning and Prioritized experienced replay to tackle openAI gym.
We will be utilizing Keras to build our Deep Q Learning and will try to solve CartPole provided by gym. Though only CartPole is used in the project, it can be easily extended to other gym environment as well such as MountainCar, or even Atari.

## Getting Started

This repo consists of 2 main files:
- Jupyter Notebook (CartPole with Deep Q Learning.ipynb)
- Python file (Load_Execute_Model.py)

The model is built in the jupyter notebook and the learned model can be loaded again with the provided python file. You can change the environment that you'd like to explore by changing the PROBLEM variable from the notebook.

### Prerequisites

- Keras
- OpenAI Gym
- Numpy

## Acknowledgments

* Jaromír Janisch's tutorial on reinforcement learning
* Hado van Hasselt, et al for the great DDQN paper
