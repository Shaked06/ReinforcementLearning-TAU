import numpy as np
import gym
from gym.utils.play import play
from gym.spaces import Box, Discrete
from question2_utils import *


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    max_iterations = 10000
    best_w, best_reward, _ = random_search(env, max_iterations)

    env.close()
    print(f"Total Reward = {best_reward}   \nWeights = {best_w}")
