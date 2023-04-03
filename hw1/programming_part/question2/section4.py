import numpy as np
import gym
from gym.utils.play import play
from gym.spaces import Box, Discrete
from question2_utils import *


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    max_iterations = 10000
    weights, rewards = random_search(env, max_iterations)

    print("\n------ DONE ------ \n")

    best_reward = np.max(rewards)
    best_w = weights[np.argmax(rewards), :]
    print(f"Total Reward = {best_reward}   \nWeights = {best_w}")
