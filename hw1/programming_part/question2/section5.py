import numpy as np
import gym
from gym.utils.play import play
from gym.spaces import Box, Discrete
from question2_utils import *


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    max_iterations = 1000
    num_episode_arr = np.zeros(max_iterations)

    for i in range(max_iterations):
        num_episode_arr[i] = random_search_200(env, max_reward=200)
        if i % 100 == 0:
            print(f"Iteration #{i}")

    print("\n------ DONE ------ \n")

    hist_plot(num_episode_arr, title=f"Histogram of Random Search \n Mean of episodes {np.mean(num_episode_arr)}",
              file_name="histogram-episodes-number")