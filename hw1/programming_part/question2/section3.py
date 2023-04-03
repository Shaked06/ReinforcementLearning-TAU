import numpy as np
import gym
from gym.utils.play import play
from gym.spaces import Box, Discrete
from question2_utils import *

if __name__ == "__main__":
    env = gym.make("CartPole-v0", render_mode="human")
    w, total_reward = estimate_agent_200(env)


env.close()

print(f"Total Reward = {total_reward}")
