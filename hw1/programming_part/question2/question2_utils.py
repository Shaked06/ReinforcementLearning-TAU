
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'color':  '#004C99',
        'weight': 'normal',
        'size': 14,
        }


def hist_plot(values, title, file_name):
    plt.hist(values, color="#004C99", alpha=0.8, bins=20, density=True,
             edgecolor='white', linewidth=1.5)
    plt.title(title, fontdict=font)
    plt.xlabel('Number of Episodes', fontdict=font)
    plt.ylabel('Count', fontdict=font)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")
    plt.show()


def estimate_agent_200(env):
    observation, info = env.reset(seed=42)
    total_reward = 0
    w = np.random.uniform(-1, 1, size=4)

    for _ in range(200):
        if np.dot(observation, w) >= 0:
            action = 1
        else:
            action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"action: {action}, reward = {reward}")

        if terminated or truncated:
            break

    env.close()
    return (w, total_reward)


def estimate_agent(env):
    observation, info = env.reset(seed=42)
    total_reward = 0
    w = np.random.uniform(-1, 1, size=4)

    while True:
        if np.dot(observation, w) >= 0:
            action = 1
        else:
            action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return (w, total_reward)


def random_search(env, max_iterations):
    weights = np.zeros((max_iterations, 4))
    rewards = np.zeros(max_iterations)

    for i in range(max_iterations):
        curr_w, curr_reward = estimate_agent(env)
        weights[i, :] = curr_w
        rewards[i] = curr_reward

        if i % 1000 == 0:
            print(f"Iteration #{i}")
    return weights, rewards


def random_search_200(env, max_reward):

    curr_reward = 0
    episode_num = 0
    while curr_reward < max_reward:
        curr_w, curr_reward = estimate_agent(env)
        episode_num += 1

    return episode_num
