
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'color':  '#004C99',
        'weight': 'normal',
        'size': 14,
        }


def hist_plot(values, title, file_name):
    plt.hist(values, color="#004C99", alpha=0.8, bins=20,
             edgecolor='white', linewidth=1.5)
    plt.title(title, fontdict=font)
    plt.xlabel('Number of Episodes', fontdict=font)
    plt.ylabel('Count', fontdict=font)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")
    # plt.show()


def estimate_agent(env):
    """
        estimate the agent over an episode and return its weights and rewards
        episode: until the agent is terminated or reaches to rewards of 200
    """

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

        if terminated or truncated:
            break

    return w, total_reward


def random_search(env, max_iterations):
    """
        calculate the number of episodes it takes to reach to rewards of 200
        using random weights and return the number of episodes, the weights and the total rewards

        episode: until the agent is terminated or reaches to rewards of 200
    """
    curr_reward = 0
    best_reward = 0
    best_w = np.zeros(4)
    for i in range(max_iterations):
        curr_w, curr_reward = estimate_agent(env)
        if curr_reward > best_reward:
            best_w, best_reward = curr_w, curr_reward

        if best_reward == 200:
            return best_w, best_reward, i

    return best_w, best_reward, max_iterations
