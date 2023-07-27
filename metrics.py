import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(rewards, opt_rewards):
    # if array is not 1d or a list take mean along axis 0
    if type(rewards) is not list and len(rewards.shape) > 1:
        rewards = rewards.mean(axis=0)

    instantaneous_regret = opt_rewards - rewards
    cumulative_regret = np.cumsum(instantaneous_regret)
    rewards_trend = np.polyfit(np.arange(len(rewards)), rewards, 1)

    return rewards, opt_rewards, instantaneous_regret, cumulative_regret, rewards_trend


def plot_metrics(
    rewards, opt_rewards, instantaneous_regret, cumulative_regret, rewards_trend=None
):
    plt.plot(rewards)
    if rewards_trend is not None:
        plt.plot(
            np.arange(len(rewards)), np.polyval(rewards_trend, np.arange(len(rewards)))
        )
    plt.plot(opt_rewards)
    plt.show()
    plt.plot(instantaneous_regret)
    plt.show()
    plt.plot(cumulative_regret)
    plt.show()
