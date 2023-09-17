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
    rewards, opt_rewards, instantaneous_regret, cumulative_regret, rewards_trend=None, model_name=None, env_name=None
):
    if env_name is not None:
        plt.title(env_name)

    if model_name is not None:
        plt.plot(rewards, label=model_name)
    else:
        plt.plot(rewards)

    if rewards_trend is not None:
        plt.plot(
            np.arange(len(rewards)), np.polyval(rewards_trend, np.arange(len(rewards))), label="trend"
        )
    plt.plot(opt_rewards, label="optimal")
    plt.xlabel("t")
    plt.ylabel("expected reward")
    plt.legend()
    plt.show()

    if env_name is not None:
        plt.title(env_name)
        
    plt.plot(instantaneous_regret, label="instantaneous regret")
    plt.xlabel("t")
    plt.ylabel("instantaneous regret")
    plt.legend()
    plt.show()

    if env_name is not None:
        plt.title(env_name)

    plt.plot(cumulative_regret, label="cumulative regret")
    plt.xlabel("t")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.show()
