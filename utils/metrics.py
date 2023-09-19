import numpy as np
import matplotlib.pyplot as plt
import os
import time


def compute_metrics(rewards, opt_rewards):
    # if array is not 1d or a list take mean along axis 0
    if type(rewards) is not list and len(rewards.shape) > 1:
        rewards = rewards.mean(axis=0)

    instantaneous_regret = opt_rewards - rewards
    cumulative_regret = np.cumsum(instantaneous_regret)
    rewards_trend = np.polyfit(np.arange(len(rewards)), rewards, 1)

    return rewards, opt_rewards, instantaneous_regret, cumulative_regret, rewards_trend


def plot_metrics(
    all_rewards,
    all_opt_rewards,
    all_instantaneous_regrets,
    all_cumulative_regrets,
    model_names=None,
    env_name=None,
):
    # Create a new directory for saving the plots
    folder_name = f"run_{int(time.time())}"
    os.makedirs(folder_name, exist_ok=True)

    num_models = len(all_rewards)

    # Plot for Expected Rewards
    plt.figure()
    for i in range(num_models):
        plt.plot(
            all_rewards[i], label=model_names[i] if model_names else f"Model_{i+1}"
        )
    plt.plot(
        all_opt_rewards[0], label="optimal", linestyle="--"
    )  # Assuming optimal is same for all models
    plt.xlabel("t")
    plt.ylabel("expected reward")
    plt.legend()
    plt.title(env_name if env_name else "Expected Rewards Comparison")
    plt.savefig(f"{folder_name}/comparison_expected_reward.png")
    plt.show()

    # Plot for Instantaneous Regret
    plt.figure()
    for i in range(num_models):
        plt.plot(
            all_instantaneous_regrets[i],
            label=model_names[i] if model_names else f"Model_{i+1}",
        )
    plt.xlabel("t")
    plt.ylabel("instantaneous regret")
    plt.legend()
    plt.title(env_name if env_name else "Instantaneous Regret Comparison")
    plt.savefig(f"{folder_name}/comparison_instantaneous_regret.png")
    plt.show()

    # Plot for Cumulative Regret
    plt.figure()
    for i in range(num_models):
        plt.plot(
            all_cumulative_regrets[i],
            label=model_names[i] if model_names else f"Model_{i+1}",
        )
    plt.xlabel("t")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.title(env_name if env_name else "Cumulative Regret Comparison")
    plt.savefig(f"{folder_name}/comparison_cumulative_regret.png")
    plt.show()
