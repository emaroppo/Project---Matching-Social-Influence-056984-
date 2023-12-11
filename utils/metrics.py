import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Dict


def compute_metrics(model, env):
    metrics = dict()
    metrics["model_name"] = model.__class__.__name__
    metrics["instantaneous_reward"] = model.expected_rewards

    # calculate instantaneous regret expected value (var[i][0]) and std (var[i][1])
    expected_inst_regret = env.optimal_rewards[:, 0] - model.expected_rewards[:, 0]
    std_inst_regret = np.sqrt(
        env.optimal_rewards[:, 1] ** 2 + model.expected_rewards[:, 1] ** 2
    )
    metrics["instantaneous_regret"] = np.array(
        [expected_inst_regret, std_inst_regret]
    ).T  # why do i have to Transpose?

    # calculate cumulative reward expected value (var[i][0]) and std (var[i][1]) (cumsum of expected rewards and std of rewards)
    expected_cumulative_reward = np.cumsum(model.expected_rewards[:, 0])
    std_cumulative_reward = np.sqrt(np.cumsum(model.expected_rewards[:, 1] ** 2))
    metrics["cumulative_reward"] = np.array(
        [expected_cumulative_reward, std_cumulative_reward]
    ).T

    # calculate cumulative regret expected value (var[i][0]) and std (var[i][1]) (cumsum of expected regret and std of regret)
    expected_cumulative_regret = np.cumsum(expected_inst_regret)
    std_cumulative_regret = np.sqrt(np.cumsum(std_inst_regret**2))
    metrics["cumulative_regret"] = np.array(
        [expected_cumulative_regret, std_cumulative_regret]
    ).T

    return metrics


def plot_metrics(metrics_list: List[Dict], env_name=None):
    # Create a new directory for saving the plots
    folder_name = f"plots/run_{int(time.time())}"
    os.makedirs(folder_name, exist_ok=True)

    # Determine the set of all metric names across all models
    metric_names = set()
    for metrics in metrics_list:
        metric_names.update(metrics.keys())

    # Plot each metric
    for metric_name in metric_names:
        if metric_name == "model_name":
            continue
        plt.figure()
        
        for model_metrics in metrics_list:
            if metric_name in model_metrics:
                # Extract the expected value and standard deviation
                expected_values = model_metrics[metric_name][:, 0]
                std_devs = model_metrics[metric_name][:, 1]
                timesteps = range(len(expected_values))

                # Plot the expected values
                plt.plot(
                    timesteps,
                    expected_values,
                    label=model_metrics.get("model_name", "Unknown Model"),
                )

                # Add a colored band for the standard deviation
                plt.fill_between(
                    timesteps,
                    expected_values - std_devs,
                    expected_values + std_devs,
                    alpha=0.2,
                )

        plt.title(
            f'{ " ".join(metric_name.split("_")).title()} over Time'
            + ("" if env_name is None else f" in {env_name}")
        )
        plt.xlabel("Time Step")
        plt.ylabel(" ".join(metric_name.split("_")).title())
        plt.legend()
        plt.savefig(f"{folder_name}/{metric_name}.png")
        plt.show()
        plt.close()
