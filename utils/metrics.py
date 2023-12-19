import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import time
import pickle
from typing import List, Dict
import scipy.stats as stats


def compute_metrics(model, env, to_file=True):
    metrics = dict()
    metrics["model_name"] = model.name
    metrics["instantaneous_reward"] = model.expected_rewards
    metrics["optimal_reward"] = env.optimal_rewards

    # calculate instantaneous regret expected value (var[i][0]) and std (var[i][1])
    expected_inst_regret = env.optimal_rewards[:, 0] - model.expected_rewards[:, 0]
    # min regret is 0
    expected_inst_regret[expected_inst_regret < 0] = 0
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

    # save influence probability / expected rewards estimates
    if "UCB" in metrics["model_name"]:
        metrics["estimates"] = model.empirical_means + model.confidence

    if "TS" in metrics["model_name"]:
        if "Prob" in metrics["model_name"]:
            probs = model.beta_parameters[:, :, 0] / (
                model.beta_parameters[:, :, 0] + model.beta_parameters[:, :, 1]
            )
            if model.graph_structure is not None:
                metrics["estimates"] = probs * model.graph_structure

        elif "Matching" in metrics["model_name"]:
            metrics["estimates"] = model.mu

    if "Matching" in env.__class__.__name__:
        metrics["real_values"] = env.reward_parameters
    elif "Social" in env.__class__.__name__:
        metrics["real_values"] = env.probabilities

    if to_file:
        with open(f"metrics/{metrics['model_name']}_{time.time()}.pkl", "wb") as f:
            pickle.dump(metrics, f)
    return metrics


def plot_metrics(metrics_list: List[Dict], env_name=None, show=False):
    # Create a new directory for saving the plots
    folder_name = f"plots/run_{int(time.time())}"
    os.makedirs(folder_name, exist_ok=True)

    # Determine the set of all metric names across all models
    metric_names = set()
    for metrics in metrics_list:
        metric_names.update(metrics.keys())

    # Plot each metric
    for metric_name in metric_names:
        if metric_name in ["model_name", "optimal_reward", "real_values", "estimates"]:
            continue
        plt.figure()

        for model_metrics in metrics_list:
            if metric_name in model_metrics:
                # Extract the expected value and standard deviation
                expected_values = model_metrics[metric_name][:, 0]
                std_devs = model_metrics[metric_name][:, 1]
                timesteps = range(len(expected_values))
                # extract optimal rewards

                if metric_name == "instantaneous_reward":
                    optimal_rewards = model_metrics["optimal_reward"][:, 0]
                    plt.plot(
                        timesteps,
                        optimal_rewards,
                        label="Optimal Reward",
                        color="black",
                        linestyle="dashed",
                    )
                # plot optimal reward only if it is instantaneous reward

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
        if show:
            plt.show()
        plt.close()


def plot_network(
    matrix, labels, filename="plots/network_probs.png", dpi=300, show=False
):
    # Validate that the number of labels matches the size of the matrix
    filename = f"plots/network_probs_{time.time()}.png"
    if len(labels) != len(matrix):
        raise ValueError("The number of labels must match the number of nodes.")

    # Create a graph
    G = nx.DiGraph()

    # Add nodes with labels
    for i, label in enumerate(labels):
        G.add_node(i, label=label)

    # Add edges with weights
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=round(matrix[i][j], 3))

    # Position nodes using the Kamada-Kawai layout for better spacing
    pos = nx.kamada_kawai_layout(G)

    # Draw the nodes, smaller size
    nx.draw_networkx_nodes(G, pos, node_size=300)

    # Draw the edges
    nx.draw_networkx_edges(G, pos)

    # Label nodes with custom labels
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Save the plot with high resolution
    plt.savefig(filename, dpi=dpi)

    # Show the plot
    if show:
        plt.show()
    plt.close()


def plot_heatmap(array, title=None, use_log_scale=False, show=False):
    if use_log_scale:
        # Apply logarithmic scaling (adding a small value to avoid log(0))
        data = np.log(array + np.abs(array.min()) + 1e-5)
    else:
        # Use percentile-based scaling
        vmin, vmax = np.percentile(array, [5, 95])
        data = np.clip(array, vmin, vmax)
    if title is not None:
        plt.title(title)
    plt.imshow(data, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.savefig(f"plots/heatmap_{time.time()}.png")
    if show:
        plt.show()
    plt.close()


def plot_influence_probabilities(env, learner):
    real_values = env.probabilities

    real_values = env.probabilities
    if "UCB" in learner.name:
        estimates = learner.empirical_means + learner.confidence
        if learner.graph_structure is not None:
            estimates = estimates * learner.graph_structure
    elif "TS" in learner.name:
        estimates = learner.beta_parameters[:, :, 0] / (
            learner.beta_parameters[:, :, 0] + learner.beta_parameters[:, :, 1]
        )
        if learner.graph_structure is not None:
            estimates = estimates * learner.graph_structure

    elif "EXP3" in learner.name:
        estimates = learner.get_probabilities()
        if learner.graph_structure is not None:
            estimates = estimates * learner.graph_structure

    diff = estimates - real_values
    plot_heatmap(real_values, title="Real Probabilities")
    plot_heatmap(estimates, title="Estimated Probabilities")
    if "UCB" in learner.name:
        plot_heatmap(learner.empirical_means, title="Empirical Means")
        plot_heatmap(
            learner.empirical_means - real_values, title="Delta (Empirical Means)"
        )
    plot_heatmap(diff, title="Delta (Estimated Probabilities)")


def plot_matching_rewards(env, learner):
    real_values = env.reward_parameters[:, :, 0]

    if "UCB" in learner.name:
        estimates = learner.empirical_means + learner.confidence
        estimates = estimates[:3, :3]
    elif "TS" in learner.name:
        estimates = learner.mu
        estimates = estimates[:3, :3]

    diff = estimates - real_values
    plot_heatmap(real_values, title="Real Rewards")
    plot_heatmap(estimates, title="Estimated Rewards")
    plot_heatmap(diff, title="Delta (Estimated Rewards)")


def plot_reward_distributions(reward_params, show=False):
    """
    Plots the distribution of expected rewards for customer-product pairings.

    :param reward_params: A tuple where the first element is a 3x3 numpy array of means and the
                          second element is a 3x3 numpy array of standard deviations.
    """
    means, std_devs = reward_params
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            mean, std_dev = means[i, j], std_devs[i, j]
            x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)
            y = stats.norm.pdf(x, mean, std_dev)

            ax = axes[i, j]
            ax.plot(x, y, color="blue")
            ax.set_title(f"Customer Class {i+1} - Product Class {j+1}")
            ax.set_xlabel("Reward")
            ax.set_ylabel("Probability Density")

    plt.tight_layout()
    plt.savefig(f"plots/reward_distributions_{time.time()}.png")
    if show:
        plt.show()
    plt.close()
    return fig
