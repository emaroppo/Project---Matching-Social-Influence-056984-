from tqdm import tqdm
import numpy as np
from utils.metrics import compute_metrics, plot_metrics


def matching_simulation(
    env,
    models,
    n_episodes,
    active_nodes=None,
    class_mapping=None,
    product_classes=[0, 1, 2],
    products_per_class=3,
):
    all_metrics = []  # To store metrics for all models
    for model in models:
        collected_rewards = []
        opts = []
        env.optimal_rewards = np.empty((0, 2))  # Reset for each model

        products = product_classes * products_per_class
        for i in tqdm(range(n_episodes)):
            episode_active_nodes = (
                np.argwhere(active_nodes[i]) if active_nodes else None
            )
            customers = class_mapping[episode_active_nodes]

            # Pull arm
            pulled_arm = model.pull_arm(customers)
            # Retrieve reward
            reward = env.round(pulled_arm)
            reward = [i[1] for i in reward]
            opt = env.opt(customers, products)
            print("Regret: ", opt[0][0] - np.sum(reward))
            # Update bandit
            model.update(pulled_arm, reward)
            opts.append(opt)
            collected_rewards.append(np.sum(reward))

            # Metrics collection (similar to influence_simulation)
            exp_reward = env.expected_reward(pulled_arm)
            model.expected_rewards = np.append(
                model.expected_rewards, [exp_reward], axis=0
            )
            env.optimal_rewards = np.concatenate(
                (env.optimal_rewards, env.opt(customers, product_classes)),
                axis=0,
            )  # Assuming opt gives the optimal reward for the round

        # Compute and store metrics for the current model
        metrics = compute_metrics(model, env)
        all_metrics.append(metrics)
        env.optimal_rewards = np.empty((0,))  # Reset for next model

    # Plot metrics for all models
    plot_metrics(all_metrics, env_name="Matching Environment")

    return all_metrics
