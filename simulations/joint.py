import numpy as np
from tqdm import tqdm
from utils.metrics import plot_metrics
from simulations.influence import influence_simulation
from simulations.matching import matching_simulation


def joint_simulation(
    env,
    influence_models,
    matching_models,
    n_episodes,
    n_phases=1,
    class_mapping=None,
    product_classes=[0, 1, 2],
    products_per_class=3,
):
    # Run the influence simulation
    influence_metrics, all_active_nodes = influence_simulation(
        env.social_environment, influence_models, n_episodes, n_phases, joint=True
    )

    # Store metrics for matching models
    matching_metrics = []

    # For each matching model, use the corresponding active nodes from the influence simulation
    for model, active_nodes in zip(matching_models, all_active_nodes):
        # Run the matching simulation
        metrics = matching_simulation(
            env.matching_environment,
            [model],
            n_episodes,
            active_nodes=active_nodes,
            class_mapping=class_mapping,
            product_classes=product_classes,
            products_per_class=products_per_class,
        )

        # Compute and store metrics for the current matching model
        matching_metrics.append(metrics[0])

    # Optionally, you can plot the combined metrics of influence and matching models
    plot_metrics(matching_metrics, env_name="Joint Environment")

    return metrics, influence_metrics


def merged_matching_simulation_with_features(
    env,
    model,
    n_episodes,
    node_features,  # New argument to pass customer features
    use_context_updates=False,  # Argument to decide which logic to run
    context_generator=None,
    initial_class=0,  # Default class for all customers at the start
    active_nodes=None,
    class_mapping=None,
    product_classes=[0, 1, 2],
    products_per_class=3,
    context_update_frequency=14,  # Update context every 14 days
):
    all_collected_rewards = []
    all_optimal_rewards = []
    products = product_classes * products_per_class
    for i in tqdm(range(n_episodes)):
        # Get the active nodes (customers) for this episode
        episode_active_nodes = np.squeeze(active_nodes)[i] if active_nodes else None

        # Access the features of the active customers
        active_node_features = node_features[episode_active_nodes]

        # Logic for adapted simulation with context updates
        if use_context_updates:
            # Update context every context_update_frequency days
            if i % context_update_frequency == 0:
                # If it's the first episode, assume all customers belong to the initial class
                if i == 0:
                    customers = [initial_class] * len(episode_active_nodes)
                else:
                    # Update context using the context generator and the active node features
                    updated_context = context_generator.update()
                    customers = class_mapping[updated_context]

                # TODO: Update the model arms based on the new context
        else:
            # Original logic without context updates
            customers = class_mapping[episode_active_nodes]

        # Common logic for pulling arm, retrieving reward, and updating bandit
        pulled_arm = model.pull_arm(episode_active_nodes, customers)
        reward = env.round(pulled_arm)
        # reward = [i[1] for i in reward]
        opt = env.opt(customers, products)
        model.update(pulled_arm, reward)
        all_optimal_rewards.append(opt)
        matching_rewards = [matching_reward for _, matching_reward in reward]
        all_collected_rewards.append(np.sum(matching_rewards))

    return np.array(all_collected_rewards), np.array(all_optimal_rewards)


# The function is modified to access customer features during the simulation.
