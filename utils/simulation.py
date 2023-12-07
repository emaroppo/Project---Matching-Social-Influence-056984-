import numpy as np
from tqdm import tqdm
from utils.metrics import compute_metrics, plot_metrics


def influence_simulation(env, models, n_episodes, n_phases=1, joint=False):
    all_metrics = []
    for model in models:
        max_ = env.opt(3)

        for i in tqdm(range(n_episodes)):
            pulled_arm = model.pull_arm()

            if n_phases == 1:
                episode, reward = env.round(pulled_arm, joint=joint)
            else:
                episode, reward, change = env.round(pulled_arm)
                if change:
                    print("change at t=", i)
                    max_ = env.opt(3)

            exp_reward = env.expected_reward(pulled_arm, 100)

            model.expected_rewards = np.append(
                model.expected_rewards, [exp_reward], axis=0
            )
            env.optimal_rewards = np.append(env.optimal_rewards, [max_[:2]], axis=0)
            model.update(episode)

        metrics = compute_metrics(model, env)
        all_metrics.append(metrics)
        env.optimal_rewards = np.empty((0, 2))  # temporary fix

    plot_metrics(all_metrics, env_name="Social Environment")

    return all_metrics


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

        products = product_classes * products_per_class
        for i in tqdm(range(n_episodes)):
            episode_active_nodes = (
                np.argwhere(active_nodes[i]) if active_nodes else None
            )
            customers = class_mapping[episode_active_nodes]

            # Pull arm
            pulled_arm = model.pull_arm(episode_active_nodes, customers)
            # Retrieve reward
            reward = env.round(pulled_arm)
            reward = [i[1] for i in reward]
            opt = env.opt(customers, products)
            print("Regret: ", opt - np.sum(reward))
            # Update bandit
            model.update(pulled_arm, reward)
            opts.append(opt)
            collected_rewards.append(np.sum(reward))

            # Metrics collection (similar to influence_simulation)
            exp_reward = env.expected_reward(pulled_arm)
            model.expected_rewards = np.append(model.expected_rewards, [exp_reward], axis=0)
            env.optimal_rewards = np.append(env.optimal_rewards, [opt], axis=0)  # Assuming opt gives the optimal reward for the round

        # Compute and store metrics for the current model
        metrics = compute_metrics(model, env)
        all_metrics.append(metrics)
        env.optimal_rewards = np.empty((0,))  # Reset for next model

    # Plot metrics for all models
    plot_metrics(all_metrics, env_name="Matching Environment")

    return all_metrics



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
    (
        all_mean_rewards,
        all_optimal_rewards,
        all_models_active_nodes,
    ) = influence_simulation(
        env.social_environment, influence_models, n_episodes, n_phases, joint=True
    )

    all_collected_rewards = []
    all_opts = []

    # For each matching model, use the corresponding active nodes from the influence simulation
    for idx, matching_model in enumerate(matching_models):
        active_nodes_for_model = all_models_active_nodes[idx]
        collected_rewards, opts = matching_simulation(
            env.matching_environment,
            [matching_model],
            n_episodes,
            active_nodes=active_nodes_for_model,
            class_mapping=class_mapping,
            product_classes=product_classes,
            products_per_class=products_per_class,
        )
        all_collected_rewards.append(collected_rewards)
        all_opts.append(opts)

    return (all_mean_rewards, all_optimal_rewards), (all_collected_rewards, all_opts)


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
