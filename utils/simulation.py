import numpy as np
from tqdm import tqdm


def influence_simulation(env, models, n_episodes, n_phases=1, joint=False):
    all_mean_rewards = []
    all_optimal_rewards = []
    all_models_active_nodes = []  # List of lists to store active nodes for each model

    for model in models:
        mean_rewards = []
        optimal_reward = []
        active_nodes = []  # List to store active nodes for this model

        max_ = env.opt(3)

        for i in tqdm(range(n_episodes)):
            pulled_arm = model.pull_arm()

            if n_phases == 1:
                episode, reward = env.round(pulled_arm, joint=joint)
                exp_reward = env.expected_reward(pulled_arm, 100)[0]
                mean_rewards.append(exp_reward)
                optimal_reward.append(max_[0])
                active_nodes.append(reward)

                regret = max_[0] - exp_reward
                if regret > 0.5:
                    print("Regret: ", regret)

                model.update(episode)
            else:
                episode, rew, change = env.round(pulled_arm)
                if change:
                    print("change at t=", i)
                    max_ = env.opt(3)
                optimal_reward.append(max_[0])  # Moved this inside the loop
                exp_reward = env.expected_reward(pulled_arm, 100)[0]
                mean_rewards.append(exp_reward)

                regret = max_[0] - exp_reward
                if regret > 0.5:
                    print("Regret: ", regret)

                model.update(episode)

        all_mean_rewards.append(mean_rewards)
        all_optimal_rewards.append(optimal_reward)
        all_models_active_nodes.append(
            active_nodes
        )  # Append the active nodes list for this model to the master list

    return (
        np.array(all_mean_rewards),
        np.array(all_optimal_rewards),
        all_models_active_nodes,
    )


def matching_simulation(
    env,
    models,
    n_episodes,
    active_nodes=None,
    class_mapping=None,
    product_classes=[0, 1, 2],
    products_per_class=3,
):
    all_collected_rewards = []
    all_opts = []

    for model in models:
        collected_rewards = []
        opts = []

        products = product_classes * products_per_class
        for i in tqdm(range(n_episodes)):
            episode_active_nodes = (
                np.argwhere(active_nodes[i]) if active_nodes else None
            )
            customers = class_mapping[episode_active_nodes]

            # pull arm
            pulled_arm = model.pull_arm(episode_active_nodes, customers)
            # retrieve reward
            reward = env.round(pulled_arm)
            reward = [i[1] for i in reward]
            opt = env.opt(customers, products)
            print("Regret: ", opt - np.sum(reward))
            # update bandit
            model.update(pulled_arm, reward)
            opts.append(opt)
            collected_rewards.append(np.sum(reward))

        all_collected_rewards.append(collected_rewards)
        all_opts.append(opts)

    return np.array(all_collected_rewards), np.array(all_opts)


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
