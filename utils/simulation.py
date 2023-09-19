import numpy as np
from tqdm import tqdm


def influence_simulation(env, models, n_episodes, n_phases=1):
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
                episode, active_nodes = env.round(pulled_arm)
                exp_reward = env.expected_reward(pulled_arm, 100)[0]
                mean_rewards.append(exp_reward)
                optimal_reward.append(max_[0])  # Moved this inside the loop

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
            customers = class_mapping[active_nodes[i]]

            # pull arm
            pulled_arm = model.pull_arm(active_nodes[i], customers)
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
        env.social_environment, influence_models, n_episodes, n_phases
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
