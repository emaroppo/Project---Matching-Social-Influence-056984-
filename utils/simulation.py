import numpy as np
from tqdm import tqdm
from metrics import compute_metrics


def influence_simulation(env, model, n_episodes, n_phases=1):
    mean_rewards = []
    optimal_reward = []

    max_ = env.opt(3)

    for i in tqdm(range(n_episodes)):
        pulled_arm = model.pull_arm()

        if n_phases == 1:
            episode, _ = env.round(pulled_arm)
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

    metrics = compute_metrics(np.array(mean_rewards), np.array(optimal_reward))

    return metrics


def matching_simulation(
    env,
    model,
    n_episodes,
    active_nodes=None,
    class_mapping=None,
    product_classes=[0, 1, 2],
    products_per_class=3,
):
    products = product_classes * products_per_class
    opts = []

    for i in tqdm(range(n_episodes)):
        if active_nodes is None and class_mapping is not None:
            customers_idx = np.random.randint(
                1, len(class_mapping), np.random.randint(6, 12)
            )
            customers = class_mapping[customers_idx]

        # pull arm
        pulled_arm = model.pull_arm(customers)
        # retrieve reward
        reward = env.round(pulled_arm)
        opt = env.opt(customers, products)
        # update bandit
        model.update(pulled_arm, reward)
        opts.append(opt)

    metrics = compute_metrics(model.collected_rewards, np.array(opts))
    return metrics
