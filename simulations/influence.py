from tqdm import tqdm
import numpy as np
from utils.metrics import compute_metrics, plot_metrics


def influence_simulation(env, models, n_episodes, n_phases=1, joint=False):
    all_metrics = []
    all_active_nodes = []
    for model in models:
        model_active_nodes = []
        max_ = env.opt(3)

        for i in tqdm(range(n_episodes)):
            pulled_arm = model.pull_arm()

            if n_phases == 1:
                episode, active_nodes = env.round(pulled_arm, joint=joint)
                if joint:
                    model_active_nodes.append(active_nodes)
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
        if joint:
            all_active_nodes.append(model_active_nodes)
        env.optimal_rewards = np.empty((0, 2))  # temporary fix

    plot_metrics(all_metrics, env_name="Social Environment")
    if joint:
        return all_metrics, all_active_nodes

    return all_metrics
