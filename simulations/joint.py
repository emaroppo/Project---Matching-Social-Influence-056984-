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
    skip_expected_rewards=False,
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
            skip_expected_rewards=skip_expected_rewards,
        )

        # Compute and store metrics for the current matching model
        matching_metrics.append(metrics[0])

    # Optionally, you can plot the combined metrics of influence and matching models
    plot_metrics(matching_metrics, env_name="Joint Environment")

    return metrics, influence_metrics
