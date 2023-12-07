from environments.matching_environment import MatchingEnvironment2
from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ts_learners.matching_ts import TSMatching
from utils.metrics import compute_metrics, plot_metrics
from utils.simulation import matching_simulation
import numpy as np


def step_2(
    reward_parameters,
    n_node_classes,
    n_product_classes,
    n_products_per_class,
    n_episodes=365,
    active_nodes=None,
    class_mapping=None,
):
    env = MatchingEnvironment2(reward_parameters=reward_parameters)

    # initialize bandit
    ucb_bandit = UCBMatching(
        n_node_classes * n_product_classes,
        n_node_classes,
        n_product_classes,
        n_products_per_class,
    )
    ts_bandit = TSMatching(
        n_node_classes * n_product_classes,
        n_node_classes,
        n_product_classes,
        n_products_per_class,
    )

    # run simulation

    metrics, models = matching_simulation(
        env=env,
        models=[ucb_bandit, ts_bandit],
        n_episodes=n_episodes,
        active_nodes=active_nodes,
        class_mapping=class_mapping,
    )
    return metrics, models
