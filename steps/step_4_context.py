from environments.joint_environment import JointEnvironment
from learners.ts_learners.matching_ts_context import AdaptiveTSMatching
from learners.ts_learners.ts_prob_learner import TSProbLearner
from simulations.joint import joint_simulation
from utils.data_generator import generate_reward_parameters, generate_graph

import numpy as np


def step_4(
    n_nodes,
    graph_probabilities,
    graph_structure,
    n_seeds,
    customer_features,
    feature_mapping,
    n_customer_classes,
    n_product_classes,
    products_per_class,
    reward_parameters,
    n_exp=365,
):
    ts_bandit = TSProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)
    ts_matching = AdaptiveTSMatching(
        n_customer_classes * n_product_classes,
        n_product_classes,
        products_per_class,
    )

    # initialise environment
    joint_env = JointEnvironment(
        graph_probabilities, reward_parameters, feature_mapping, context=True
    )

    social, matching = joint_simulation(
        joint_env,
        influence_models=[ts_bandit],
        matching_models=[ts_matching],
        class_mapping=customer_features,
        n_episodes=n_exp,
        n_phases=1,
    )

    return social, matching
