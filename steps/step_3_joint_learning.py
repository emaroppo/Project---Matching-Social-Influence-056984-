from learners.ts_learners.matching_ts import TSMatching
from learners.ts_learners.ts_prob_learner import TSProbLearner
from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
from learners.ucb_learners.matching_ucb import UCBMatching
from environments.joint_environment import JointEnvironment
from simulations.joint import joint_simulation


def step_3(
    n_nodes,
    graph_probabilities,
    graph_structure,
    n_seeds,
    class_mapping,
    n_customer_classes,
    n_product_classes,
    products_per_class,
    reward_parameters,
    n_exp=365,
):
    ts_bandit = TSProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)
    ucb_bandit = UCBProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)

    ts_matching = TSMatching(
        n_customer_classes * n_product_classes,
        n_customer_classes,
        n_product_classes,
        products_per_class,
    )
    ucb_matching = UCBMatching(
        n_customer_classes * n_product_classes,
        n_customer_classes,
        n_product_classes,
        products_per_class,
    )
    # initialise environment
    joint_env = JointEnvironment(graph_probabilities, reward_parameters, class_mapping)

    social, matching = joint_simulation(
        joint_env,
        [ts_bandit, ucb_bandit],
        [ts_matching, ucb_matching],
        class_mapping=class_mapping,
        n_episodes=n_exp,
        n_phases=1,
    )

    return social, matching
