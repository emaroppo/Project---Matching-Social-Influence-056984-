from environments.ns_environment import SocialUnknownAbruptChanges, SocialNChanges
from learners.exp3 import EXP3ProbLearner
from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
from learners.ucb_learners.ns_ucb import SWUCBProbLearner, CDUCBProbLearner
from simulations.influence import influence_simulation


def step_6_inner(env, graph_structure, n_nodes=30, n_phases=5, n_episodes=365):
    # generate learner
    learner = EXP3ProbLearner(
        n_nodes=n_nodes, n_seeds=3, gamma=0.1, graph_structure=graph_structure[0]
    )
    ucb_bandit = UCBProbLearner(
        n_nodes=n_nodes, n_seeds=3, graph_structure=graph_structure[0]
    )

    sw_bandit = SWUCBProbLearner(
        n_nodes=n_nodes, n_seeds=3, window_size=90, graph_structure=graph_structure[0]
    )

    cd_bandit = CDUCBProbLearner(
        n_nodes=n_nodes, n_seeds=3, eps=0.05, graph_structure=graph_structure[0]
    )

    metrics, models = influence_simulation(
        env, [learner, ucb_bandit, sw_bandit, cd_bandit], n_episodes=n_episodes, n_phases=n_phases
    )

    return metrics, models


def step_6(graph_probabilities, graph_structure, n_episodes=365):
    # define environments
    env1 = SocialNChanges(graph_probabilities, n_phases=3)
    

    env2 = SocialUnknownAbruptChanges(
        graph_probabilities, n_episodes=n_episodes, n_phases=5, change_prob=0.05
    )

    step_6_inner(env1, graph_structure=graph_structure, n_nodes=30, n_phases=3, n_episodes=n_episodes)
    step_6_inner(env2, graph_structure=graph_structure, n_nodes=30, n_phases=5, n_episodes=n_episodes)
