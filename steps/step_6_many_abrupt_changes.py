from environments.ns_environment import SocialUnknownAbruptChanges, SocialNChanges
from learners.exp3 import EXP3ProbLearner
from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
from utils.metrics import compute_metrics, plot_metrics
from utils.simulation import influence_simulation


def step_6(
    env, graph_structure, n_nodes=30, n_phases=5, n_episodes=365
):

    # generate learner
    learner = EXP3ProbLearner(
        n_nodes=n_nodes, n_seeds=3, gamma=0.1, graph_structure=graph_structure[0]
    )
    ucb_bandit = UCBProbLearner(
        n_nodes=n_nodes, n_seeds=3, graph_structure=graph_structure[0]
    )

    metrics, models = influence_simulation(
        env, [learner, ucb_bandit], n_episodes=n_episodes, n_phases=n_phases
    )

    return metrics, models

def step_6_wrapper(
    graph_probabilities, graph_structure, n_episodes=365
):
    #define environments
    env1=SocialUnknownAbruptChanges(
        graph_probabilities, horizon=n_episodes, n_phases=5, change_prob=0.2
    )

    env2=SocialNChanges(graph_probabilities, n_phases=3)

    step_6(env1, graph_structure=graph_structure, n_nodes=30)
    step_6(env2, graph_structure=graph_structure, n_nodes=30)

    

    