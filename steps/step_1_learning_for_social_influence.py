from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
from learners.ts_learners.ts_prob_learner import TSProbLearner
from simulations.influence import influence_simulation


def step_1(graph_probabilities, graph_structure, n_episodes=365):
    env = SocialEnvironment(graph_probabilities)

    # initialise bandit
    model1 = UCBProbLearner(
        graph_probabilities.shape[0],
        n_seeds=3,
        graph_structure=graph_structure,
    )
    model2 = TSProbLearner(
        graph_probabilities.shape[0],
        n_seeds=3,
        graph_structure=graph_structure,
    )

    models = [model1, model2]

    # run simulation
    metrics, models = influence_simulation(
        env, models, n_episodes=n_episodes, n_phases=1
    )

    return (metrics, models, env)
