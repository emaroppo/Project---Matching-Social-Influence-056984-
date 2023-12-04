from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
from learners.ts_learners.ts_prob_learner import TSProbLearner
from utils.metrics import compute_metrics, plot_metrics
from utils.simulation import influence_simulation


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
    all_rewards, all_optimal_rewards, _ = influence_simulation(
        env, models, n_episodes=n_episodes, n_phases=1
    )

    all_instantaneous_regrets = [
        compute_metrics(r, o)[2] for r, o in zip(all_rewards, all_optimal_rewards)
    ]
    all_cumulative_regrets = [
        compute_metrics(r, o)[3] for r, o in zip(all_rewards, all_optimal_rewards)
    ]

    plot_metrics(
        all_rewards,
        all_optimal_rewards,
        all_instantaneous_regrets,
        all_cumulative_regrets,
        model_names=["Model1", "Model2"],
        env_name="Social Environment",
    )

    return (models, env), (
        all_rewards,
        all_optimal_rewards,
        all_instantaneous_regrets,
        all_cumulative_regrets,
    )
