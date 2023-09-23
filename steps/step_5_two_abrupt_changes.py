from environments.ns_environment import SocialNChanges
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ucb_learners.ns_ucb import SWUCBProbLearner, CDUCBProbLearner
from utils.data_generator import generate_graph
from utils.metrics import compute_metrics, plot_metrics
from utils.simulation import influence_simulation


def step_5(graph_probabilities, graph_structure, n_phases=3, n_episodes=365):
    env = SocialNChanges(graph_probabilities, n_phases=n_phases)
    # initialise bandit
    model1 = UCBProbLearner(30, 3, graph_structure=graph_structure)
    model2 = SWUCBProbLearner(30, 3, 121, graph_structure=graph_structure)
    model3 = CDUCBProbLearner(30, 3, graph_structure=graph_structure)

    # run simulation
    all_rewards, all_optimal_rewards, _ = influence_simulation(
        env, [model1, model2, model3], n_episodes=n_episodes, n_phases=3
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
        model_names=["UCB", "SW UCB", "CD UCB"],
        env_name="Social Environment",
    )
