from environments.ns_environment import SocialUnknownAbruptChanges
from learners.exp3 import EXP3ProbLearner
from learners.ucb_learners.ucb_learner import UCBProbLearner
from utils.metrics import compute_metrics, plot_metrics
from utils.simulation import influence_simulation

def step_6(graph_probabilities, graph_structure, n_nodes=30, n_phases=5, horizon=365):

    # generate environment
    env = SocialUnknownAbruptChanges(graph_probabilities, horizon=horizon, n_phases=n_phases, change_prob=0.2)

    # generate learner
    learner = EXP3ProbLearner(n_nodes=n_nodes, n_seeds=3, gamma=0.1, graph_structure=graph_structure[0])
    ucb_bandit = UCBProbLearner(n_nodes=n_nodes, n_seeds=3, graph_structure=graph_structure[0])


    all_rewards, all_optimal_rewards, _=influence_simulation(env, [learner, ucb_bandit], n_episodes=horizon, n_phases=n_phases)

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
    return (learner, env), (all_rewards, all_optimal_rewards, all_instantaneous_regrets, all_cumulative_regrets)