from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ts_learners.ts_learner import TSProbLearner
from utils.metrics import compute_metrics, plot_metrics
from utils.simulation import influence_simulation
from utils.data_generator import generate_graph
import numpy as np

# set arguments
n_nodes = 30
edge_rate = 0.2
n_episodes = 365

# generate graph and environment
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
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

# run simulation
all_rewards, all_optimal_rewards = influence_simulation(
    env, [model1, model2], n_episodes=n_episodes, n_phases=1
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
