from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ts_learners.ts_learner import TSProbLearner
from metrics import plot_metrics
from simulation import influence_simulation
from data_generator import generate_graph

# initialize graph
n_nodes = 30
edge_rate = 0.2
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
n_episodes = 365

model = UCBProbLearner(
    graph_probabilities.shape[0],
    n_seeds=3,
    graph_structure=graph_structure,
)
env = SocialEnvironment(graph_probabilities)

simulation_metrics=influence_simulation(env, model, n_episodes=n_episodes)
plot_metrics(*simulation_metrics, model_name="TSProbLearner", env_name="Social Environment")