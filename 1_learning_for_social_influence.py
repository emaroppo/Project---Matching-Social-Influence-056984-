from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ts_learners.ts_learner import TSProbLearner
from utils.metrics import plot_metrics
from utils.simulation import influence_simulation
from utils.data_generator import generate_graph

# set arguments
n_nodes = 30
edge_rate = 0.2
n_episodes = 365

# generate graph and environment
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
env = SocialEnvironment(graph_probabilities)

# initialise bandit
model = UCBProbLearner(
    graph_probabilities.shape[0],
    n_seeds=3,
    graph_structure=graph_structure,
)

# run simulation
simulation_metrics = influence_simulation(env, model, n_episodes=n_episodes, n_phases=1)
plot_metrics(
    *simulation_metrics, model_name="TSProbLearner", env_name="Social Environment"
)
