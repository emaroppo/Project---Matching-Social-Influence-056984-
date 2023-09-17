from environments.ns_environment import SocialNChanges
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ucb_learners.ns_ucb import SWUCBProbLearner, CDUCBProbLearner
import numpy as np
from data_generator import generate_graph
from metrics import plot_metrics
from simulation import influence_simulation

# set arguments
n_nodes = 30
edge_rate = 0.2
n_phases = 2
n_episodes = 365

# initialise graph and environment
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate, n_phases=3)
env = SocialNChanges(graph_probabilities, n_phases=n_phases)

# initialise bandit
model = SWUCBProbLearner(30, 3, 121, graph_structure=graph_structure)

# run simulation
simulation_metrics = influence_simulation(env, model, n_episodes=n_episodes, n_phases=3)
plot_metrics(
    *simulation_metrics, model_name="SWUCBProbLearner", env_name="Social Environment"
)
