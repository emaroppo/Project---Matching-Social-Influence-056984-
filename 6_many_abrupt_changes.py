from environments.ns_environment import SocialUnknownAbruptChanges
from learners.exp3 import EXP3
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


phases = 5
horizon = 365
n_nodes = 30
edge_rate = 0.1

# generate graph
graph_structure = [
    np.random.binomial(1, edge_rate, (n_nodes, n_nodes)) for i in range(phases)
]
graph_probabilities = [
    np.random.uniform(0.1, 0.2, (n_nodes, n_nodes)) for i in range(phases)
]

graph_probabilities = [i * j for i, j in zip(graph_structure, graph_probabilities)]

# generate environment
env = SocialUnknownAbruptChanges(
    graph_probabilities, horizon=horizon, n_phases=phases, change_prob=0.3
)

# generate learner
learner = EXP3(n_arms=n_nodes, gamma=0.1)

# run experiment
for i in tqdm(range(horizon)):
    pulled_arm = learner.pull_arm()
    reward = env.round([pulled_arm], joint=False)
    learner.update(pulled_arm, reward)
