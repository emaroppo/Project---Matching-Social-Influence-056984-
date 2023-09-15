import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from environments.environment import Environment
from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ucb_learners.ns_ucb import (
    SWUCBProbLearner,
    CDUCBProbLearner,
)
from learners.ts_learners.ts_learner import TSLearner, TSProbLearner
from metrics import compute_metrics, plot_metrics
from data_generator import generate_graph

# from clairvoyant import clairvoyant


# initialize graph
n_nodes = 30
edge_rate = 0.2
graph_structure = np.random.binomial(1, 0.1, (30, 30))
graph_probabilities = np.random.uniform(0.1, 0.2, (30, 30)) * graph_structure

graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)

# parameters for (gaussian) reward distributions for each node class and product class
means = np.random.uniform(10, 20, (3, 3))
stds = np.random.uniform(1, 3, (3, 3))
reward_parameters = (means, stds)

n_episodes = 365

# Main simulation loop
model = CDUCBProbLearner(
    graph_probabilities.shape[0],
    n_seeds=3,
    graph_structure=graph_structure,
)
env = SocialEnvironment(graph_probabilities)
max_ = env.opt(3)
print(max_[0])
mean_rewards = []

for i in tqdm(range(n_episodes)):
    pulled_arm = model.pull_arm()
    episode, rew = env.round(pulled_arm)
    exp_reward = env.expected_reward(pulled_arm, 100)[0]
    mean_rewards.append(exp_reward)

    regret = max_[0] - exp_reward
    if regret > 0.5:
        print("Regret: ", regret)

    model.update(episode)

metrics = compute_metrics(np.array(mean_rewards), np.array([max_[0]] * n_episodes))
plot_metrics(*metrics, model_name="TSProbLearner", env_name="Social Environment")
