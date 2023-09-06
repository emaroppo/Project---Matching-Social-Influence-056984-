import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from environments.environment import Environment
from environments.social_environment import SocialEnvironment
from learners.ucb_learners.ucb_learner import UCBLearner, MAUCBLearner, UCBProbLearner
from learners.ts_learners.ts_learner import TSLearner
from metrics import compute_metrics, plot_metrics
from data_generator import generate_graph

# from clairvoyant import clairvoyant


# initialize graph
n_nodes = 30
edge_rate = 0.1
graph_structure = np.random.binomial(1, 0.1, (30, 30))
graph_probabilities = np.random.uniform(0.05, 0.2, (30, 30)) * graph_structure
#print type of graph_probabilities
print(type(graph_probabilities))


print(graph_probabilities.shape)
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
print(graph_probabilities.shape)
print(graph_probabilities)

# parameters for (gaussian) reward distributions for each node class and product class
means = np.random.uniform(10, 20, (3, 3))
stds = np.random.uniform(1, 3, (3, 3))
reward_parameters = (means, stds)

n_episodes = 365

"""
1. estimate activation probabilities implicitly by using bandits 
to find set of seeds maximising reward"""


def step_1(graph_probabilities, n_episodes, n_experiments=100):
    env = SocialEnvironment(graph_probabilities)
    ucb_bandits = []
    for i in tqdm(range(n_experiments)):
        ucb_bandit = MAUCBLearner(graph_probabilities.shape[0], n_seeds=3)

        for j in range(n_episodes):
            reward = env.round(ucb_bandit.pull_arm()) / 3
            ucb_bandit.update(ucb_bandit.pull_arm(), reward)
            ucb_bandits.append(ucb_bandit)

    return ucb_bandits, env


"""
1. estimate activation probabilities explicitly by using bandits
2. use estimated probabilities to compute optimal seeds with greedy algo + montecarlo
"""

# need to modify to keep track of reward for each episode
# (to plot regret etc.)


def step_1v2(graph_probabilities, n_episodes):
    # retrieve edge activation probabilities for each node
    envs = []
    ucb_bandits = []
    ts_bandits = []

    for p in graph_probabilities:
        # drop 0s / non-existing edges
        non_zero_indices = np.where(p != 0)[0]
        # initialize environment for each node
        env = Environment(p[non_zero_indices])
        envs.append(env)
        # initialize bandit for each node
        ucb_bandit = UCBLearner(len(non_zero_indices))
        ucb_bandits.append(ucb_bandit)
        ts_bandit = TSLearner(len(non_zero_indices))
        ts_bandits.append(ts_bandit)

    for env, ucb_bandit, ts_bandit in tqdm(
        zip(envs, ucb_bandits, ts_bandits),
        total=len(envs),
        desc="Estimating Activation Probabilities...",
    ):
        for _ in range(n_episodes):
            # pull arm
            ucb_pulled_arm = ucb_bandit.pull_arm()
            ts_pulled_arm = ts_bandit.pull_arm()
            # retrieve reward
            ucb_reward = env.round(ucb_pulled_arm)
            ts_reward = env.round(ts_pulled_arm)

            # update bandit
            ucb_bandit.update(ucb_pulled_arm, ucb_reward)
            ts_bandit.update(ts_pulled_arm, ts_reward)

    ucb_graph_probabilities = np.zeros(graph_probabilities.shape)
    ts_graph_probabilities = np.zeros(graph_probabilities.shape)

    # change all the non-0 terms of graph probabilities assigning respective bandit estimates
    for i, p in enumerate(graph_probabilities):
        non_zero_indices = np.where(p != 0)[0]
        ucb_graph_probabilities[i][non_zero_indices] = ucb_bandits[i].empirical_means
        ts_graph_probabilities[i][non_zero_indices] = ts_bandits[i].beta_parameters[
            :, 0
        ] / np.sum(ts_bandits[i].beta_parameters, axis=1)

    best_seeds_ucb = SocialEnvironment(ucb_graph_probabilities).opt_arm(3)
    best_seeds_ts = SocialEnvironment(ts_graph_probabilities).opt_arm(3)

    social_env = SocialEnvironment(graph_probabilities)

    ts_rewards = [social_env.round(best_seeds_ts) for i in range(n_episodes)]
    ucb_rewards = [social_env.round(best_seeds_ucb) for i in range(n_episodes)]

    return sum(ts_rewards) / len(ts_rewards), sum(ucb_rewards) / len(ucb_rewards)


model=UCBProbLearner(graph_probabilities.shape[0], n_seeds=3, graph_structure=graph_structure)
env=SocialEnvironment(graph_probabilities)
max_=env.opt(3)


mean_rewards=[]

for i in tqdm(range(n_episodes*2)):
    pulled_arm=model.pull_arm()
    episode,rew=env.round(pulled_arm)
    rewards = [env.round(pulled_arm) for _ in range(1000)]
    rewards = [r[1] for r in rewards]
    print(np.mean(rewards))
    print(max_)

    model.update(episode)


print(model.empirical_means)