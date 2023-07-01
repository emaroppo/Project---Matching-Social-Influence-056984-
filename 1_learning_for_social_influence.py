import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from environments.environment import Environment, SocialEnvironment
from learners.ucb_learner import UCBLearner, MAUCBLearner
from learners.ts_learner import TSLearner
from utils.influence import greedy_algorithm
#from clairvoyant import clairvoyant



#initialize graph
n_nodes=30
edge_rate=0.1
graph_structure=np.random.binomial(1, 0.3, (30,30))
graph_probabilities=np.random.uniform(0.1, 0.2, (30, 30))*graph_structure
node_classes=np.random.randint(1,4, graph_probabilities.shape[0])

#parameters for (gaussian) reward distributions for each node class and product class
means=np.random.uniform(10, 20, (3,3))
stds=np.random.uniform(1, 3, (3,3))
reward_parameters=(means, stds)

n_episodes=20000

'''
1. estimate activation probabilities implicitly by using bandits 
to find set of seeds maximising reward'''

def step_1(graph_probabilities, n_nodes, n_episodes):
    env=SocialEnvironment(graph_probabilities)
    ucb_bandit=MAUCBLearner(n_nodes, n_seeds=3)
    rewards=[]

    for i in tqdm(range(n_episodes)):
        reward=env.round(ucb_bandit.pull_arm())/3
        rewards.append(reward)
        ucb_bandit.update(ucb_bandit.pull_arm(), reward)

        if i%1000==0: print(sum(rewards)/i)
        
    return ucb_bandit, rewards

'''
1. estimate activation probabilities explicitly by using bandits
2. use estimated probabilities to compute optimal seeds with greedy algo + montecarlo
'''

#need to modify to keep track of reward for each episode 
#(to plot regret etc.)

def step_1v2(graph_probabilities, n_nodes, n_episodes):
    
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

    for env, ucb_bandit, ts_bandit in tqdm(zip(envs, ucb_bandits, ts_bandits), total=len(envs), desc='Estimating Activation Probabilities...'):
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
        ts_graph_probabilities[i][non_zero_indices] = ts_bandits[i].beta_parameters[:, 0] / np.sum(ts_bandits[i].beta_parameters, axis=1)

    best_seeds_ucb=greedy_algorithm(ucb_graph_probabilities, 3, 1000, 100)
    best_seeds_ts=greedy_algorithm(ts_graph_probabilities, 3, 1000, 100)

    social_env=SocialEnvironment(graph_probabilities)

    ts_rewards=[social_env.round(best_seeds_ts) for i in range(n_episodes)]
    ucb_rewards=[social_env.round(best_seeds_ucb) for i in range(n_episodes)]

    return sum(ts_rewards)/len(ts_rewards), sum(ucb_rewards)/len(ucb_rewards)
