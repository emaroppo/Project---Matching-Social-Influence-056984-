import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from environments.environment import SocialEnvironment
from learners.ucb_learner import UCBLearner, MAUCBLearner
from learners.ts_learner import TSLearner
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


#retrieve edge activation probabilities for each node
env=SocialEnvironment(graph_probabilities)
ucb_bandit=MAUCBLearner(n_arms=30, n_seeds=3)
ts_bandits=[]
rewards=[]

for i in tqdm(range(15000)):
    reward=env.round(ucb_bandit.pull_arm())/3
    rewards.append(reward)
    ucb_bandit.update(ucb_bandit.pull_arm(), reward)

    if i%1000==0: print(sum(rewards)/i)
    
print(sum(rewards[-1000:])/1000)
print(ucb_bandit.empirical_means)

