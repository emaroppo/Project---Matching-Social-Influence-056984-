import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from environments.environment import Environment
from learners.ucb_learner import UCBLearner
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
envs=[]
ucb_bandits=[]
ts_bandits=[]

for i in graph_probabilities:
    #drop 0s / non existing edges
    p=i[i!=0]
    #initialize environment for each node
    env=Environment(p)
    envs.append(env)
    #initialize bandit for each node
    ucb_bandit=UCBLearner(len(p))
    ucb_bandits.append(ucb_bandit)
    ts_bandit=TSLearner(len(p))
    ts_bandits.append(ts_bandit)


for i in tqdm(range(n_episodes)):
    for env, ucb_bandit, ts_bandit in zip(envs, ucb_bandits, ts_bandits):
        
        #pull arm
        ucb_pulled_arm=ucb_bandit.pull_arm()
        ts_pulled_arm=ts_bandit.pull_arm()
        #retrieve reward
        ucb_reward=env.round(ucb_pulled_arm)
        ts_reward=env.round(ts_pulled_arm)
        
        #update bandit
        ucb_bandit.update(ucb_pulled_arm, ucb_reward)
        ts_bandit.update(ts_pulled_arm, ts_reward)


#print estimated probabilities

for env, ucb_bandit, ts_bandit in zip(envs, ucb_bandits, ts_bandits):
    print(ucb_bandit.empirical_means)

    print(env.probabilities)
    #calculate probs from beta parameters
    ts_probs=ts_bandit.beta_parameters[:,0]/(ts_bandit.beta_parameters[:,0]+ts_bandit.beta_parameters[:,1])
    print(ts_probs)
    print('------------------')
