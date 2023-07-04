from environments.matching_environment import MatchingEnvironment
from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ts_learners.matching_ts import TSMatching
from utils.matching import generate_reward

import numpy as np
from tqdm import tqdm

node_classes=3
product_classes=3
products_per_class=3
#reward parameters
means=np.random.uniform(10, 20, (3,3))
#arr of 1s of size (3,3)
std_dev=np.ones((3,3))
reward_parameters=(means, std_dev)
print(means)




constant_nodes = False
rewards_per_node = np.empty((node_classes, product_classes), dtype=object)
if constant_nodes:
    rewards_matrix=np.random.normal(reward_parameters[0], reward_parameters[1], (node_classes, product_classes))
    
else:
    rewards_matrix=np.empty((node_classes, product_classes), dtype=object)
    for i in range(node_classes):
        for j in range(product_classes):
            rewards_matrix[i,j]=generate_reward(reward_parameters[0][i,j], reward_parameters[1][i,j])
print(rewards_matrix.shape)
#initialize environment
env=MatchingEnvironment(rewards_matrix)
#initialize bandit
ucb_bandit=UCBMatching(rewards_matrix.size, rewards_matrix.shape[0], rewards_matrix.shape[1], products_per_class)
ts_bandit=TSMatching(rewards_matrix.size, rewards_matrix.shape[0], rewards_matrix.shape[1], products_per_class)


n_experiments=1000

#random list of 0s, 1s, 2s of variable length between 6 and 12

for i in tqdm(range(n_experiments)):
    customers=np.random.randint(0, 3, 4)
    #pull arm
    ucb_pulled_arm=ucb_bandit.pull_arms(customers)
    ts_bandit_pulled_arm=ts_bandit.pull_arms(customers)
    #print(ucb_pulled_arm, ts_bandit_pulled_arm)
    #retrieve reward
    ucb_reward=env.round(ucb_pulled_arm)
    ts_bandit_reward=env.round(ts_bandit_pulled_arm)
    #update bandit
    ucb_bandit.update(ucb_pulled_arm, ucb_reward)
    ts_bandit.update(ts_bandit_pulled_arm, ts_bandit_reward)


print(env.round(ucb_bandit.pull_arms(customers)).sum())
print(env.round(ts_bandit.pull_arms(customers)).sum())
print(customers)