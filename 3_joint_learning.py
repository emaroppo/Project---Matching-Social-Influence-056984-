import numpy as np
import matplotlib.pyplot as plt
from utils.matching import generate_reward
from learners.ts_learners.matching_ts import TSMatching
from learners.ts_learners.joint_ts import JointTSLearner
from learners.ucb_learners.matching_ucb import UCBMatching 
from learners.ucb_learners.joint_ucb import JointMAUCBLearner
from environments.joint_environment import JointEnvironment
from tqdm import tqdm

#init reward matrix, graph probabilities
n_nodes = 30
edge_rate=0.2
min_prob = 0.1
max_prob = 0.2
n_seeds = 3
customer_classes = 3
product_classes = 3
products_per_class = 3
n_exp = 1000

graph_probabilities = np.random.uniform(0, 1, (n_nodes, n_nodes))
constant_rewards = False
reward_means = np.random.uniform(10, 20, (customer_classes, product_classes))
reward_std_dev = np.ones((customer_classes, product_classes))



if constant_rewards:
    reward_matrix = np.random.normal(reward_means, reward_std_dev, (customer_classes, product_classes))
else:
    reward_matrix = np.empty((customer_classes, product_classes), dtype=object)
    for i in range(customer_classes):
        for j in range(product_classes):
            reward_matrix[i, j] = generate_reward(reward_means[i, j], reward_std_dev[i, j])

graph_structure = np.random.binomial(1, edge_rate, (n_nodes, n_nodes))
graph_probabilities = np.random.uniform(0.1, 0.2, (n_nodes,n_nodes)) * graph_structure
np.fill_diagonal(graph_probabilities, 0)

#link node ids to customer classes
node_classes = np.random.randint(0, customer_classes, n_nodes)

#initialise environments
joint_env = JointEnvironment(graph_probabilities, (reward_means, reward_std_dev), node_classes)
social_env = joint_env.social_environment
matching_env = joint_env.matching_environment

#initialise bandits
maucb_bandit = JointMAUCBLearner(n_nodes, n_seeds)
matching_ucb = UCBMatching(reward_matrix.size,reward_matrix.shape[0], reward_matrix.shape[1],products_per_class)

mats_bandit = JointTSLearner(n_nodes, n_seeds)
matching_ts = TSMatching(reward_matrix.size,reward_matrix.shape[0], reward_matrix.shape[1],products_per_class)

for i in tqdm(range(n_exp)):
    ucb_pulled_arms=maucb_bandit.pull_arms()
    ts_pulled_arms=mats_bandit.pull_arms()
    
    #retrieve activated nodes
    ucb_social_reward = social_env.round(ucb_pulled_arms, joint=True)
    ts_social_reward = social_env.round(ts_pulled_arms, joint=True)

    #print(social_reward)
    #convert to list of integer indices corresponding to the position of activated nodes in the reward matrix
    ucb_social_reward = np.where(ucb_social_reward == 1)[0]
    ts_social_reward = np.where(ts_social_reward == 1)[0]
    #convert to list of classes
    ucb_social_reward = [node_classes[i] for i in ucb_social_reward]
    ts_social_reward = [node_classes[i] for i in ts_social_reward]

    #perform matching assuming customer classes are known but distributions unknown
    ucb_prop_match = matching_ucb.pull_arms(ucb_social_reward)
    ts_prop_match = matching_ts.pull_arms(ts_social_reward)
    #retrieve reward
    ucb_matching_reward = matching_env.round(ucb_prop_match)
    ts_matching_reward = matching_env.round(ts_prop_match)

    #update bandits
    matching_ucb.update(ucb_prop_match, ucb_matching_reward)
    maucb_bandit.update(ucb_pulled_arms, ucb_matching_reward.sum())

    matching_ts.update(ts_prop_match, ts_matching_reward)
    mats_bandit.update(ts_pulled_arms, ts_matching_reward.sum())

    #convert social 

print(ucb_pulled_arms)
print(len(ucb_social_reward))
print(ucb_matching_reward.sum())
print(ts_pulled_arms)
print(len(ts_social_reward))
print(ts_matching_reward.sum())
print(joint_env.opt(n_seeds, [0,1,2]*3))

#estimate best seeds then perform matching?
#estimate seeds on the basis of matching payoff?
