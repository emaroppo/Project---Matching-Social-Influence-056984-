import numpy as np
import matplotlib.pyplot as plt
from utils.matching import generate_reward
from learners.ts_learners.matching_ts import TSMatching
from learners.ucb_learners.matching_ucb import UCBMatching 
from learners.ucb_learners.joint_ucb import JointMAUCBLearner
from environments.matching_environment import MatchingEnvironment 
from environments.social_environment import SocialEnvironment
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
np.random.seed(43)
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
social_env = SocialEnvironment(graph_probabilities)
matching_env = MatchingEnvironment(reward_matrix)

#initialise bandits
maucb_bandit = JointMAUCBLearner(n_nodes, n_seeds)
matching_ucb = UCBMatching(reward_matrix.size,reward_matrix.shape[0], reward_matrix.shape[1],products_per_class)

for i in tqdm(range(n_exp)):
    ucb_pulled_arms=maucb_bandit.pull_arms()
    print(ucb_pulled_arms)
    #retrieve activated nodes
    social_reward = social_env.round(ucb_pulled_arms, joint=True)
    #print(social_reward)
    #convert to list of integer indices corresponding to the position of activated nodes in the reward matrix
    social_reward = np.where(social_reward == 1)[0]
    #convert to list of classes
    social_reward = [node_classes[i] for i in social_reward]
    #perform matching assuming customer classes are known but distributions unknown
    prop_match = matching_ucb.pull_arms(social_reward)
    #retrieve reward
    matching_reward = matching_env.round(prop_match)

    #update bandits
    matching_ucb.update(prop_match, matching_reward)
    maucb_bandit.update(ucb_pulled_arms, matching_reward.sum())

    #convert social 

print(ucb_pulled_arms)
print(len(social_reward))
print(matching_reward.sum())
#estimate best seeds then perform matching?
#estimate seeds on the basis of matching payoff?
