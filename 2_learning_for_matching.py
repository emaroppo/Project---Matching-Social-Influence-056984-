from environments.environment import MatchingEnvironment
from learners.ucb_learner import UCBMatching

import numpy as np
from tqdm import tqdm

node_classes=3
product_classes=3
products_per_class=3
#reward parameters
means=np.random.uniform(10, 20, (3,3))
stds=np.random.uniform(1, 3, (3,3))
reward_parameters=(means, stds)

#reward matrix 
def generate_reward(mean, std_dev):
    return lambda: np.random.normal(mean, std_dev)

node_classes = np.random.randint(1, node_classes+1, 9)
product_classes = np.arange(1, product_classes+1)

constant_nodes = False
rewards_per_node = np.empty((node_classes.size, product_classes.size), dtype=object)
for i in range(node_classes.size):
    for j in range(product_classes.size):
        if constant_nodes:

            rewards_per_node[i,j] = np.random.normal(reward_parameters[0][node_classes[i]-1, product_classes[j]-1], reward_parameters[1][node_classes[i]-1, product_classes[j]-1])
        
        else:
            mean = reward_parameters[0][node_classes[i]-1, product_classes[j]-1]
            std_dev = reward_parameters[1][node_classes[i]-1, product_classes[j]-1]
            rewards_per_node[i,j] = generate_reward(mean, std_dev)
#repeat rewards for each product class, pad rewards matrix with 0s to match the number of products and nodes
rewards_matrix=np.repeat(rewards_per_node, products_per_class, axis=1)
rewards_matrix=np.pad(rewards_matrix, ((0,0), (0, len(node_classes)-len(product_classes))), 'constant', constant_values=0)

#initialize environment
env=MatchingEnvironment(rewards_matrix)
#initialize bandit
ucb_bandit=UCBMatching(rewards_matrix.size, rewards_matrix.shape[0], rewards_matrix.shape[1])

n_experiments=1000

for i in tqdm(range(n_experiments)):
    #pull arm
    pulled_arm=ucb_bandit.pull_arm()
    print(pulled_arm)
    #retrieve reward
    reward=env.round(pulled_arm)
    #update bandit
    ucb_bandit.update(pulled_arm, reward)


