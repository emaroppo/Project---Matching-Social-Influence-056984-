import numpy as np
from tqdm import tqdm
from utils.matching import hungarian_algorithm
from utils.influence import greedy_algorithm, simulate_episode


def get_reward(node_class, product_class, rewards_parameters):
    return 100-np.random.normal(rewards_parameters[0][node_class-1, product_class], rewards_parameters[1][node_class-1, product_class])

def clairvoyant(graph_probabilities, node_classes, rewards_parameters, n_exp):

    rewards_per_exp=np.zeros(n_exp)
    opt_seeds=greedy_algorithm(graph_probabilities, 3, 1000, 100)

    for i in tqdm(range(n_exp)):
        active_nodes=simulate_episode(graph_probabilities, opt_seeds, 100)[1]*node_classes
        
        #remove all zeros from array (nodes that were not activated)
        active_nodes=active_nodes[active_nodes!=0]
        
        #print(active_nodes)
        #compute rewards
        hun_matrix_dim=max(len(active_nodes), 9)
        
        rewards=np.zeros((hun_matrix_dim, hun_matrix_dim))

        for j in range(hun_matrix_dim):
            for l in range(hun_matrix_dim):
                if j<len(active_nodes) and l<9:
                    rewards[j,l]=get_reward(int(active_nodes[j]), l//3, rewards_parameters)
                else:
                    rewards[j,l]=0

        optimum= hungarian_algorithm(rewards)[0]
        optimum= 100-optimum #convert to reward
        #set all 100s to 0s
        optimum[optimum==100]=0
        #print(optimum)
        rewards_per_exp[i]=np.sum(optimum)
        #print('Optimum reward: ', np.sum(optimum))
        #print('-------------------')
    return np.mean(rewards_per_exp), np.std(rewards_per_exp)