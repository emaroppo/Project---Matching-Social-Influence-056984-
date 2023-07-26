from environments.environment import Environment
from scipy.optimize import linear_sum_assignment
import numpy as np


def generate_reward(mean, std_dev):
    return lambda: np.random.normal(mean, std_dev)

class MatchingEnvironment(Environment):
    def __init__(self, reward_parameters, constant_nodes=True):

        if constant_nodes:
            reward_matrix=np.random.normal(reward_parameters[0], reward_parameters[1])            
        else:
            reward_matrix=np.empty((reward_parameters[0].shape[0], reward_parameters[0].shape[1]), dtype=object)
            for i in range(reward_parameters[0].shape[0]):
                for j in range(reward_parameters[0].shape[1]):
                    reward_matrix[i,j]=generate_reward(reward_parameters[0][i,j], reward_parameters[1][i,j])
        
        # add a row and a column of zeros to the reward matrix to represent the case in which no match is made
        reward_matrix = np.hstack((reward_matrix, np.zeros((reward_matrix.shape[0], 1))))
        reward_matrix = np.vstack((reward_matrix, np.zeros((1, reward_matrix.shape[1]))))
        self.reward_matrix = reward_matrix

        self.reward_parameters = reward_parameters

        self.n_arms = reward_matrix.size
        self.t = 0

    def round(self, pulled_arms):
        rewards = [self.reward_matrix[pulled_arm] for pulled_arm in pulled_arms]
        
        #iterate through all cells of rewards, if a cell is callable, call it and replace it with the result
        for i in range(len(rewards)):
            if callable(rewards[i]):
                rewards[i] = rewards[i]()
        
        return np.array(rewards)
    
    def opt(self, customer_classes, product_classes):
        n_customers= len(customer_classes)
        n_products= len(product_classes)
        
        if n_customers> n_products:
            #pad products with 3s
            product_classes= [*product_classes, *[3]*(n_customers-n_products)]
        elif n_products> n_customers:
            #pad customers with 3s
            customer_classes= [*customer_classes, *[3]*(n_products-n_customers)]

        new_reward_matrix = np.zeros((n_customers, n_products))
        expected_rewards = self.reward_parameters[0].copy()
        expected_rewards=np.hstack((expected_rewards, np.zeros((expected_rewards.shape[0], 1))))
        expected_rewards=np.vstack((expected_rewards, np.zeros((1, expected_rewards.shape[1]))))
        for i in range(n_customers):
            for j in range(n_products):
                new_reward_matrix[i,j] = expected_rewards[customer_classes[i], product_classes[j]]

        #print(new_reward_matrix)
        #find optimal assignment
        row_ind, col_ind = linear_sum_assignment(-new_reward_matrix)
        #compute reward of optimal assignment
        reward = new_reward_matrix[row_ind, col_ind].sum()
        return reward