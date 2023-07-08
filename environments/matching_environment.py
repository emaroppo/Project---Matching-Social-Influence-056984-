from environments.environment import Environment
from scipy.optimize import linear_sum_assignment
import numpy as np

class MatchingEnvironment(Environment):
    def __init__(self, reward_matrix):
        self.reward_matrix = reward_matrix
        # add a row and a column of zeros to the reward matrix to represent the case in which no match is made
        self.reward_matrix = np.hstack((self.reward_matrix, np.zeros((self.reward_matrix.shape[0], 1))))
        self.reward_matrix = np.vstack((self.reward_matrix, np.zeros((1, self.reward_matrix.shape[1]))))

        self.n_arms = reward_matrix.size
        self.t = 0

    def round(self, pulled_arms):
        rewards = [self.reward_matrix[pulled_arm] for pulled_arm in pulled_arms]
        
        #iterate through all cells of rewards, if a cell is callable, call it and replace it with the result
        for i in range(len(rewards)):
            if callable(rewards[i]):
                rewards[i] = rewards[i]()
        
        return np.array(rewards)
    
    def opt(self, customer_classes):
        #from the original reward matrix, given the customer_classes and 3 products for each product class, determine new reward matrix
        new_reward_matrix = np.zeros((len(customer_classes), 3))
        for i in range(len(customer_classes)):
            new_reward_matrix[i] = self.reward_matrix[customer_classes[i], :3]
        
        new_reward_matrix = np.repeat(new_reward_matrix, 3, axis=1)
        #find optimal assignment
        row_ind, col_ind = linear_sum_assignment(-new_reward_matrix)
        #compute reward of optimal assignment
        reward = new_reward_matrix[row_ind, col_ind].sum()
        return reward