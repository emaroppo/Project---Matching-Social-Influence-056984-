import numpy as np

class Environment:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])       
        return reward
    
class MatchingEnvironment(Environment):
    def __init__(self, reward_matrix):
        self.reward_matrix = reward_matrix
        self.n_arms = reward_matrix.size
        self.t = 0

    def round(self, pulled_arm):
        customer_indices, product_indices = pulled_arm
        reward_functions = self.reward_matrix[customer_indices, product_indices]
        # Apply each function or retrieve number to get rewards
        reward = np.array([func() if callable(func) else func for func in reward_functions])


        return reward
