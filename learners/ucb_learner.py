import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from learners.learner import Learner


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)  # count the number of times each arm has been pulled
        self.confidence = np.array([np.inf] * n_arms)
    
    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward) / self.n_pulls[pulled_arm]
        for a in range(self.n_arms):
            n_samples = max(1, self.n_pulls[a])
            self.confidence[a] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf
        
        self.update_observations(pulled_arm, reward)

class MAUCBLearner(UCBLearner):
    def __init__(self, n_arms, n_seeds):
        super().__init__(n_arms)
        self.n_seeds = n_seeds
        self.n_pulls = np.zeros(n_arms)  # count the number of times each arm has been pulled
    
    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        pulled_arms = []
        while len(pulled_arms) < self.n_seeds:
            pulled_arm = np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])
            upper_confidence_bound[pulled_arm] = -np.inf
            pulled_arms.append(pulled_arm)
        
            
        return pulled_arms

    def update(self, pulled_arms, reward):
        self.t += 1
        for pulled_arm in pulled_arms:
            self.n_pulls[pulled_arm] += 1
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward) / self.n_pulls[pulled_arm]
            n_samples = max(1, self.n_pulls[pulled_arm])
            self.confidence[pulled_arm] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf

            self.update_observations(pulled_arm, reward)


class UCBMatching(UCBLearner):
    def __init__(self, n_arms, n_customer_classes, n_product_classes):
        super().__init__(n_arms)
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.empirical_means = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.n_pulls = np.zeros((n_customer_classes + 1, n_product_classes + 1), dtype=int)
        self.confidence = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.rewards_per_arm = {} #temporary fix

    def pull_arms(self, customer_classes):
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3

        extended_upper_conf = np.repeat(upper_conf, [self.n_product_classes]*(self.n_customer_classes) + [1], axis=1)
    
        available_upper_conf = extended_upper_conf[customer_classes, :]
        row_ind, col_ind = linear_sum_assignment(-available_upper_conf)

        best_arms_global = [(customer_classes[row] if row < len(customer_classes) else self.n_customer_classes, col//3) 
                            for row, col in zip(row_ind, col_ind)]
        return best_arms_global

    def update(self, pulled_arms, rewards):
        for pulled_arm, reward in zip(pulled_arms, rewards):
            pulled_arm_customer, pulled_arm_product = pulled_arm  # unpacking tuple
            self.t += 1
            
            self.n_pulls[pulled_arm_customer, pulled_arm_product] += 1
            self.empirical_means[pulled_arm_customer, pulled_arm_product] = (self.empirical_means[pulled_arm_customer, pulled_arm_product] * (self.n_pulls[pulled_arm_customer, pulled_arm_product] - 1) + reward) / self.n_pulls[pulled_arm_customer, pulled_arm_product]            
            # Update observations
            # If `self.rewards_per_arm` is a dictionary with tuple keys
            if pulled_arm not in self.rewards_per_arm:
                self.rewards_per_arm[pulled_arm] = []
            self.rewards_per_arm[pulled_arm].append(reward)

        




        
#da finire
class CUMSUMUCBMatching(UCBMatching):
    def __init__(self, n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols)
        self.change_detection = [[] for _ in range(n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(n_arms)]
        self.detection = [[] for _ in range(n_arms)]
        self.alpa = alpha 

    