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
