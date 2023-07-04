from learners.ts_learners.ts_learner import TSLearner
import numpy as np

class JointTSLearner(TSLearner):
    def __init__(self, n_nodes, n_seeds):
        super().__init__(n_nodes)
        self.n_seeds = n_seeds
        self.mu = np.zeros(n_nodes)
        self.n_samples = np.zeros(n_nodes, dtype=int)
        
    def pull_arms(self):
        theta_sample = np.random.normal(self.mu, 1)
        pulled_arms = []
        for _ in range(self.n_seeds):
            pulled_arm = np.argmax(theta_sample)
            theta_sample[pulled_arm] = -np.inf
            pulled_arms.append(pulled_arm)
        return pulled_arms
    
    def update(self, pulled_arms, reward):
        self.t += 1
        for pulled_arm in pulled_arms:
            self.update_observations(pulled_arm, reward)
            self.n_samples[pulled_arm] += 1
            self.mu[pulled_arm] = (self.mu[pulled_arm] * (self.n_samples[pulled_arm] - 1) + reward) / self.n_samples[pulled_arm]
