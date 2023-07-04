from learners.ucb_learners.ucb_learner import UCBLearner
import numpy as np


class JointMAUCBLearner(UCBLearner):
    def __init__(self, n_nodes, n_seeds):
        super().__init__(n_nodes)
        self.n_seeds = n_seeds
        self.n_pulls = np.zeros(n_nodes)  # count the number of times each arm has been pulled
    def pull_arms(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        pulled_arms = []
        while len(pulled_arms) < self.n_seeds:
            pulled_arm = np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])
            upper_confidence_bound[pulled_arm] = -np.inf
            pulled_arms.append(pulled_arm) #check whether ties are broken randomly or deterministically
            
            
        return pulled_arms
    
    def update(self, pulled_arms, reward):
        self.t += 1
        for pulled_arm in pulled_arms:
            self.n_pulls[pulled_arm] += 1
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward) / self.n_pulls[pulled_arm]
            n_samples = max(1, self.n_pulls[pulled_arm])
            self.confidence[pulled_arm] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf

            self.update_observations(pulled_arm, reward)

            