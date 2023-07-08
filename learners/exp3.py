from learners.learner import Learner
import numpy as np


class EXP3(Learner):
    def __init__(self, n_arms, gamma):
        super().__init__(n_arms)
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    def get_probabilities(self):
        total_weight = sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + (
            self.gamma / self.n_arms
        )
        return probs

    def pull_arm(self):
        # Choose an action based on the current probability distribution
        probs = self.get_probabilities()
        arm = np.random.choice(self.n_arms, p=probs)
        return arm

    def update(self, pulled_arm, reward):
        # Update the weight of the selected arm based on the received reward
        estimated_reward = reward / self.get_probabilities()[pulled_arm]
        self.weights[pulled_arm] *= np.exp(
            (self.gamma / self.n_arms) * estimated_reward
        )
        self.update_observations(pulled_arm, reward)
