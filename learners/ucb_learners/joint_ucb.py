from learners.ucb_learners.ucb_learner import UCBLearner
import numpy as np


class JointMAUCBLearner(UCBLearner):
    def __init__(self, n_nodes, n_seeds, initial_confidence=np.inf):
        super().__init__(n_nodes)
        self.n_seeds = n_seeds
        self.n_pulls = np.zeros(n_nodes)
        self.confidence = np.full(
            n_nodes, initial_confidence
        )  # Fill initial confidence with a large finite number

    def pull_arms(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        pulled_arms = []
        while len(pulled_arms) < self.n_seeds:
            max_confidence_arms = np.where(
                upper_confidence_bound == upper_confidence_bound.max()
            )[0]
            pulled_arm = np.random.choice(max_confidence_arms)
            upper_confidence_bound[pulled_arm] = -np.inf
            pulled_arms.append(pulled_arm)
        return pulled_arms

    def update(self, pulled_arms, reward):
        self.t += 1
        for pulled_arm in pulled_arms:
            self.n_pulls[pulled_arm] += 1
            self.empirical_means[pulled_arm] = (
                self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1)
                + reward
            ) / self.n_pulls[pulled_arm]
            n_samples = max(1, self.n_pulls[pulled_arm])
            self.confidence[pulled_arm] = (
                np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf
            )
            self.update_observations(pulled_arm, reward)
