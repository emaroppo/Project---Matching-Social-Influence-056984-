from learners.ts_learners.ts_learner import TSLearner
import numpy as np

class SWTS_Learner(TSLearner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)

        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_reward = np.sum(self.rewards_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.beta_parameters[arm, 0] = cum_reward + 1.0
            self.beta_parameters[arm, 1] = n_samples - cum_reward + 1.0
    
    