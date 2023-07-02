from learners.learner import Learner
import numpy as np
from scipy.optimize import linear_sum_assignment

class TSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

class TSMatching(TSLearner):
    def __init__(self, n_arms, n_customer_classes, n_product_classes, n_products_per_class=3):
        super().__init__(n_arms)
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.mu = np.zeros((n_customer_classes+1, n_product_classes+1))  # include the phantom class
        self.n_samples = np.zeros((n_customer_classes+1, n_product_classes+1), dtype=int)
        self.rewards_per_arm = {} #temporary fix
        self.n_products_per_class = n_products_per_class

    def pull_arms(self, customer_classes):
        theta_sample = np.random.normal(self.mu, 1)

        # Extended theta_sample matrix to include phantom class
        extended_theta_sample = np.zeros((self.n_customer_classes + 1, self.n_product_classes + 1))
        extended_theta_sample[:self.n_customer_classes+1, :self.n_product_classes+1] = theta_sample

        # Repeat entries in available_theta for each product
        available_theta = np.repeat(extended_theta_sample[customer_classes, :], self.n_products_per_class, axis=1)
        print(available_theta)

        row_ind, col_ind = linear_sum_assignment(-available_theta)
        best_arms_global = [(customer_classes[row] if row < len(customer_classes) else self.n_customer_classes, col//3)
                            for row, col in zip(row_ind, col_ind)]
        
        print(best_arms_global)

        return best_arms_global


    def update(self, pulled_arms, rewards):
        for pulled_arm, reward in zip(pulled_arms, rewards):
            pulled_arm_customer, pulled_arm_product = pulled_arm  # unpacking tuple
            self.t += 1
            self.n_samples[pulled_arm_customer, pulled_arm_product] += 1
            self.mu[pulled_arm_customer, pulled_arm_product] = (self.mu[pulled_arm_customer, pulled_arm_product] * (self.n_samples[pulled_arm_customer, pulled_arm_product] - 1) + reward) / self.n_samples[pulled_arm_customer, pulled_arm_product]
            
            # Update observations
            # If `self.rewards_per_arm` is a dictionary with tuple keys
            if pulled_arm not in self.rewards_per_arm:
                self.rewards_per_arm[pulled_arm] = []
            self.rewards_per_arm[pulled_arm].append(reward)


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
    
    