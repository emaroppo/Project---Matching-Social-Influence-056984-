from learners.ts_learners.ts_learner import TSLearner
import numpy as np
from scipy.optimize import linear_sum_assignment


class TSMatching(TSLearner):
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class=3
    ):
        super().__init__(n_arms)
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.mu = np.zeros(
            (n_customer_classes + 1, n_product_classes + 1)
        )  # include the phantom class
        self.n_samples = np.zeros(
            (n_customer_classes + 1, n_product_classes + 1), dtype=int
        )
        self.rewards_per_arm = {}  # temporary fix
        self.n_products_per_class = n_products_per_class

    def pull_arms(self, customer_classes):
        theta_sample = np.random.normal(self.mu, 1)

        # Extended theta_sample matrix to include phantom class
        extended_theta_sample = np.zeros(
            (self.n_customer_classes + 1, self.n_product_classes + 1)
        )
        extended_theta_sample[
            : self.n_customer_classes + 1, : self.n_product_classes + 1
        ] = theta_sample

        # Repeat entries in available_theta for each product
        available_theta = np.repeat(
            extended_theta_sample[customer_classes, :],
            self.n_products_per_class,
            axis=1,
        )
        row_ind, col_ind = linear_sum_assignment(-available_theta)
        best_arms_global = [
            (
                customer_classes[row]
                if row < len(customer_classes)
                else self.n_customer_classes,
                col // 3,
            )
            for row, col in zip(row_ind, col_ind)
        ]

        return best_arms_global

    def update(self, pulled_arms, rewards):
        self.collected_rewards = np.append(
            self.collected_rewards, np.array(rewards).sum()
        )
        for pulled_arm, reward in zip(pulled_arms, rewards):
            pulled_arm_customer, pulled_arm_product = pulled_arm  # unpacking tuple
            self.t += 1
            self.n_samples[pulled_arm_customer, pulled_arm_product] += 1
            self.mu[pulled_arm_customer, pulled_arm_product] = (
                self.mu[pulled_arm_customer, pulled_arm_product]
                * (self.n_samples[pulled_arm_customer, pulled_arm_product] - 1)
                + reward
            ) / self.n_samples[pulled_arm_customer, pulled_arm_product]

            # Update observations
            # If `self.rewards_per_arm` is a dictionary with tuple keys
            if pulled_arm not in self.rewards_per_arm:
                self.rewards_per_arm[pulled_arm] = []
            self.rewards_per_arm[pulled_arm].append(reward)
