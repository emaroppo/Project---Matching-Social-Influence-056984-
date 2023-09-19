from learners.ucb_learners.ucb_learner import UCBLearner
import numpy as np
from scipy.optimize import linear_sum_assignment


class UCBMatching(UCBLearner):
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class
    ):
        super().__init__(n_arms)
        self.n_pulls = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.confidence = np.full((n_customer_classes + 1, n_product_classes + 1), 10e4)
        self.n_customer_classes = n_customer_classes
        self.n_products_per_class = n_products_per_class
        self.n_product_classes = n_product_classes
        self.empirical_means = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.rewards_per_arm = {}  # temporary fix
        self.collected_rewards = np.array([])  # Initialize this attribute

    def pull_arm(self, active_customers, customer_classes):
        upper_conf = self.empirical_means + self.confidence

        extended_upper_conf = np.repeat(upper_conf, self.n_products_per_class, axis=1)
        available_upper_conf = extended_upper_conf[customer_classes, :]
        row_ind, col_ind = linear_sum_assignment(-available_upper_conf)

        best_arms_global = [
            (
                active_customers[row],
                customer_classes[row]
                if row < len(customer_classes)
                else self.n_customer_classes,
                col // 3,
            )
            for row, col in zip(row_ind, col_ind)
        ]

        return best_arms_global

    def update(self, pulled_arms, rewards):
        self.collected_rewards = np.append(self.collected_rewards, np.sum(rewards))
        for pulled_arm, reward in zip(pulled_arms, rewards):
            _, pulled_arm_customer, pulled_arm_product = pulled_arm  # unpacking tuple
            self.t += 1

            self.n_pulls[pulled_arm_customer, pulled_arm_product] += 1
            self.empirical_means[pulled_arm_customer, pulled_arm_product] = (
                self.empirical_means[pulled_arm_customer, pulled_arm_product]
                * (self.n_pulls[pulled_arm_customer, pulled_arm_product] - 1)
                + reward
            ) / self.n_pulls[pulled_arm_customer, pulled_arm_product]
            # Update observations
            # If `self.rewards_per_arm` is a dictionary with tuple keys
            if pulled_arm not in self.rewards_per_arm:
                self.rewards_per_arm[pulled_arm] = []
            self.rewards_per_arm[pulled_arm].append(reward)
            # Update confidence
            self.confidence[pulled_arm_customer, pulled_arm_product] = np.sqrt(
                2
                * np.log(self.t)
                / self.n_pulls[pulled_arm_customer, pulled_arm_product]
            )
