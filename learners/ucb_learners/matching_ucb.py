from learners.matching_learner import MatchingLearner
import numpy as np


class UCBMatching(MatchingLearner):
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class
    ):
        super().__init__(
            n_arms, n_customer_classes, n_product_classes, n_products_per_class
        )
        self.empirical_means = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.n_pulls = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.confidence = np.full((n_customer_classes + 1, n_product_classes + 1), 10e4)

    def pull_arm(self, active_customers, customer_classes):
        # Compute upper confidence bounds
        upper_confidence_bound = self.empirical_means + self.confidence
        indices = super().pull_arm(
            upper_confidence_bound, active_customers, customer_classes
        )
        return indices

    def update(self, pulled_arms, rewards):
        super().update_observations(pulled_arms, rewards)
        self.update_empirical_means(pulled_arms, rewards)
        self.update_confidence()

    def update_empirical_means(self, pulled_arms, rewards):
        for pulled_arm, reward in zip(pulled_arms, rewards):
            pulled_arm_customer, pulled_arm_product = pulled_arm
            self.n_pulls[pulled_arm_customer, pulled_arm_product] += 1
            self.empirical_means[pulled_arm_customer, pulled_arm_product] = (
                self.empirical_means[pulled_arm_customer, pulled_arm_product]
                * (self.n_pulls[pulled_arm_customer, pulled_arm_product] - 1)
                + reward
            ) / self.n_pulls[pulled_arm_customer, pulled_arm_product]

    def update_confidence(self):
        for customer_class in range(self.n_customer_classes + 1):
            for product_class in range(self.n_product_classes + 1):
                if self.n_pulls[customer_class, product_class] > 0:
                    self.confidence[customer_class, product_class] = np.sqrt(
                        2 * np.log(self.t) / self.n_pulls[customer_class, product_class]
                    )
