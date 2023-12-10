from learners.matching_learner import MatchingLearner
import numpy as np


class TSMatching(MatchingLearner):
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class
    ):
        super().__init__(
            n_arms, n_customer_classes, n_product_classes, n_products_per_class
        )
        self.mu = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.lambda_ = np.ones((n_customer_classes + 1, n_product_classes + 1))
        self.alpha = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.beta = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.n_samples = np.zeros(
            (n_customer_classes + 1, n_product_classes + 1), dtype=int
        )

    def _compute_extended_theta_sample(self):
        theta_sample = np.zeros_like(self.mu)
        for i in range(self.mu.shape[0]):
            for j in range(self.mu.shape[1]):
                sigma2 = 1.0 / np.random.gamma(self.alpha[i, j], 1.0 / self.beta[i, j])
                theta_sample[i, j] = np.random.normal(
                    self.mu[i, j], np.sqrt(sigma2 / self.lambda_[i, j])
                )
        extended_theta_sample = np.zeros(
            (self.n_customer_classes + 1, self.n_product_classes + 1)
        )
        extended_theta_sample[
            : self.n_customer_classes + 1, : self.n_product_classes + 1
        ] = theta_sample
        return extended_theta_sample

    def pull_arm(self, customer_classes, context=False):
        theta_sample = self._compute_extended_theta_sample()
        best_arms_global = super().pull_arm(theta_sample, customer_classes, context)
        return best_arms_global

    def update(self, pulled_arms, rewards):
        super().update_observations(pulled_arms, rewards)
        self.collected_rewards = np.append(
            self.collected_rewards, np.array(rewards).sum()
        )
        for pulled_arm, reward in zip(pulled_arms, rewards):
            self._update_arm_parameters(pulled_arm, reward)

    def _update_arm_parameters(self, pulled_arm, reward):
        pulled_arm_customer, pulled_arm_product = pulled_arm
        n = self.n_samples[pulled_arm_customer, pulled_arm_product]

        # NIG parameter updates
        cur_mu = self.mu[pulled_arm_customer, pulled_arm_product]
        self.mu[pulled_arm_customer, pulled_arm_product] = (
            self.lambda_[pulled_arm_customer, pulled_arm_product] * cur_mu + reward
        ) / (self.lambda_[pulled_arm_customer, pulled_arm_product] + 1)
        self.lambda_[pulled_arm_customer, pulled_arm_product] += 1
        self.alpha[pulled_arm_customer, pulled_arm_product] += 0.5
        self.beta[pulled_arm_customer, pulled_arm_product] += (
            self.lambda_[pulled_arm_customer, pulled_arm_product]
            * (reward - cur_mu) ** 2
        ) / (2.0 * (self.lambda_[pulled_arm_customer, pulled_arm_product] + 1))

        self.n_samples[pulled_arm_customer, pulled_arm_product] += 1
