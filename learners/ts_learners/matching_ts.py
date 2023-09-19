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


class TSMatching2:
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class=3
    ):
        self.n_arms = n_arms
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.mu = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.lambda_ = np.ones((n_customer_classes + 1, n_product_classes + 1))
        self.alpha = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.beta = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.n_samples = np.zeros(
            (n_customer_classes + 1, n_product_classes + 1), dtype=int
        )
        self.collected_rewards = np.array([])
        self.n_products_per_class = n_products_per_class

    def pull_arm(self, active_customers, customer_classes):
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
        available_theta = np.repeat(
            extended_theta_sample[customer_classes, :],
            self.n_products_per_class,
            axis=1,
        )
        row_ind, col_ind = linear_sum_assignment(-available_theta)

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
        self.collected_rewards = np.append(
            self.collected_rewards, np.array(rewards).sum()
        )
        for pulled_arm, reward in zip(pulled_arms, rewards):
            _, pulled_arm_customer, pulled_arm_product = pulled_arm
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


class TSMatching3:
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class=3
    ):
        self.n_arms = n_arms
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.mu = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.lambda_ = np.ones((n_customer_classes + 1, n_product_classes + 1))
        self.alpha = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.beta = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.n_samples = np.zeros(
            (n_customer_classes + 1, n_product_classes + 1), dtype=int
        )
        self.collected_rewards = np.array([])
        self.n_products_per_class = n_products_per_class

    def pull_arm(self, active_customers, customer_classes):
        active_customers = active_customers.reshape(-1)
        customer_classes = customer_classes.reshape(-1)
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
        available_theta = np.repeat(
            extended_theta_sample[customer_classes, :],
            self.n_products_per_class,
            axis=1,
        )
        row_ind, col_ind = linear_sum_assignment(-available_theta)

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
        self.collected_rewards = np.append(
            self.collected_rewards, np.array(rewards).sum()
        )
        for pulled_arm, reward in zip(pulled_arms, rewards):
            _, pulled_arm_customer, pulled_arm_product = pulled_arm
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


class TSMatching4:
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class=3
    ):
        self.n_arms = n_arms
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.mu = np.zeros((n_customer_classes + 1, n_product_classes + 1))
        self.lambda_ = np.ones((n_customer_classes + 1, n_product_classes + 1))
        self.alpha = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.beta = np.ones((n_customer_classes + 1, n_product_classes + 1)) * 0.5
        self.n_samples = np.zeros(
            (n_customer_classes + 1, n_product_classes + 1), dtype=int
        )
        self.collected_rewards = np.array([])
        self.n_products_per_class = n_products_per_class
        self.current_classes = []  # Keep track of current leaves/classes

    def resize_arms(self, new_context_structure):
        current_splits = set(
            tuple(tuple(s) for s in sorted(leaf.split)) for leaf in self.current_classes
        )
        new_splits = set(
            tuple(tuple(s) for s in sorted(leaf.split))
            for leaf in new_context_structure.leaves
        )

        obsolete_splits = current_splits - new_splits
        new_added_splits = new_splits - current_splits

        # if current classes is empty, remove the first row of mu, lambda, alpha, beta, n_samples
        if len(self.current_classes) == 0:
            self.mu = np.delete(self.mu, 0, axis=0)
            self.lambda_ = np.delete(self.lambda_, 0, axis=0)
            self.alpha = np.delete(self.alpha, 0, axis=0)
            self.beta = np.delete(self.beta, 0, axis=0)
            self.n_samples = np.delete(self.n_samples, 0, axis=0)

        # Remove obsolete classes from the bandit
        for split in obsolete_splits:
            idx = next(
                i
                for i, leaf in enumerate(self.current_classes)
                if tuple(sorted(leaf.split)) == split
            )

            self.mu = np.delete(self.mu, idx, axis=0)
            self.lambda_ = np.delete(self.lambda_, idx, axis=0)
            self.alpha = np.delete(self.alpha, idx, axis=0)
            self.beta = np.delete(self.beta, idx, axis=0)
            self.n_samples = np.delete(self.n_samples, idx, axis=0)

        # Add new classes to the bandit
        for _ in new_added_splits:
            self.mu = np.vstack([self.mu, np.zeros((1, self.n_product_classes + 1))])
            self.lambda_ = np.vstack(
                [self.lambda_, np.ones((1, self.n_product_classes + 1))]
            )
            self.alpha = np.vstack(
                [self.alpha, np.ones((1, self.n_product_classes + 1)) * 0.5]
            )
            self.beta = np.vstack(
                [self.beta, np.ones((1, self.n_product_classes + 1)) * 0.5]
            )
            self.n_samples = np.vstack(
                [self.n_samples, np.zeros((1, self.n_product_classes + 1), dtype=int)]
            )

        # Update the current classes list with the leaves of the new context structure
        self.current_classes = new_context_structure.leaves
        self.n_customer_classes = len(self.current_classes)

    def pull_arms(self, active_customers, customer_classes):
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
        available_theta = np.repeat(
            extended_theta_sample[customer_classes, :],
            self.n_products_per_class,
            axis=1,
        )
        row_ind, col_ind = linear_sum_assignment(-available_theta)

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
        self.collected_rewards = np.append(
            self.collected_rewards, np.array(rewards).sum()
        )
        for pulled_arm, reward in zip(pulled_arms, rewards):
            active_customer, pulled_arm_customer, pulled_arm_product = pulled_arm
            _, reward = reward
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
