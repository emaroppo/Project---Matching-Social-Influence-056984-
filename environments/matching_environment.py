from environments.environment import Environment
from scipy.optimize import linear_sum_assignment
import numpy as np


def generate_reward(mean, std_dev):
    return lambda: np.random.normal(mean, std_dev)


class MatchingEnvironment(Environment):
    def __init__(self, reward_parameters, constant_nodes=True):
        self.optimal_rewards = np.empty((0, 2))
        # Stack the reward parameters along a new third axis
        reward_parameters = np.stack(reward_parameters, axis=-1)
        if constant_nodes:
            reward_matrix = np.random.normal(
                reward_parameters[..., 0], reward_parameters[..., 1]
            )
        else:
            reward_matrix = np.empty(
                (*reward_parameters.shape[:-1],),
                dtype=object,
            )
            for i in range(reward_parameters.shape[0]):
                for j in range(reward_parameters.shape[1]):
                    reward_matrix[i, j] = generate_reward(
                        reward_parameters[i, j, 0], reward_parameters[i, j, 1]
                    )

        # add a row and a column of zeros to the reward matrix to represent the case in which no match is made
        reward_matrix = np.hstack(
            (reward_matrix, np.zeros((reward_matrix.shape[0], 1)))
        )
        reward_matrix = np.vstack(
            (reward_matrix, np.zeros((1, reward_matrix.shape[1])))
        )
        self.reward_matrix = reward_matrix

        self.reward_parameters = reward_parameters

        self.n_arms = reward_matrix.size
        self.t = 0

    def round(self, pulled_arms):
        rewards = [
            self.reward_matrix[pulled_arm[0], pulled_arm[1]]
            for pulled_arm in pulled_arms
        ]

        # Iterate through all cells of rewards; if a cell is callable, call it and replace it with the result
        for i in range(len(rewards)):
            if callable(rewards[i]):
                rewards[i] = rewards[i]()

        # Return tuples of (customer_id, reward)
        return [(pulled_arm[0], rewards[i]) for i, pulled_arm in enumerate(pulled_arms)]

    def opt(self, customer_classes, product_classes):
        n_customers = len(customer_classes)
        n_products = len(product_classes)

        if n_customers > n_products:
            # pad products with 3s
            product_classes = [*product_classes, *[3] * (n_customers - n_products)]
        elif n_products > n_customers:
            # pad customers with 3s
            customer_classes = [*customer_classes, *[3] * (n_products - n_customers)]

        expected_rewards = self.reward_parameters.copy()
        expected_rewards = np.hstack(
            (expected_rewards, np.zeros((expected_rewards.shape[0], 1, 2)))
        )
        expected_rewards = np.vstack(
            (expected_rewards, np.zeros((1, expected_rewards.shape[1], 2)))
        )

        new_reward_matrix = np.zeros((n_customers, n_products, 2))

        for i in range(n_customers):
            for j in range(n_products):
                # Store both mean and std in new_reward_matrix
                new_reward_matrix[i, j] = expected_rewards[
                    customer_classes[i], product_classes[j]
                ]

        # find optimal assignment based on the mean rewards
        row_ind, col_ind = linear_sum_assignment(-new_reward_matrix[..., 0])

        # compute reward of optimal assignment
        reward = new_reward_matrix[row_ind, col_ind, 0].sum()

        # compute variance of optimal assignment
        variances = new_reward_matrix[row_ind, col_ind, 1] ** 2
        total_variance = variances.sum()

        # compute standard deviation of optimal assignment
        std_dev = np.sqrt(total_variance)

        # return a 2D array with a single row
        return np.array([[reward, std_dev]])

    def expected_reward(self, pulled_arm):
        return self.reward_matrix[pulled_arm[0], pulled_arm[1]]
