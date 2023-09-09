import numpy as np
from environments.joint_environment import JointEnvironment

node_features = np.random.binomial(1, 0.3, (30, 10))


class ContextGeneration:
    def __init__(self) -> None:
        customer_features = np.random.binomial(1, 0.3, (30, 2))  # list of customers
        assignments = np.random.randint(0, 3, 30)  # list of customers/products
        rewards = np.random.normal(
            0, 1, (30, 3)
        )  # list of rewards for each customer/product
        # compute expected reward for each feature/product pair
        expected_rewards = np.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                expected_rewards[i][j] = np.mean(rewards[assignments == j, i])
        # compute lower confidence bound for each feature/product pair
        lower_confidence_bound = np.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                lower_confidence_bound[i][j] = expected_rewards[i][j] - np.sqrt(
                    np.log(1) / 1
                )
        # compute lower confidence bound of probability occurence for each feature
        lower_confidence_bound_prob = np.zeros(2)
        for i in range(2):
            lower_confidence_bound_prob[i] = np.mean(lower_confidence_bound[i])
