from environments.environment import Environment
from environments.social_environment import SocialEnvironment
from environments.matching_environment import MatchingEnvironment
from environments.matching_environment_context import MatchingEnvironmentContext
import numpy as np


class JointEnvironment(Environment):
    def __init__(
        self,
        probabilities,
        reward_parameters,
        node_classes,
        opt_value=None,
        context=False,
    ):
        self.social_environment = SocialEnvironment(probabilities)
        if context:
            self.matching_environment = MatchingEnvironmentContext(
                reward_parameters, node_classes
            )
        else:
            self.matching_environment = MatchingEnvironment(reward_parameters)
        self.node_classes = node_classes
        self.t = 0
        self.opt_value = opt_value
        # Additional attributes for metrics
        self.active_nodes = None
        self.expected_rewards = np.empty((0,))
        self.optimal_rewards = np.empty((0,))

    def round(self, pulled_arms):
        social_reward, active_nodes = self.social_environment.round(
            pulled_arms, return_active_nodes=True
        )
        # Track active nodes
        self.active_nodes = np.where(active_nodes == 1)[0]
        customer_classes = [self.node_classes[i] for i in self.active_nodes]
        matching_reward = self.matching_environment.round(pulled_arms, customer_classes)
        # Collect rewards for metrics
        self.expected_rewards = np.append(
            self.expected_rewards, [np.sum(matching_reward)]
        )
        return social_reward, matching_reward, self.active_nodes

    def opt(self, n_seeds=3, prod_classes=[0, 1, 2] * 3, n_exp=1000):
        if self.opt_value is None:
            opt_seeds = self.social_environment.opt(n_seeds)[2]

            rewards_per_exp = np.array([])

            for i in range(n_exp):
                active_nodes = self.social_environment.round(opt_seeds, joint=True)
                active_nodes = np.where(active_nodes == 1)[0]
                customer_classes = [self.node_classes[i] for i in active_nodes]
                matching_reward = self.matching_environment.opt(
                    customer_classes, prod_classes
                )
                rewards_per_exp = np.append(rewards_per_exp, matching_reward)

            self.opt_value = (
                np.mean(rewards_per_exp),
                np.std(rewards_per_exp),
                opt_seeds,
            )
        return self.opt_value
