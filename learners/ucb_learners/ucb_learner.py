import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from learners.learner import Learner
from environments.social_environment import SocialEnvironment


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.n_pulls = np.zeros(
            n_arms
        )  # count the number of times each arm has been pulled
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        return np.random.choice(
            np.where(upper_confidence_bound == upper_confidence_bound.max())[0]
        )

    def update(self, pulled_arm, reward):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        self.empirical_means[pulled_arm] = (
            self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward
        ) / self.n_pulls[pulled_arm]
        for a in range(self.n_arms):
            n_samples = max(1, self.n_pulls[a])
            self.confidence[a] = (
                np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf
            )

        self.update_observations(pulled_arm, reward)


class MAUCBLearner(UCBLearner):
    def __init__(self, n_arms, n_seeds):
        super().__init__(n_arms)
        self.n_seeds = n_seeds
        self.n_pulls = np.zeros(
            n_arms
        )  # count the number of times each arm has been pulled

    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        pulled_arms = []
        while len(pulled_arms) < self.n_seeds:
            pulled_arm = np.random.choice(
                np.where(upper_confidence_bound == upper_confidence_bound.max())[0]
            )
            upper_confidence_bound[pulled_arm] = -np.inf
            pulled_arms.append(pulled_arm)

        return pulled_arms

    def update(self, pulled_arms, reward):
        self.t += 1
        for pulled_arm in pulled_arms:
            self.n_pulls[pulled_arm] += 1
            self.empirical_means[pulled_arm] = (
                self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1)
                + reward
            ) / self.n_pulls[pulled_arm]
            n_samples = max(1, self.n_pulls[pulled_arm])
            self.confidence[pulled_arm] = (
                np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf
            )

            self.update_observations(pulled_arm, reward)


class UCBProbLearner:
    def __init__(self, n_nodes, n_seeds, graph_structure=None):
        self.n_nodes = n_nodes
        self.n_seeds = n_seeds
        self.empirical_means = np.zeros((n_nodes, n_nodes))
        self.confidence = np.full((n_nodes, n_nodes), 1000)

        self.graph_structure = graph_structure
        if graph_structure is not None:
            mask = graph_structure == 0
            self.empirical_means[mask] = 0
            self.confidence[mask] = 0

        self.edge_rewards = np.zeros((n_nodes, n_nodes))
        self.n_pulls = np.zeros((n_nodes, n_nodes))
        self.t = 0

        self.collected_rewards = []

    def pull_arm(self):
        sqrt_factor = np.sqrt(max(1, (365 - self.t)) / 365)
        upper_confidence_bound = (
            self.empirical_means + sqrt_factor * self.confidence
        ) / (1 + sqrt_factor * self.confidence)

        # Cap upper confidence bound to 10
        np.clip(upper_confidence_bound, None, 10, out=upper_confidence_bound)

        world_representation = SocialEnvironment(upper_confidence_bound)
        seeds = world_representation.opt_arm(self.n_seeds)
        return seeds

    def update(self, episode):
        self.t += 1

        for step in episode:
            active_nodes, newly_active_nodes, activated_edges = step
            susceptible_edges = np.outer(newly_active_nodes, 1 - active_nodes)

            self.n_pulls += susceptible_edges
            self.edge_rewards += activated_edges

        # Where clause handles the division by zero, so no need for np.divide
        self.empirical_means[self.n_pulls != 0] = (
            self.edge_rewards[self.n_pulls != 0] / self.n_pulls[self.n_pulls != 0]
        )
        self.confidence[self.n_pulls != 0] = np.sqrt(
            2 * np.log(1 + np.sum(self.n_pulls)) / self.n_pulls[self.n_pulls != 0]
        )

        self.confidence[self.graph_structure == 0] = 0
        self.collected_rewards.append(np.sum(activated_edges))


class UCBProbLearner2:
    def __init__(self, n_nodes, n_seeds, graph_structure=None):
        self.n_nodes = n_nodes
        self.n_seeds = n_seeds
        self.empirical_means = np.zeros((n_nodes, n_nodes))
        self.confidence = np.full((n_nodes, n_nodes), 1000)
        self.graph_structure = graph_structure
        if graph_structure is not None:
            mask = graph_structure == 0
            self.empirical_means[mask] = 0
            self.confidence[mask] = 0

        self.edge_rewards = np.zeros((n_nodes, n_nodes))
        self.n_pulls = np.zeros((n_nodes, n_nodes))
        self.t = 0
        self.collected_rewards = []

        # Store all episodes
        self.episodes = []

    def pull_arm(self):
        sqrt_factor = np.sqrt(max(1, (365 - self.t)) / 365)
        upper_confidence_bound = (
            self.empirical_means + sqrt_factor * self.confidence
        ) / (1 + sqrt_factor * self.confidence)
        np.clip(upper_confidence_bound, None, 10, out=upper_confidence_bound)

        world_representation = SocialEnvironment(upper_confidence_bound)
        seeds = world_representation.opt_arm(self.n_seeds)
        return seeds

    def update_observations(self, episode):
        self.episodes.append(episode)
        self.t += 1
        for step in episode:
            active_nodes, newly_active_nodes, activated_edges = step
            susceptible_edges = np.outer(newly_active_nodes, 1 - active_nodes)

            self.n_pulls += susceptible_edges
            self.edge_rewards += activated_edges

    def update(self, episode):
        self.update_observations(episode)

        # Recompute empirical means and confidence based on all episodes
        self.empirical_means[self.n_pulls != 0] = (
            self.edge_rewards[self.n_pulls != 0] / self.n_pulls[self.n_pulls != 0]
        )
        self.confidence[self.n_pulls != 0] = np.sqrt(
            2 * np.log(1 + np.sum(self.n_pulls)) / self.n_pulls[self.n_pulls != 0]
        )

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0
        self.collected_rewards.append(np.sum(self.edge_rewards))
