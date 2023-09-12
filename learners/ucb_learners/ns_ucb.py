from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ucb_learners.ucb_learner import UCBProbLearner2
from environments.social_environment import SocialEnvironment
from change_detection import CUSUM
import numpy as np


class SWUCBProbLearner(UCBProbLearner2):
    def __init__(self, n_nodes, n_seeds, window_size, graph_structure=None):
        super().__init__(n_nodes, n_seeds, graph_structure)
        self.window_size = window_size
        self.pulled_edges = []

    def update(self, episode):
        self.update_observations(episode)
        self.pulled_edges.append(episode)

        if len(self.pulled_edges) > self.window_size:
            self.pulled_edges.pop(0)

        # Reset the edge rewards and n_pulls before recomputing based on the sliding window
        self.edge_rewards = np.zeros((self.n_nodes, self.n_nodes))
        self.n_pulls = np.zeros((self.n_nodes, self.n_nodes))

        # Now, compute based on episodes in the sliding window
        for ep in self.pulled_edges:
            for step in ep:
                active_nodes, newly_active_nodes, activated_edges = step
                susceptible_edges = np.outer(newly_active_nodes, 1 - active_nodes)
                self.n_pulls += susceptible_edges
                self.edge_rewards += activated_edges

        # Recompute empirical means and confidence based on the episodes in the window
        self.empirical_means[self.n_pulls != 0] = (
            self.edge_rewards[self.n_pulls != 0] / self.n_pulls[self.n_pulls != 0]
        )
        self.confidence[self.n_pulls != 0] = np.sqrt(
            2 * np.log(1 + np.sum(self.n_pulls)) / self.n_pulls[self.n_pulls != 0]
        )

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0
        self.collected_rewards.append(np.sum(self.edge_rewards))


# da finire
class CUMSUMUCBMatching(UCBMatching):
    def __init__(
        self,
        n_arms,
        n_rows,
        n_cols,
        n_products_per_class=3,
        M=100,
        eps=0.05,
        h=20,
        alpha=0.01,
    ):
        super().__init__(n_arms, n_rows, n_cols, n_products_per_class)
        self.change_detection = [CUSUM(M=M, eps=eps, h=h) for _ in range(n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arms(self):
        if np.random.binomial(1, 1 - self.alpha):
            upper_confidence_bound = self.empirical_means + self.confidence
            upper_confidence_bound[np.isinf(upper_confidence_bound)] = 1e3
            row_ind, col_ind = linear_sum_assignment(
                -upper_confidence_bound.reshape(self.n_rows, self.n_cols)
            )
            return row_ind, col_ind

        else:
            costs_random = np.random.randint(0, 10, (self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update(self, pulled_arms, reward):
        self.t += 1
        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        self.update_observations(pulled_arms_flat, reward)

        for pulled_arm, reward in zip(pulled_arms_flat, reward):
            self.valid_rewards_per_arm[pulled_arm].append(reward)
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arm[pulled_arm] = []
                self.change_detection[pulled_arm].reset()
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(
                self.valid_rewards_per_arm[pulled_arm]
            )
        total_valid_samples = sum([len(i) for i in self.valid_rewards_per_arm])
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arm[a])
            self.confidence[a] = (
                np.sqrt(2 * np.log(total_valid_samples) / n_samples)
                if n_samples > 0
                else np.inf
            )


class UCBProbLearner2WithChangeDetection:
    def __init__(self, n_nodes, n_seeds, graph_structure=None, M=100, eps=0.05, h=20):
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

        # CUSUM Change Detection
        self.change_detection = [
            [CUSUM(M=M, eps=eps, h=h) for _ in range(n_nodes)] for _ in range(n_nodes)
        ]
        self.valid_rewards = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]

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

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.change_detection[i][j].update(self.edge_rewards[i][j]):
                    self.valid_rewards[i][j] = []
                    self.change_detection[i][j].reset()

                self.valid_rewards[i][j].append(self.edge_rewards[i][j])
                self.empirical_means[i][j] = np.mean(self.valid_rewards[i][j])

        total_valid_samples = sum(
            [len(row) for sublist in self.valid_rewards for row in sublist]
        )

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                n_samples = len(self.valid_rewards[i][j])
                self.confidence[i][j] = (
                    np.sqrt(2 * np.log(total_valid_samples) / n_samples)
                    if n_samples > 0
                    else np.inf
                )

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0

        self.collected_rewards.append(np.sum(self.edge_rewards))
