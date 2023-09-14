from learners.ucb_learners.ucb_learner import UCBProbLearner
from change_detection import CUSUM
import numpy as np


class SWUCBProbLearner(UCBProbLearner):
    def __init__(self, n_nodes, n_seeds, window_size, graph_structure=None):
        super().__init__(n_nodes, n_seeds, graph_structure)
        self.window_size = window_size
        self.pulled_edges = []

    def update(self, episode):
        self.pulled_edges.append(episode)
        if len(self.pulled_edges) > self.window_size:
            self.pulled_edges.pop(0)

        # Reset before recomputing based on the sliding window
        self.edge_rewards = np.zeros((self.n_nodes, self.n_nodes))
        self.n_pulls = np.zeros((self.n_nodes, self.n_nodes))

        # Now, compute based on episodes in the sliding window
        for ep in self.pulled_edges:
            for _, _, susceptible, activated in ep:
                self.update_observations(susceptible, activated)

        # Recompute empirical means and confidence
        super().update_confidence()

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0
        self.collected_rewards.append(np.sum(self.edge_rewards))


class CDUCBProbLearner(UCBProbLearner):
    def __init__(self, n_nodes, n_seeds, graph_structure=None, M=100, eps=0.05, h=20):
        super().__init__(n_nodes, n_seeds, graph_structure)
        # CUSUM Change Detection
        self.change_detection = [
            [CUSUM(M=M, eps=eps, h=h) for _ in range(n_nodes)] for _ in range(n_nodes)
        ]
        self.valid_rewards = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]

    def update(self, episode):
        susceptible, activated = episode
        self.update_observations(susceptible, activated)

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
