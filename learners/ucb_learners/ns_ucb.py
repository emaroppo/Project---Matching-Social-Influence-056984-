from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
from utils.change_detection import CUSUM
import numpy as np


class SWUCBProbLearner(UCBProbLearner):
    @classmethod
    def sensitivity_analysis(cls, parameters, n_nodes, n_seeds, graph_structure):
        return [cls(n_nodes, n_seeds=n_seeds, window_size=i, graph_structure=graph_structure) for i in parameters] 

    def __init__(self, n_nodes, n_seeds, window_size, graph_structure=None):
        super().__init__(n_nodes, n_seeds, graph_structure)
        self.window_size = window_size
        self.current_index = 0

    def update(self, episode):
        susceptible, activated = episode
        self.susceptible_edges.append(susceptible)
        self.activated_edges.append(activated)

        self.current_index += 1
        start_index = max(0, self.current_index - self.window_size)

        # Reset statistics before recomputing based on the sliding window
        self.edge_rewards = np.zeros((self.n_nodes, self.n_nodes))
        self.n_pulls = np.zeros((self.n_nodes, self.n_nodes))

        # Now, compute based on episodes in the sliding window
        for i in range(start_index, self.current_index):
            self.n_pulls += self.susceptible_edges[i]
            self.edge_rewards += self.activated_edges[i]

        # Adjust empirical means based on episodes in the window
        self.empirical_means[self.n_pulls != 0] = (
            self.edge_rewards[self.n_pulls != 0] / self.n_pulls[self.n_pulls != 0]
        )

        # Update confidence
        self.update_confidence()

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0

        self.collected_rewards= np.append(self.collected_rewards, (np.sum(self.edge_rewards)))


class CDUCBProbLearner(UCBProbLearner):

    @classmethod
    def sensitivity_analysis(cls, parameters, n_nodes, n_seeds, graph_structure):
        return [cls(eps=i, n_nodes=n_nodes, n_seeds=n_seeds) for i in parameters] 


    def __init__(self, n_nodes, n_seeds, graph_structure=None, M=100, eps=0.05, h=20):
        super().__init__(n_nodes, n_seeds, graph_structure)
        # CUSUM Change Detection
        self.change_detection = [
            [CUSUM(M=M, eps=eps, h=h) for _ in range(n_nodes)] for _ in range(n_nodes)
        ]
        # Index from which rewards and pulls are valid since last change point for each edge
        self.valid_since_episode = np.zeros((n_nodes, n_nodes), dtype=int)

    def update(self, episode):
        self.t += 1
        susceptible, activated = episode
        self.update_observations(susceptible, activated)

        total_valid_susceptibles = 0

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.change_detection[i][j].update(self.edge_rewards[i][j]):
                    self.valid_since_episode[i][j] = self.t
                    self.change_detection[i][j].reset()

                # Fetch valid rewards and susceptibles from the history
                valid_rewards = [
                    self.activated_edges[k][i][j]
                    for k in range(self.valid_since_episode[i][j], self.t)
                ]
                valid_susceptibles = [
                    self.susceptible_edges[k][i][j]
                    for k in range(self.valid_since_episode[i][j], self.t)
                ]

                self.empirical_means[i][j] = (
                    np.mean(valid_rewards) if valid_rewards else 0
                )
                n_samples = sum(valid_susceptibles)
                total_valid_susceptibles += n_samples

                self.confidence[i][j] = (
                    np.sqrt(2 * np.log(total_valid_susceptibles) / n_samples)
                    if n_samples > 0
                    else 10e4
                )

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0

        self.collected_rewards.append(np.sum(self.edge_rewards))
