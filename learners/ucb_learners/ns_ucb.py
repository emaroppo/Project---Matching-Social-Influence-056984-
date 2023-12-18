from learners.ucb_learners.ucb_prob_learner import UCBProbLearner
import numpy as np
from utils.change_detection import CUSUM


class SWUCBProbLearner(UCBProbLearner):
    @classmethod
    def sensitivity_analysis(cls, parameters, n_nodes, n_seeds, graph_structure):
        models = [
            cls(
                n_nodes, n_seeds=n_seeds, window_size=i, graph_structure=graph_structure
            )
            for i in parameters
        ]
        for model, parameter in zip(models, parameters):
            model.name = f"SW-UCB (window_size={parameter})"

        return models

    def __init__(self, n_nodes, n_seeds, window_size, graph_structure=None):
        super().__init__(n_nodes, n_seeds, graph_structure)
        self.window_size = window_size
        self.current_index = 0
        self.susceptible_edges = []
        self.activated_edges = []

    def update(self, episode):
        susceptible, activated = episode
        self.susceptible_edges.append(susceptible)
        self.activated_edges.append(activated)

        self.current_index += 1
        start_index = max(0, self.current_index - self.window_size)

        # Convert lists to NumPy arrays for calculations within the window
        for i in range(start_index, self.current_index):
            susceptible_array = np.array(self.susceptible_edges[i])
            activated_array = np.array(self.activated_edges[i])
            super().update((susceptible_array, activated_array))


class CDUCBProbLearner(UCBProbLearner):
    @classmethod
    def sensitivity_analysis(cls, parameters, n_nodes, n_seeds, graph_structure):
        models = [cls(eps=i, n_nodes=n_nodes, n_seeds=n_seeds) for i in parameters]
        for model, parameter in zip(models, parameters):
            model.name = f"CD-UCB (eps={parameter})"
        return models
    def __init__(self, n_nodes, n_seeds, graph_structure=None, M=100, eps=0.05, h=20):
        super().__init__(n_nodes, n_seeds, graph_structure)
        self.change_detection = CUSUM(n_nodes, M, eps, h)
        self.valid_since_episode = np.zeros((n_nodes, n_nodes), dtype=int)

    def update(self, episode):
        susceptible, activated = episode
        self.susceptible_edges.append(susceptible)
        self.activated_edges.append(activated)
        super().update_observations(susceptible, activated)

        # Vectorized change detection
        # Convert the entire edge_rewards to a NumPy array for vectorized operation
        edge_rewards_array = np.array(self.edge_rewards)
        changes_detected = self.change_detection.update(edge_rewards_array)
        self.valid_since_episode[changes_detected] = self.t
        self.change_detection.reset(np.where(changes_detected))

        # Efficiently update means and confidence for each node
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                valid_range = slice(self.valid_since_episode[i][j], self.t)
                valid_rewards = np.array(self.activated_edges)[valid_range, i, j]
                valid_susceptibles = np.array(self.susceptible_edges)[valid_range, i, j]

                if valid_rewards.size > 0:
                    self.empirical_means[i][j] = np.mean(valid_rewards)
                    n_samples = np.sum(valid_susceptibles)
                    if n_samples > 0:
                        self.confidence[i][j] = np.sqrt(2 * np.log(self.t) / n_samples)
                    else:
                        self.confidence[i][j] = 10e4
