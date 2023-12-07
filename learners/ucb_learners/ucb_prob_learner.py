from learners.prob_learner import ProbLearner
import numpy as np


class UCBProbLearner(ProbLearner):
    def __init__(self, n_nodes, n_seeds, graph_structure=None):
        super().__init__(n_nodes, n_seeds, graph_structure)

        self.empirical_means = np.zeros((n_nodes, n_nodes))
        self.n_pulls = np.zeros((n_nodes, n_nodes))
        self.confidence = np.full((n_nodes, n_nodes), 10e4)

        if graph_structure is not None:
            mask = graph_structure == 0
            self.empirical_means[mask] = 0
            self.confidence[mask] = 0

    def pull_arm(self):
        sqrt_factor = max(1, (365 - self.t)) / 365
        upper_confidence_bound = (
            self.empirical_means + sqrt_factor * self.confidence
        ) / (1 + sqrt_factor * self.confidence)
        np.clip(upper_confidence_bound, None, 10, out=upper_confidence_bound)
        return super().pull_arm(upper_confidence_bound)

    def update_confidence(self):
        nonzero_pulls = self.n_pulls != 0
        total_pulls = np.sum(self.n_pulls[nonzero_pulls])
        if total_pulls > 0:
            self.confidence[nonzero_pulls] = np.sqrt(
                2 * np.log(total_pulls) / self.n_pulls[nonzero_pulls]
            )

    def update(self, episode):
        susceptible, activated = episode
        super().update_observations(susceptible, activated)

        # Increment number of pulls and update rewards
        self.n_pulls += susceptible
        self.edge_rewards += activated

        # Update empirical means for non-zero pulls
        nonzero_pulls = self.n_pulls != 0
        self.empirical_means[nonzero_pulls] = (
            self.edge_rewards[nonzero_pulls] / self.n_pulls[nonzero_pulls]
        )

        # Update confidence values
        self.update_confidence()

        # Handle graph structure if present
        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0
