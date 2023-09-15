import numpy as np
from learners.learner import Learner
from environments.social_environment import SocialEnvironment


class UCBLearner(Learner):
    def __init__(self, n_arms, matrix_form=False):
        super().__init__(n_arms)
        shape = (n_arms, n_arms) if matrix_form else n_arms
        self.empirical_means = np.zeros(shape)
        self.n_pulls = np.zeros(shape)
        self.confidence = np.full(shape, 10e4)

    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        return np.random.choice(
            np.where(upper_confidence_bound == upper_confidence_bound.max())[0]
        )

    def update_observations(self, pulled_arm, reward):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_confidence(self):
        # where n_pulls>0, self.confidence is updated
        self.confidence[self.n_pulls != 0] = np.sqrt(
            2 * np.log(self.t) / self.n_pulls[self.n_pulls != 0]
        )

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = (
            (self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward)
            / self.n_pulls[pulled_arm]
            if self.n_pulls[pulled_arm] != 0
            else 0
        )
        self.update_confidence()


class UCBProbLearner(UCBLearner):
    def __init__(self, n_nodes, n_seeds, graph_structure=None):
        super().__init__(n_nodes, matrix_form=True)
        self.n_seeds = n_seeds
        self.graph_structure = graph_structure
        self.n_nodes = n_nodes
        if graph_structure is not None:
            mask = graph_structure == 0
            self.empirical_means[mask] = 0
            self.confidence[mask] = 0

        self.edge_rewards = np.zeros((n_nodes, n_nodes))
        self.collected_rewards = []
        self.susceptible_edges = []
        self.activated_edges = []

    def pull_arm(self):
        sqrt_factor = max(1, (365 - self.t)) / 365
        upper_confidence_bound = (
            self.empirical_means + sqrt_factor * self.confidence
        ) / (1 + sqrt_factor * self.confidence)
        np.clip(upper_confidence_bound, None, 10, out=upper_confidence_bound)

        world_representation = SocialEnvironment(upper_confidence_bound)
        seeds = world_representation.opt_arm(self.n_seeds)
        return seeds

    def update_observations(self, susceptible_edges, activated_edges):
        self.susceptible_edges.append(susceptible_edges)
        self.activated_edges.append(activated_edges)

    def update_confidence(self):
        self.confidence[self.n_pulls != 0] = np.sqrt(
            2 * np.log(np.sum(self.n_pulls)) / self.n_pulls[self.n_pulls != 0]
        )

    def update(self, episode):
        self.t += 1
        susceptible, activated = episode
        self.update_observations(susceptible, activated)

        self.n_pulls += susceptible
        self.edge_rewards += activated

        self.empirical_means[self.n_pulls != 0] = (
            self.edge_rewards[self.n_pulls != 0] / self.n_pulls[self.n_pulls != 0]
        )

        self.update_confidence()  # use the confidence update from parent class

        if self.graph_structure is not None:
            self.confidence[self.graph_structure == 0] = 0

        self.collected_rewards.append(np.sum(self.edge_rewards))
