import numpy as np
from learners.base.learner import Learner
from environments.social_environment import SocialEnvironment


class EXP3ProbLearner(Learner):
    def __init__(self, n_nodes, n_seeds, graph_structure=None, gamma=0.05):
        super().__init__(n_nodes)
        self.gamma = gamma
        self.n_seeds = n_seeds
        self.graph_structure = graph_structure
        self.n_nodes = n_nodes

        # Initialize matrix form of weights
        self.weights = np.ones((n_nodes, n_nodes))

        if graph_structure is not None:
            self.weights[graph_structure == 0] = 0

        self.edge_rewards = np.zeros((n_nodes, n_nodes))
        self.n_pulls = np.zeros((n_nodes, n_nodes))
        self.collected_rewards = []
        self.susceptible_edges = []
        self.activated_edges = []

    def get_probabilities(self):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + (
            self.gamma / (self.n_nodes**2)
        )
        return probs

    def pull_arm(self):
        probs = self.get_probabilities()
        world_representation = SocialEnvironment(probs)
        seeds = world_representation.opt_arm(self.n_seeds)
        return seeds

    def update_observations(self, susceptible_edges, activated_edges):
        self.susceptible_edges.append(susceptible_edges)
        self.activated_edges.append(activated_edges)

    def update(self, episode):
        self.t += 1
        susceptible, activated = episode
        self.update_observations(susceptible, activated)

        self.n_pulls += susceptible
        self.edge_rewards += activated

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.n_pulls[i][j] != 0:
                    estimated_reward = (
                        self.edge_rewards[i][j] / self.get_probabilities()[i][j]
                    )
                    self.weights[i][j] *= np.exp(
                        (self.gamma / (self.n_nodes**2)) * estimated_reward
                    )

        if self.graph_structure is not None:
            self.weights[self.graph_structure == 0] = 0

        self.collected_rewards.append(np.sum(self.edge_rewards))
