from learners.learner import Learner
from environments.social_environment import SocialEnvironment
import numpy as np


class ProbLearner(Learner):
    def __init__(self, n_nodes, n_seeds, graph_structure=None):
        super().__init__(n_nodes)
        self.n_nodes = n_nodes
        self.n_seeds = n_seeds
        self.graph_structure = graph_structure

        self.edge_rewards = np.zeros((n_nodes, n_nodes))
        self.susceptible_edges = []
        self.activated_edges = []

    def pull_arm(self, probs):
        world_representation = SocialEnvironment(probs)
        seeds = world_representation.opt_arm(self.n_seeds)

        return seeds

    def update_observations(self, susceptible_edges, activated_edges):
        self.susceptible_edges.append(susceptible_edges)
        self.activated_edges.append(activated_edges)
        self.collected_rewards = np.append(
            self.collected_rewards, np.sum(activated_edges)
        )
        self.t += 1
