from learners.learner import Learner
from environments.social_environment import SocialEnvironment
import numpy as np


class TSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(
            np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        )
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = (
            self.beta_parameters[pulled_arm, 0] + reward
        )
        self.beta_parameters[pulled_arm, 1] = (
            self.beta_parameters[pulled_arm, 1] + 1.0 - reward
        )
        self.graph = self.graph + reward


class TSProbLearner:
    def __init__(self, n_nodes, n_seeds, graph_structure=None) -> None:
        self.n_nodes = n_nodes
        self.n_seeds = n_seeds
        self.beta_parameters = np.ones((self.n_nodes, self.n_nodes, 2))
        self.graph_structure = graph_structure
        self.t = 0
        # Initialize the new attributes as empty lists
        self.activated_edges = []
        self.susceptible_edges = []

    def pull_arm(self):
        probs = self.beta_parameters[:, :, 0] / (
            self.beta_parameters[:, :, 0] + self.beta_parameters[:, :, 1]
        )
        if self.graph_structure is not None:
            probs *= self.graph_structure

        world_representation = SocialEnvironment(probs)
        seeds = world_representation.opt_arm(self.n_seeds)

        return seeds

    def update(self, episode):
        self.t += 1

        # Extract susceptible and activated edges from the episode tuple
        susceptible_edges, activated_edges = episode

        # Call update_observations to append the episode data to the respective lists
        self.update_observations(susceptible_edges, activated_edges)

        self.beta_parameters[:, :, 0] += activated_edges
        self.beta_parameters[:, :, 1] += susceptible_edges - activated_edges

    # Define the update_observations method
    def update_observations(self, susceptible_edges, activated_edges):
        self.susceptible_edges.append(susceptible_edges)
        self.activated_edges.append(activated_edges)
