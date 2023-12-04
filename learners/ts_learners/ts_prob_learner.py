import numpy as np
from learners.prob_learner import ProbLearner

class TSProbLearner(ProbLearner):
    def __init__(self, n_nodes, n_seeds, graph_structure=None):
        super().__init__(n_nodes, n_seeds, graph_structure)
        self.beta_parameters = np.ones((n_nodes, n_nodes, 2))
    
    def pull_arm(self):
        probs = self.beta_parameters[:, :, 0] / (
            self.beta_parameters[:, :, 0] + self.beta_parameters[:, :, 1]
        )
        if self.graph_structure is not None:
            probs *= self.graph_structure
        return super().pull_arm(probs)
    
    def update(self, episode):
        susceptible, activated = episode
        super().update_observations(susceptible, activated)
        self.beta_parameters[:, :, 0] += activated
        self.beta_parameters[:, :, 1] += susceptible - activated
        self.collected_rewards = np.append(self.collected_rewards, np.sum(activated))
        self.t += 1