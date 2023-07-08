from environments.environment import Environment
from environments.social_environment import SocialEnvironment
from environments.matching_environment import MatchingEnvironment
import numpy as np


class NonStationaryEnvironment(Environment):
    def __init__(self, probabilities, horizon):
        super().__init__(probabilities)
        self.t = 0
        n_phases = len(self.probabilities)
        self.phase_size = horizon / n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward


class UnknownAbruptChanges(Environment):
    def __init__(self, probabilities, horizon, change_prob=0.3, n_phases=5):
        super().__init__(probabilities)
        self.t = 0
        # assign random length to each phase; they should still add up to horizon
        self.n_phases = n_phases
        self.change_prob = change_prob
        self.current_phase = 0

    def round(self, pulled_arm):
        if np.random.rand() < self.change_prob:
            self.current_phase += 1
            if self.current_phase >= self.n_phases:
                self.current_phase = 0
        p = self.probabilities[self.current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward


class SocialUnknownAbruptChanges(UnknownAbruptChanges, SocialEnvironment):
    def __init__(self, probabilities, horizon, n_phases=5, change_prob=0.3):
        UnknownAbruptChanges.__init__(self, probabilities, horizon, n_phases)
        SocialEnvironment.__init__(self, probabilities)

    def round(self, pulled_arm, joint=False):
        seeds = pulled_arm
        # interpreting 'phases cyclically change with a high frequency' as
        # the sequence of the phases is known, the phase length is short, but unknown
        if np.random.rand() < self.change_prob:
            self.current_phase += 1
            if self.current_phase >= self.n_phases:
                self.current_phase = 0

        history, active_nodes = self.simulate_episode(
            seeds, max_steps=100, prob_matrix=self.probabilities[self.current_phase]
        )
        if joint:
            return active_nodes

        reward = np.sum(active_nodes)
        return reward
