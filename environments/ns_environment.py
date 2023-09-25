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
            print("phase change!")
            self.current_phase += 1
            if self.current_phase >= self.n_phases:
                self.current_phase = 0
        p = self.probabilities[self.current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward


class SocialNChanges(SocialEnvironment):
    def __init__(self, phase_probabilities, horizon=365, n_phases=5):
        curr_probabilities = phase_probabilities[0]
        super().__init__(curr_probabilities)
        self.phase_probabilities = phase_probabilities
        self.phase_changes = np.random.randint(1, horizon, n_phases - 1)
        self.phase_changes.sort()
        self.curr_phase = 0
        self.t = 0

    def round(self, pulled_arms, joint=False):
        # check if phase changes at t
        change = False
        if self.curr_phase < len(self.phase_changes):
            if self.t == self.phase_changes[self.curr_phase]:
                print("phase change!")
                self.curr_phase += 1
                self.probabilities = self.phase_probabilities[self.curr_phase]
                change = True

        episode, reward = super().round(pulled_arms, joint=joint)

        self.t += 1

        return episode, reward, change


class SocialUnknownAbruptChanges(SocialEnvironment):
    def __init__(
        self, phase_probabilities, n_episodes=365, n_phases=5, change_prob=0.3
    ):
        curr_probabilities = phase_probabilities[0]
        super().__init__(curr_probabilities)
        self.phase_probabilities = phase_probabilities

        # Generate random phase change points, but ensure they happen with at least the change_prob chance.
        self.phase_changes = []
        t = 0
        for _ in range(n_phases - 1):
            delta = np.random.geometric(change_prob)
            t += delta
            if t >= n_episodes:
                break
            self.phase_changes.append(t)

        self.curr_phase = 0
        self.t = 0

    def round(self, pulled_arms, joint=False):
        # Check if phase changes at t
        change = False
        if self.curr_phase < len(self.phase_changes):
            if self.t == self.phase_changes[self.curr_phase]:
                print("phase change!")
                self.curr_phase += 1
                self.probabilities = self.phase_probabilities[self.curr_phase]
                change = True

        episode, reward = super().round(pulled_arms, joint=joint)

        self.t += 1

        return episode, reward, change
