import numpy as np


class CUSUM:
    def __init__(self, n_nodes, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = np.zeros((n_nodes, n_nodes))
        self.reference = np.zeros((n_nodes, n_nodes))
        self.g_plus = np.zeros((n_nodes, n_nodes))
        self.g_minus = np.zeros((n_nodes, n_nodes))

    def update(self, samples):
        self.t += 1
        still_in_initial_phase = self.t <= self.M
        self.reference += np.where(still_in_initial_phase, samples / self.M, 0)

        s_plus = (samples - self.reference) - self.eps
        s_minus = -(samples - self.reference) - self.eps
        self.g_plus = np.maximum(0, self.g_plus + s_plus)
        self.g_minus = np.maximum(0, self.g_minus + s_minus)

        changes_detected = (self.g_plus > self.h) | (self.g_minus > self.h)
        return changes_detected

    def reset(self, indices):
        self.t[indices] = 0
        self.g_plus[indices] = 0
        self.g_minus[indices] = 0
        self.reference[indices] = 0
