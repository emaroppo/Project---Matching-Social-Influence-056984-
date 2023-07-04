from learners.ucb_learners.matching_ucb import UCBMatching
import numpy as np

#da finire
class CUMSUMUCBMatching(UCBMatching):
    def __init__(self, n_arms, n_rows, n_cols, n_products_per_class=3, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols, n_products_per_class)
        self.change_detection = [[] for _ in range(n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(n_arms)]
        self.detection = [[] for _ in range(n_arms)]
        self.alpa = alpha 

    