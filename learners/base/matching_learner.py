from learners.base.learner import Learner
from scipy.optimize import linear_sum_assignment
import numpy as np


class MatchingLearner(Learner):
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class
    ):
        super().__init__(n_arms)
        self.n_customer_classes = n_customer_classes
        self.n_product_classes = n_product_classes
        self.n_products_per_class = n_products_per_class

    def pull_arm(self, parameters, customer_classes, context=False):
        customer_classes = customer_classes.reshape(-1)
        available_matches = np.repeat(
            parameters[customer_classes, :],
            self.n_products_per_class,
            axis=1,
        )
        row_ind, col_ind = linear_sum_assignment(-available_matches)

        if context:
            best_arms_global = [
                (
                    row if row < len(customer_classes) else self.n_customer_classes,
                    col // 3,
                )
                for row, col in zip(row_ind, col_ind)
            ]

        else:
            best_arms_global = [
                (
                    customer_classes[row]
                    if row < len(customer_classes)
                    else self.n_customer_classes,
                    col // 3,
                )
                for row, col in zip(row_ind, col_ind)
            ]
        return best_arms_global

    def update_observations(self, pulled_arms, rewards):
        pulled_arms_idx = [
            customer_class * self.n_product_classes + product_class
            for customer_class, product_class in pulled_arms
            if customer_class != self.n_customer_classes
            and product_class != self.n_product_classes
        ]
        print(pulled_arms_idx)
        for pulled_arm, reward in zip(pulled_arms_idx, rewards):
            super().update_observations(pulled_arm, reward)
