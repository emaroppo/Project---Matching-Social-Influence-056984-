from learners.ts_learners.matching_ts import TSMatching
from utils.context_generation import ContextGenerator

import numpy as np


class AdaptiveTSMatching(TSMatching):
    def __init__(self, n_arms, n_product_classes, n_products_per_class):
        super().__init__(n_arms, 1, n_product_classes, n_products_per_class)
        self.context_generator = ContextGenerator()
        self.last_context_update = None
        self.dataset = []
        self.episode_counter = 0

    def update_context(self):
        if self.episode_counter + 1 % 14 == 0:
            self.context_generator.generate_new_context(
                self.dataset, self.n_product_classes
            )
            self.adapt_to_new_context()
            self.dataset = []
        self.episode_counter += 1

    def adapt_to_new_context(self):
        new_customer_classes = len(set(self.context_generator.current_context.values()))
        self.initialize_parameters(new_customer_classes)

    def initialize_parameters(self, new_customer_classes):
        self.mu = np.zeros((new_customer_classes, self.n_product_classes))
        self.lambda_ = np.ones((new_customer_classes, self.n_product_classes))
        self.alpha = np.ones((new_customer_classes, self.n_product_classes))
        self.beta = np.ones((new_customer_classes, self.n_product_classes))

        for features, product_class, reward in self.dataset:
            customer_class = self.context_generator.map_features_to_class(features)
            self.mu[customer_class, product_class] += reward
            self.lambda_[customer_class, product_class] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_reward = np.divide(self.mu, self.lambda_)
            avg_reward[np.isnan(avg_reward)] = 0

        self.mu = avg_reward

    def pull_arm(self, available_customers_features):
        customer_classes = np.array(
            [
                self.context_generator.map_features_to_class(features)
                for features in available_customers_features
            ]
        )
        best_arm_global = super().pull_arm(customer_classes, context=True)
        # replace index with features from available_customers_features
        best_arm_global = [
            (available_customers_features[idx], product_class)
            for idx, product_class in best_arm_global
        ]
        return best_arm_global

    def update(self, pulled_arms, reward):
        pulled_arms = [
            (self.context_generator.map_features_to_class(pulled_arm[0]), pulled_arm[1])
            for pulled_arm in pulled_arms
        ]
        episode_data = [
            (pulled_arm[0], pulled_arm[1], reward) for pulled_arm in pulled_arms
        ]

        self.dataset.append(episode_data)
        super().update(pulled_arms, reward)
