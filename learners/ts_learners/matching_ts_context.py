import numpy as np
from learners.ts_learners.matching_ts import TSMatching

class TSMatchingContext(TSMatching):
    def __init__(
        self, n_arms, n_customer_classes, n_product_classes, n_products_per_class=3
    ):
        super().__init__(
            n_arms, n_customer_classes, n_product_classes, n_products_per_class
        )
        self.context_generator=ContextGenerationAlgorithm(n_customer_classes, [0, 1, 2])
        self.t=0

    
    def pull_arm(self, active_customers, customer_features):
        customer_classes = self.context_generator.predict(customer_features)
        super().pull_arm(active_customers, customer_classes)

    def update(self, pulled_arms, rewards):
        self.collected_rewards = np.append(
            self.collected_rewards, np.array(rewards).sum()
        )
        for pulled_arm, reward in zip(pulled_arms, rewards):
            active_customer, pulled_arm_customer, pulled_arm_product = pulled_arm
            _, actual_reward = reward
            self._update_arm_parameters(
                (None, pulled_arm_customer, pulled_arm_product), actual_reward
            )

        #TODO: create episode variable
        self.t += 1
        self.context_generator.add_episode(episode)
        
        if self.t % 14 == 0:
            new_context_structure = self.context_generator.generate_context_structure()
            
            if new_context_structure:
                self.resize_arms(new_context_structure)

    def resize_arms(self, new_context_structure):
        current_splits = set(
            tuple(tuple(s) for s in sorted(leaf.split)) for leaf in self.current_classes
        )
        new_splits = set(
            tuple(tuple(s) for s in sorted(leaf.split))
            for leaf in new_context_structure.leaves
        )

        obsolete_splits = current_splits - new_splits
        new_added_splits = new_splits - current_splits

        if len(self.current_classes) == 0:
            self.mu = np.delete(self.mu, 0, axis=0)
            self.lambda_ = np.delete(self.lambda_, 0, axis=0)
            self.alpha = np.delete(self.alpha, 0, axis=0)
            self.beta = np.delete(self.beta, 0, axis=0)
            self.n_samples = np.delete(self.n_samples, 0, axis=0)

        for split in obsolete_splits:
            idx = next(
                i
                for i, leaf in enumerate(self.current_classes)
                if tuple(sorted(leaf.split)) == split
            )

            self.mu = np.delete(self.mu, idx, axis=0)
            self.lambda_ = np.delete(self.lambda_, idx, axis=0)
            self.alpha = np.delete(self.alpha, idx, axis=0)
            self.beta = np.delete(self.beta, idx, axis=0)
            self.n_samples = np.delete(self.n_samples, idx, axis=0)

        for _ in new_added_splits:
            self.mu = np.vstack([self.mu, np.zeros((1, self.n_product_classes + 1))])
            self.lambda_ = np.vstack(
                [self.lambda_, np.ones((1, self.n_product_classes + 1))]
            )
            self.alpha = np.vstack(
                [self.alpha, np.ones((1, self.n_product_classes + 1)) * 0.5]
            )
            self.beta = np.vstack(
                [self.beta, np.ones((1, self.n_product_classes + 1)) * 0.5]
            )
            self.n_samples = np.vstack(
                [self.n_samples, np.zeros((1, self.n_product_classes + 1), dtype=int)]
            )

        self.current_classes = new_context_structure.leaves
        self.n_customer_classes = len(self.current_classes)
