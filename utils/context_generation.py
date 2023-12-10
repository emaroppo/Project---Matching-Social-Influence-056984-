import numpy as np


class ContextGenerator:
    def __init__(self):
        # Initial context with a single cluster containing all feature combinations
        self.current_context = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        self.clusters = [set(self.current_context.keys())]

    def generate_new_context(self, data, n_product_classes):
        # Calculate average reward for each combination of features and product class
        feature_product_rewards = {((0, 0), i): [] for i in range(n_product_classes)}
        feature_product_rewards.update(
            {((0, 1), i): [] for i in range(n_product_classes)}
        )
        feature_product_rewards.update(
            {((1, 0), i): [] for i in range(n_product_classes)}
        )
        feature_product_rewards.update(
            {((1, 1), i): [] for i in range(n_product_classes)}
        )

        for episode in data:
            for features, product_class, reward in episode:
                feature_product_rewards[(tuple(features), product_class)].append(reward)

        avg_rewards = {
            fp: np.mean(rewards) if rewards else 0
            for fp, rewards in feature_product_rewards.items()
        }

        self.split_clusters(avg_rewards)

        for idx, cluster in enumerate(self.clusters):
            for feature in cluster:
                self.current_context[feature] = idx

    def split_clusters(self, avg_rewards):
        variances = [
            np.var(
                [
                    avg_rewards[(feature, i)]
                    for feature in cluster
                    for i in range(len(avg_rewards) // 4)
                ]
            )
            for cluster in self.clusters
        ]
        cluster_to_split = np.argmax(variances)

        cluster = list(self.clusters[cluster_to_split])
        cluster.sort(
            key=lambda feature: np.mean(
                [avg_rewards[(feature, i)] for i in range(len(avg_rewards) // 4)]
            )
        )
        median_index = len(cluster) // 2
        self.clusters[cluster_to_split] = set(cluster[:median_index])
        self.clusters.append(set(cluster[median_index:]))

    def map_features_to_class(self, features):
        return self.current_context[(features[0][0], features[0][1])]
