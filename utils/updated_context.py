import numpy as np
from scipy.optimize import (
    linear_sum_assignment,
)  # Assuming you are using scipy's method
import copy


class TreeNode:
    def __init__(self, children=None, parent=None) -> None:
        if children is None:
            children = []
        self.children = children
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


class Context(TreeNode):
    def __init__(self, split, parent=None, dataset=None, n_product_classes=3):
        super().__init__(parent=parent)
        self.split = (
            split if split is not None else []
        )  # ordered list of features, values (0 or 1)
        self.dataset = dataset
        self.n_product_classes = n_product_classes
        self.dataset_features = (
            dataset[:, :-n_product_classes] if dataset is not None else None
        )
        self.dataset_rewards = (
            dataset[:, -n_product_classes:] if dataset is not None else None
        )

        self.expected_reward = 0
        self.expected_prob = 0
        if (
            self.parent
            and self.parent.dataset is not None
            and len(self.parent.dataset) > 0
        ):
            self.expected_prob = len(self.dataset) / len(self.parent.dataset)
            self.expected_reward = np.mean(self.dataset_rewards)

    def update_features_and_rewards(self):
        self.dataset_features = (
            self.dataset[:, : -self.n_product_classes]
            if self.dataset is not None
            else None
        )
        self.dataset_rewards = (
            self.dataset[:, -self.n_product_classes :]
            if self.dataset is not None
            else None
        )

    def propagate_dataset_down(self):
        if (
            not self.children
        ):  # Base case: if it's a leaf node, no further action needed
            return

        # If it's not a leaf node, split its dataset and propagate down to its children
        for child in self.children:
            feature, value = child.split[
                -1
            ]  # the last split added corresponds to this child
            child.dataset = self.dataset[self.dataset[:, feature] == value]
            print(self.dataset[self.dataset[:, feature] == value])
            child.dataset_features = (
                child.dataset[:, : -self.product_classes]
                if child.dataset is not None
                else None
            )
            child.dataset_rewards = (
                child.dataset[:, -self.product_classes :]
                if child.dataset is not None
                else None
            )
            child.propagate_dataset_down()  # Recursively do the same for this child

    def split_context(self, feature, value):
        # split dataset into children
        child_split = self.split.copy()
        child_split.append([feature, value])
        # retrieve rows of dataset that fall into this context
        dataset = self.dataset[self.dataset[:, feature] == value]
        # retrieve rows of dataset that fall into the complement of this context
        dataset_complement = self.dataset[self.dataset[:, feature] != value]
        # create children
        child = Context(child_split, self, dataset)
        # complement split
        complement_split = self.split.copy()
        complement_split.append([feature, 1 - value])

        child_complement = Context(complement_split, self, dataset_complement)

        return child, child_complement


class ContextStructure:
    def __init__(self, episodes, n_features, product_classes) -> None:
        if episodes is None:
            self.episodes = list()
        else:
            self.episodes = episodes
        self.root = Context(None, dataset=episodes)
        self.n_features = n_features
        self.product_classes = product_classes
        self.leaves = [self.root]

        self.customer_classes = list()
        self.features_mapping = dict()
        for i, j in enumerate(self.leaves):
            self.customer_classes.append(i)
            # from j.split retrieve the features and values
            # create mapping from features and values to customer classes
            for feature, value in j.split:
                if feature not in self.features_mapping:
                    self.features_mapping[feature] = dict()
                if value not in self.features_mapping[feature]:
                    self.features_mapping[feature][value] = list()
                self.features_mapping[feature][value].append(i)

    @staticmethod
    def create_mapping(data, rules):
        mapping = {}

        for i, row in enumerate(data):
            for class_label, rule in enumerate(rules):
                if all(row[j] == val for j, val in rule.items()):
                    mapping[i] = class_label
                    break

        return mapping

    def copy(self):
        return copy.deepcopy(self)

    def create_reward_matrix(self, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = self.leaves
        n_customer_classes = len(leaf_nodes)
        n_product_classes = len(self.product_classes)
        reward_matrix = np.zeros((n_customer_classes, n_product_classes))
        for i, j in enumerate(leaf_nodes):
            print(j.dataset_rewards)
            j.update_features_and_rewards()
            reward_matrix[i] = np.mean(j.dataset_rewards, axis=0)

        return reward_matrix

    def split_context(self, context_index, feature, value):
        # copy current context structure
        context_structure = self.copy()
        # split argument context into children
        context = context_structure.leaves[context_index]
        child, child_complement = context.split_context(feature, value)
        context_structure.leaves.remove(child.parent)
        context_structure.leaves.append(child)
        context_structure.leaves.append(child_complement)

        return context_structure

    def create_rules(self):
        rules = []
        for i in self.leaves:
            rule = dict()
            for j in i.split:
                rule[j[0]] = j[1]
            rules.append(rule)

        return rules

    def test_context_structure(self):
        reward_matrix = self.create_reward_matrix()
        rules = self.create_rules()

        product_classes = [0, 1, 2] * 3

        episode_cum_rew = 0
        for item in self.episodes:
            for episode in item:
                episode_customers, episode_rewards = (
                    episode[:, : len(self.product_classes)],
                    episode[:, -len(self.product_classes) :],
                )
                customer_classes_mapping = self.create_mapping(episode, rules)
                customer_classes = [
                    customer_classes_mapping[i] for i in range(len(episode_customers))
                ]

                # from list of customer classes, product classes and reward matrix, create reward matrix for episode, compute optimal assignment
                episode_table = np.zeros((len(customer_classes), len(product_classes)))

                for i, j in enumerate(customer_classes):
                    for k, l in enumerate(product_classes):
                        episode_table[i, k] = reward_matrix[j, l]

                episode_assignment = linear_sum_assignment(-episode_table)
                episode_cum_rew += episode_table[episode_assignment].sum()

        return episode_cum_rew


class ContextGenerationAlgorithm:
    def __init__(self, n_features, product_classes) -> None:
        self.n_features = n_features
        self.product_classes = product_classes
        self.context_structures = [ContextStructure(None, n_features, product_classes)]
        self.dataset = []

    def update_dataset(self, episodes):
        self.dataset.extend(episodes)
        self.context_structures[-1].root.dataset = np.concatenate(self.dataset, axis=0)
        print(self.context_structures[-1].episodes.append(episodes))
        print(self.context_structures[-1].root.dataset.shape)
        self.context_structures[
            -1
        ].root.propagate_dataset_down()  # Propagate the dataset down the tree

    def update_context_structures(self, context_structure):
        self.context_structures.append(context_structure)

    def update(self, episodes):
        self.update_dataset(episodes)

        improvement_found = True  # To keep track if we found an improvement
        while improvement_found:
            improvement_found = False  # Initially assume no improvement
            # Start with the latest context structure
            current_structure = self.context_structures[-1]
            current_reward = (
                current_structure.test_context_structure()
            )  # Initial reward
            for leaf_index in range(len(current_structure.leaves)):
                for feature in range(self.n_features):
                    for value in [0, 1]:  # Assuming binary split
                        # Attempt a split and create a new context structure
                        proposed_structure = current_structure.split_context(
                            leaf_index, feature, value
                        )
                        proposed_reward = proposed_structure.test_context_structure()
                        print(proposed_reward)

                        # Check if the proposed structure is better
                        if proposed_reward > current_reward:
                            # Replace with the new structure and set flag to start over
                            self.update_context_structures(proposed_structure)
                            improvement_found = True
                            break  # Break out of the innermost loop
                if improvement_found:
                    break  # If found an improvement, break out of this loop to start over

        return self.context_structures[-1]  # Return the best structure


def evaluate_split(self, episode):
    """
    Evaluates a potential split and decides whether to accept or reject it.
    """
    # ... [Implementation provided above]
    return optimal_reward, decision


    
    def evaluate_split(self, episode):
        """Evaluates a potential split and decides whether to accept or reject it."""
        # Compute the expected reward matrix for the episode
        reward_matrix = self.create_expected_reward_matrix(episode, self.perform_hierarchical_clustering())
        
        # The Hungarian algorithm minimizes cost, so we'll use negative rewards as the cost matrix
        cost_matrix = -reward_matrix
        
        # Using the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_reward = -cost_matrix[row_ind, col_ind].sum()
        
        # Placeholder for decision criteria:
        # For the sake of this demonstration, I'll set a threshold for accepting a split.
        # This can be replaced with more sophisticated logic based on your requirements.
        threshold = 0.5  # Placeholder value
        decision = optimal_reward > threshold
        
        return optimal_reward, decision
    