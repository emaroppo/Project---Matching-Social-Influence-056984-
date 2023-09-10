import numpy as np
from environments.matching_environment import MatchingEnvironment


class TreeNode:
    def __init__(self, children=list(), parent=None) -> None:
        self.children = children
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


class Context(TreeNode):
    def __init__(self, split, parent=None, dataset=None, n_product_classes=3):
        super().__init__(parent)
        self.split = split  # ordered list of features, values (0 or 1)
        self.dataset_features = dataset[:, :-n_product_classes]
        self.dataset_rewards = dataset[:, -n_product_classes:]

        self.expected_reward = self.expected_prob = len(self.dataset) / len(
            self.parent.dataset
        )
        self.expected_prob = np.mean(self.dataset_rewards)

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
        child_complement = Context(child_split, self, dataset_complement)

        return child, child_complement


class ContextStructure:
    def __init__(self, episodes, n_features, product_classes) -> None:
        self.episodes = episodes
        dataset = np.concatenate(episodes, axis=0)
        self.root = Context(None, dataset=dataset)
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

    def create_mapping(data, rules):
        mapping = {}

        for i, row in enumerate(data):
            for class_label, rule in enumerate(rules):
                if all(row[j] == val for j, val in rule.items()):
                    mapping[i] = class_label
                    break

        return mapping

    def create_reward_matrix(self, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = self.leaves
        n_customer_classes = len(leaf_nodes)
        n_product_classes = len(self.product_classes)
        reward_matrix = np.zeros((n_customer_classes, n_product_classes))
        for i, j in enumerate(leaf_nodes):
            reward_matrix[i] = np.mean(j.dataset_rewards, axis=0)

        return reward_matrix

    def split_context(self, context, feature, value):
        child, child_complement = context.split_context(feature, value)

        return child, child_complement

    def confirm_split(self, feature, value):
        child, child_complement = self.split_context(feature, value)
        self.children.append(child)
        self.children.append(child_complement)
        self.leaves.remove(child.parent)
        self.leaves.append(child)
        self.leaves.append(child_complement)

    def test_split(self, context, feature, value, cumulative_reward=None):
        child, child_complement = self.split_context(context, feature, value)
        if cumulative_reward is None:
            # calculate cumulative reward
            pass
        # calculate expected reward of proposed split
        # create reward matrix of proposed split
        reward_matrix = self.create_reward_matrix([child, child_complement])
        rules = []
        for i in self.leaves + [child, child_complement]:
            rule = dict()
            for j in i.split:
                rule[j[0]] = j[1]
            rules.append(rule)
        world_rep = MatchingEnvironment(reward_matrix)

        # calculate expected reward of proposed split
        class_occurences = np.zeros(len(self.leaves) + 2)
        awards = np.zeros(len(self.leaves) + 2)

        for episode in self.episodes:
            episode_customers, episode_rewards = (
                episode[:, : self.product_classes],
                episode[:, -self.product_classes :],
            )
            customer_classes_mapping = self.create_mapping(episode, rules)
            customer_classes = [
                customer_classes_mapping[i] for i in range(len(episode_customers))
            ]
            world_rep.opt(
                customer_classes=customer_classes, product_classes=[0, 1, 2] * 3
            )
            class_occurences += np.array(
                [customer_classes.count(i) for i in range(len(self.leaves) + 2)]
            )

        return world_rep.opt(), cumulative_reward


class ContextGenerationAlgorithm:
    def __init__(self, n_features, product_classes) -> None:
        self.n_features = n_features
        self.product_classes = product_classes
        self.context_structures = [ContextStructure(None, n_features, product_classes)]
        self.dataset = []

    def update_dataset(self, episodes):
        self.dataset.append(episodes)

    def update_context_structures(self, context_structure):
        self.context_structures.append(context_structure)

    def update(self, episodes):
        self.update_dataset(episodes)
        # retrieve cumulative reward of current context structure
        cumulative_reward = None
        new_expected_reward = None
        while new_expected_reward > cumulative_reward:
            for i in leaf_nodes:
                exp_reward = context_structure.test_split(
                    i, feature, value, cumulative_reward
                )
                if exp_reward > cumulative_reward:
                    i.confirm_split(feature, value)
                    break
        
    
    def generate_context_structure(self):

