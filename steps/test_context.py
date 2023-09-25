import numpy as np
from utils.updated_context import ContextGenerationAlgorithm
from scipy.optimize import linear_sum_assignment


def test_initialization():
    context_gen = ContextGenerationAlgorithm(2, 3)
    assert context_gen.n_features == 2
    assert context_gen.n_product_classes == 3
    print("Initialization Test Passed!")


def test_hierarchical_clustering():
    context_gen = ContextGenerationAlgorithm(2, 3)
    context_gen.dataset = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    cluster_labels = context_gen.perform_hierarchical_clustering()
    assert len(cluster_labels) == 2
    print("Hierarchical Clustering Test Passed!")


def test_assign_customer_to_class():
    context_gen = ContextGenerationAlgorithm(2, 3)
    context_gen.dataset = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    customer_features = np.array([[0, 1], [1, 0]])
    assigned_classes = context_gen.assign_customer_to_class(customer_features)
    assert len(assigned_classes) == 2
    print("Assign Customer to Class Test Passed!")


def test_compute_expected_rewards():
    context_gen = ContextGenerationAlgorithm(2, 3)
    context_gen.dataset = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    cluster_labels = context_gen.perform_hierarchical_clustering()
    expected_rewards = context_gen.compute_expected_rewards(cluster_labels)
    assert len(expected_rewards) == 2
    print("Compute Expected Rewards Test Passed!")


def test_map_episode_data_to_classes():
    context_gen = ContextGenerationAlgorithm(2, 3)
    context_gen.dataset = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    episode = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0]])
    cluster_labels = context_gen.perform_hierarchical_clustering()
    mapped_episode = context_gen.map_episode_data_to_classes(episode, cluster_labels)
    assert len(mapped_episode) == 2
    print("Map Episode Data to Classes Test Passed!")


def test_create_expected_reward_matrix():
    context_gen = ContextGenerationAlgorithm(2, 3)
    context_gen.dataset = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    episode = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0]])
    cluster_labels = context_gen.perform_hierarchical_clustering()
    reward_matrix = context_gen.create_expected_reward_matrix(episode, cluster_labels)
    assert reward_matrix.shape == (2, 3)
    print("Expected Reward Matrix Test Passed!")


def test_evaluate_split():
    context_gen = ContextGenerationAlgorithm(2, 3)
    context_gen.dataset = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
    episode = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0]])
    optimal_reward, decision = context_gen.evaluate_split(episode)
    assert isinstance(optimal_reward, float)
    assert isinstance(decision, bool)
    print("Evaluate Split Test Passed!")


# Running the tests
if __name__ == "__main__":
    test_initialization()
    test_hierarchical_clustering()
    test_assign_customer_to_class()
    test_compute_expected_rewards()
    test_map_episode_data_to_classes()
    test_create_expected_reward_matrix()
    test_evaluate_split()
