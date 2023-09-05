import numpy as np

def generate_graph(n_nodes, edge_rate, min_prob=0.1, max_prob=0.2):
    graph_structure = np.random.binomial(1, edge_rate, (n_nodes, n_nodes))
    #make sure no node is connected to itself
    np.fill_diagonal(graph_structure, 0)
    #if a node has no edges, add a random edge, but not to itself
    for i in range(n_nodes):
        if np.sum(graph_structure[i]) == 0:
            j = np.random.randint(0, n_nodes)
            while j == i:
                j = np.random.randint(0, n_nodes)
            graph_structure[i, j] = 1

    graph_probabilities = np.random.uniform(min_prob, max_prob, (n_nodes, n_nodes)) * graph_structure
    return graph_probabilities, graph_structure

def generate_reward_parameters(n_customer_classes, n_product_classes, min_reward=10, max_reward=20):
    # reward parameters
    means = np.random.uniform(10, 20, (n_customer_classes, n_product_classes))
    # arr of 1s of size (3,3)
    std_dev = np.ones((3, 3))
    return means, std_dev