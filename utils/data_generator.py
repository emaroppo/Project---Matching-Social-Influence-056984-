import numpy as np

def generate_graph(n_nodes, edge_rate, min_prob=0.1, max_prob=0.2, n_phases=1):
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

    
    graph_structure_expanded = np.stack([graph_structure]*n_phases, axis=0)

    graph_probabilities = np.random.uniform(min_prob, max_prob, (n_phases, n_nodes, n_nodes)) * graph_structure_expanded
    if n_phases==1:
        graph_probabilities = np.squeeze(graph_probabilities)
    return graph_probabilities, graph_structure

def generate_reward_parameters(n_customer_classes, n_product_classes, min_reward=10, max_reward=20):
    # reward parameters
    means = np.random.uniform(min_reward, max_reward, (n_customer_classes, n_product_classes))
    # arr of 1s of size (3,3)
    std_dev = np.random.uniform(0.5, 2, (n_customer_classes, n_product_classes))
    return means, std_dev

def generate_customer_classes(n_customer_classes, n_nodes):
    # generate customer classes
    class_mapping = np.random.randint(0, n_customer_classes, n_nodes)
    return class_mapping