from utils.data_generator import (
    generate_graph,
    generate_reward_parameters,
    generate_customer_classes,
)
from steps.step_1_learning_for_social_influence import step_1
from steps.step_2_learning_for_matching import step_2
from steps.step_3_joint_learning import step_3
from steps.step_5_two_abrupt_changes import step_5
from steps.step_4_context import step_4
from steps.step_6_many_abrupt_changes import step_6

import numpy as np

n_episodes = 30
# set arguments
n_nodes = 30
n_seeds = 3
edge_rate = 0.2

# feature mapping
def feature_mapping(features):
    #temporary fix
    features = features.reshape(1,2)
    if features[0][0] == 1:  # (1, 0), (1, 1) -> 0
        return 0
    elif features[0][1] == 1:  # (0, 1) -> 1
        return 1
    else:
        return 2  # (0, 0) -> 2


# set arguments
n_product_classes = 3
n_customer_classes = 3
n_products_per_class = 3
reward_parameters = generate_reward_parameters(n_customer_classes=n_customer_classes, n_product_classes=n_product_classes)
customer_features, customer_labels = generate_customer_classes(feature_mapping, 30)

# generate graph
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)

step_1(graph_probabilities, graph_structure, n_episodes=n_episodes)

# TO DO: replace with nodes generated from best arm
active_nodes = [
    np.random.choice(30, np.random.randint(6, 12), replace=False)
    for _ in range(n_episodes)
]


step_2(
    reward_parameters,
    n_customer_classes,
    n_product_classes,
    n_products_per_class,
    n_episodes=n_episodes,
    active_nodes=active_nodes,
    class_mapping=customer_labels,
)

step_3(
    n_nodes=n_nodes,
    graph_probabilities=graph_probabilities,
    graph_structure=graph_structure,
    n_seeds=n_seeds,
    class_mapping=customer_labels,
    n_customer_classes=n_customer_classes,
    n_product_classes=n_product_classes,
    products_per_class=n_products_per_class,
    reward_parameters=reward_parameters,
    n_exp=n_episodes,
)

step_4(
    n_nodes=n_nodes,
    graph_probabilities=graph_probabilities,
    graph_structure=graph_structure,
    n_seeds=n_seeds,
    customer_features=customer_features,
    feature_mapping=feature_mapping,
    n_customer_classes=n_customer_classes,
    n_product_classes=n_product_classes,
    products_per_class=n_products_per_class,
    reward_parameters=reward_parameters,
    n_exp=n_episodes,
)

graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate, n_phases=3)

step_5(
    graph_probabilities=graph_probabilities,
    graph_structure=graph_structure,
    n_phases=3,
    n_episodes=n_episodes,
)

step_6(
    graph_probabilities=graph_probabilities,
    graph_structure=graph_structure,
    n_phases=3,
    n_episodes=n_episodes,
)
