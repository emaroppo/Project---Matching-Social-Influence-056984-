from environments.matching_environment import MatchingEnvironment
from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ts_learners.matching_ts import TSMatching2
from metrics import plot_metrics
from data_generator import generate_reward_parameters, generate_customer_classes
from simulation import matching_simulation

# set arguments
n_node_classes = 3
n_product_classes = 3
n_products_per_class = 3
n_experiments = 365

# generate rewards, customer classes and environment
reward_parameters = generate_reward_parameters(n_node_classes, n_product_classes)
class_mapping = generate_customer_classes(n_node_classes, 30)

env = MatchingEnvironment(reward_parameters=reward_parameters)

# initialize bandit
ucb_bandit = UCBMatching(
    n_node_classes * n_product_classes,
    n_node_classes,
    n_product_classes,
    n_products_per_class,
)
ts_bandit = TSMatching2(
    n_node_classes * n_product_classes,
    n_node_classes,
    n_product_classes,
    n_products_per_class,
)

# run simulation

matching_metrics = matching_simulation(
    env, ucb_bandit, n_experiments, class_mapping=class_mapping
)

plot_metrics(*matching_metrics, model_name="TS", env_name="Matching TS")
