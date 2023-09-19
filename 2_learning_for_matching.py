from environments.matching_environment import MatchingEnvironment2
from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ts_learners.matching_ts import TSMatching3
from utils.metrics import compute_metrics, plot_metrics
from utils.data_generator import generate_reward_parameters, generate_customer_classes
from utils.simulation import matching_simulation
import numpy as np

# set arguments
n_node_classes = 3
n_product_classes = 3
n_products_per_class = 3
n_episodes = 365

# generate rewards, customer classes and environment
reward_parameters = generate_reward_parameters(n_node_classes, n_product_classes)
class_mapping = generate_customer_classes(n_node_classes, 30)
active_nodes = [
    np.random.choice(30, np.random.randint(6, 12), replace=False)
    for _ in range(n_episodes)
]
print(active_nodes)

env = MatchingEnvironment2(reward_parameters=reward_parameters)

# initialize bandit
ucb_bandit = UCBMatching(
    n_node_classes * n_product_classes,
    n_node_classes,
    n_product_classes,
    n_products_per_class,
)
ts_bandit = TSMatching3(
    n_node_classes * n_product_classes,
    n_node_classes,
    n_product_classes,
    n_products_per_class,
)

# run simulation

all_rewards, all_optimal_rewards = matching_simulation(
    env=env,
    models=[ucb_bandit, ts_bandit],
    n_episodes=n_episodes,
    active_nodes=active_nodes,
    class_mapping=class_mapping,
)

all_instantaneous_regrets = [
    compute_metrics(r, o)[2] for r, o in zip(all_rewards, all_optimal_rewards)
]
all_cumulative_regrets = [
    compute_metrics(r, o)[3] for r, o in zip(all_rewards, all_optimal_rewards)
]

plot_metrics(
    all_rewards,
    all_optimal_rewards,
    all_instantaneous_regrets,
    all_cumulative_regrets,
    model_names=["Model1", "Model2"],
    env_name="Social Environment",
)
