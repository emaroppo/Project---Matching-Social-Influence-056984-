import numpy as np
from learners.ts_learners.matching_ts import TSMatching
from learners.ts_learners.ts_learner import TSProbLearner
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ucb_learners.matching_ucb import UCBMatching
from environments.joint_environment import JointEnvironment
from tqdm import tqdm
from utils.metrics import compute_metrics, plot_metrics
from utils.data_generator import generate_graph, generate_reward_parameters
from utils.simulation import joint_simulation

# init reward matrix, graph probabilities
n_nodes = 30
edge_rate = 0.2

n_seeds = 3
n_customer_classes = 3
n_product_classes = 3
products_per_class = 3
n_exp = 365

reward_means, reward_std_dev = generate_reward_parameters(
    n_customer_classes, n_product_classes
)
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
# link node ids to customer classes
node_classes = np.random.randint(0, n_customer_classes, n_nodes)

# initialise bandit
ts_bandit = TSProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)
ucb_bandit = UCBProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)

ts_matching = TSMatching(n_customer_classes, n_product_classes, products_per_class)
ucb_matching = UCBMatching(
    n_customer_classes * n_product_classes,
    n_customer_classes,
    n_product_classes,
    products_per_class,
)
# initialise environment
joint_env = JointEnvironment(
    graph_probabilities, (reward_means, reward_std_dev), node_classes
)

social, matching = joint_simulation(
    joint_env,
    [ts_bandit, ucb_bandit],
    [ts_matching, ucb_matching],
    n_episodes=n_exp,
    n_phases=1,
)
all_rewards, all_optimal_rewards = matching
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
    env_name="Joint Environment",
)


# check if it should maximise rewards (social & matching) separetely or jointly
