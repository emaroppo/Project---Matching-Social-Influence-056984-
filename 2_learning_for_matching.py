from environments.matching_environment import MatchingEnvironment
from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ts_learners.matching_ts import TSMatching2
from metrics import compute_metrics, plot_metrics
from data_generator import generate_reward_parameters

import numpy as np
from tqdm import tqdm

node_classes = 3
product_classes = 3
products_per_class = 3

reward_parameters = generate_reward_parameters(node_classes, product_classes)

# initialize environment
env = MatchingEnvironment(reward_parameters=reward_parameters)
# initialize bandit
ucb_bandit = UCBMatching(node_classes*product_classes, node_classes, product_classes, products_per_class)
ts_bandit = TSMatching2(node_classes*product_classes, node_classes, product_classes, products_per_class)

n_experiments = 365

opts = []

for i in tqdm(range(n_experiments)):
    # generate customers
    customers = np.random.randint(0, 3, np.random.randint(3, 12))
    # pull arm
    ucb_pulled_arm = ucb_bandit.pull_arms(customers)
    ts_bandit_pulled_arm = ts_bandit.pull_arms(customers)
    # retrieve reward
    ucb_reward = env.round(ucb_pulled_arm)
    ts_bandit_reward = env.round(ts_bandit_pulled_arm)
    opt = env.opt(customers, [0, 1, 2] * products_per_class)
    # update bandit
    ucb_bandit.update(ucb_pulled_arm, ucb_reward)
    ts_bandit.update(ts_bandit_pulled_arm, ts_bandit_reward)
    opts.append(opt)

print(ucb_bandit.collected_rewards.shape)
print(ts_bandit.collected_rewards.shape)

metrics = compute_metrics(ucb_bandit.collected_rewards, np.array(opts))
plot_metrics(*metrics, model_name="UCB", env_name="Matching UCB")
metrics = compute_metrics(
    ts_bandit.collected_rewards / np.array(opts), np.array([1] * n_experiments)
)
plot_metrics(*metrics, model_name="TS", env_name="Matching TS")

print(env.round(ucb_bandit.pull_arms(customers)).sum())
print(env.round(ts_bandit.pull_arms(customers)).sum())
print(env.opt(customers, [0, 1, 2] * 3))
print(customers)
