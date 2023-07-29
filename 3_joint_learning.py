import numpy as np
import matplotlib.pyplot as plt
from learners.ts_learners.matching_ts import TSMatching
from learners.ts_learners.joint_ts import JointTSLearner
from learners.ucb_learners.matching_ucb import UCBMatching
from learners.ucb_learners.joint_ucb import JointMAUCBLearner
from environments.joint_environment import JointEnvironment
from tqdm import tqdm
from metrics import compute_metrics, plot_metrics
from data_generator import generate_graph, generate_reward_parameters

# init reward matrix, graph probabilities
n_nodes = 30
edge_rate = 0.2

n_seeds = 3
n_customer_classes = 3
n_product_classes = 3
products_per_class = 3
n_exp = 1000

reward_means, reward_std_dev = generate_reward_parameters(n_customer_classes, n_product_classes)
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
# link node ids to customer classes
node_classes = np.random.randint(0, n_customer_classes, n_nodes)

# initialise environments
joint_env = JointEnvironment(
    graph_probabilities, (reward_means, reward_std_dev), node_classes
)

# initialise bandits
maucb_bandit = JointMAUCBLearner(n_nodes, n_seeds)
matching_ucb = UCBMatching(
    reward_means.size, reward_means.shape[0], reward_means.shape[1], products_per_class
)

mats_bandit = JointTSLearner(n_nodes, n_seeds)
matching_ts = TSMatching(
    reward_means.size, reward_means.shape[0], reward_means.shape[1], products_per_class
)

for i in tqdm(range(n_exp)):
    ucb_pulled_arms = maucb_bandit.pull_arms()
    ts_pulled_arms = mats_bandit.pull_arms()

    # retrieve activated nodes
    ucb_social_reward = joint_env.social_environment.round(ucb_pulled_arms, joint=True)
    ts_social_reward = joint_env.social_environment.round(ts_pulled_arms, joint=True)

    # print(social_reward)
    # convert to list of integer indices corresponding to the position of activated nodes in the reward matrix
    ucb_social_reward = np.where(ucb_social_reward == 1)[0]
    ts_social_reward = np.where(ts_social_reward == 1)[0]
    # convert to list of classes
    ucb_social_reward = [node_classes[i] for i in ucb_social_reward]
    ts_social_reward = [node_classes[i] for i in ts_social_reward]

    # perform matching assuming customer classes are known but distributions unknown
    ucb_prop_match = matching_ucb.pull_arms(ucb_social_reward)
    ts_prop_match = matching_ts.pull_arms(ts_social_reward)
    # retrieve reward
    ucb_matching_reward = joint_env.matching_environment.round(ucb_prop_match)
    ts_matching_reward = joint_env.matching_environment.round(ts_prop_match)

    # update bandits
    matching_ucb.update(ucb_prop_match, ucb_matching_reward)
    maucb_bandit.update(ucb_pulled_arms, ucb_matching_reward.sum())

    matching_ts.update(ts_prop_match, ts_matching_reward)
    mats_bandit.update(ts_pulled_arms, ts_matching_reward.sum())


print(ucb_pulled_arms)
rewards = maucb_bandit.collected_rewards[::3]
metrics = compute_metrics(rewards, [joint_env.opt(n_seeds, [0, 1, 2] * 3)[0]] * n_exp)
plot_metrics(*metrics, model_name='UCB', env_name="Joint UCB")
print(ucb_matching_reward.sum())

print(ts_pulled_arms)
rewards = mats_bandit.collected_rewards[::3]
metrics = compute_metrics(rewards, [joint_env.opt(n_seeds, [0, 1, 2] * 3)[0]] * n_exp)
plot_metrics(*metrics, model_name='TS', env_name="Joint TS")
print(ts_matching_reward.sum())
print(joint_env.opt(n_seeds, [0, 1, 2] * 3))
