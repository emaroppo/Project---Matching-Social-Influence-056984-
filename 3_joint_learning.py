import numpy as np
import matplotlib.pyplot as plt
from learners.ts_learners.matching_ts import TSMatching
from learners.ts_learners.ts_learner import TSProbLearner
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
n_exp = 365

reward_means, reward_std_dev = generate_reward_parameters(
    n_customer_classes, n_product_classes
)
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
# link node ids to customer classes
node_classes = np.random.randint(0, n_customer_classes, n_nodes)


def step_3_v2(
    graph_probabilities,
    n_nodes,
    n_seeds,
    reward_means,
    reward_std_dev,
    node_classes,
    n_product_classes,
    n_customer_classes,
    products_per_class,
    n_exp,
):
    # initialise environments
    joint_env = JointEnvironment(
        graph_probabilities, (reward_means, reward_std_dev), node_classes
    )

    maucb_bandit = JointMAUCBLearner(
        n_nodes, n_seeds, n_customer_classes, n_product_classes, products_per_class
    )
    matching_ucb = UCBMatching(
        n_customer_classes, n_product_classes, products_per_class
    )

    mats_bandit = JointTSLearner(
        n_nodes, n_seeds, n_customer_classes, n_product_classes, products_per_class
    )
    matching_ts = TSMatching(n_customer_classes, n_product_classes, products_per_class)

    for i in tqdm(range(n_exp)):
        ucb_pulled_arms = maucb_bandit.pull_arms()
        ts_pulled_arms = mats_bandit.pull_arms()

        # retrieve activated nodes
        ucb_social_reward = joint_env.social_environment.round(
            ucb_pulled_arms, joint=True
        )
        ts_social_reward = joint_env.social_environment.round(
            ts_pulled_arms, joint=True
        )

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

    return maucb_bandit, mats_bandit, matching_ucb, matching_ts, joint_env


"""
maucb_bandit, mats_bandit, matching_ucb, matching_ts, joint_env = step_3_v2(n_exp)


rewards = maucb_bandit.collected_rewards[::3]
metrics = compute_metrics(rewards, [joint_env.opt(n_seeds, [0, 1, 2] * 3)[0]] * n_exp)
plot_metrics(*metrics, model_name="UCB", env_name="Joint UCB")
rewards = mats_bandit.collected_rewards[::3]
metrics = compute_metrics(rewards, [joint_env.opt(n_seeds, [0, 1, 2] * 3)[0]] * n_exp)
plot_metrics(*metrics, model_name="TS", env_name="Joint TS")
"""
# initialise bandit
ts_bandit = TSProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)
ts_matching = TSMatching(n_customer_classes, n_product_classes, products_per_class)

# initialise environment
joint_env = JointEnvironment(
    graph_probabilities, (reward_means, reward_std_dev), node_classes
)

expected_social_rewards = []
# opt_social_reward = joint_env.social_environment.opt(n_seeds)[1]
expected_matching_rewards = []
optimal_matching_rewards = []


for i in tqdm(range(n_exp)):
    # pull arm
    ts_pulled_arm = ts_bandit.pull_arm()

    # retrieve episode
    ts_social_reward, active_nodes = joint_env.social_environment.round(
        ts_pulled_arm, joint=True
    )

    ts_bandit.update(ts_social_reward)
    expected_social_reward = [
        joint_env.social_environment.round(ts_pulled_arm, joint=True)[1].sum()
        for _ in range(1000)
    ]

    expected_social_rewards.append(np.mean(expected_social_reward))

    active_classes = np.array(active_nodes)
    # convert to list of integer indices corresponding to the position of activated nodes in the reward matrix
    active_classes = np.where(active_nodes == 1)[-1]
    # convert to list of classes
    active_classes = [node_classes[i] for i in active_classes]

    # perform matching assuming customer classes are known but distributions unknown
    ts_prop_match = ts_matching.pull_arms(active_classes)
    # retrieve reward
    ts_matching_reward = joint_env.matching_environment.round(ts_prop_match)
    expected_matching_reward = [
        joint_env.matching_environment.round(ts_prop_match).sum() for _ in range(1000)
    ]
    optimal_matching_reward = joint_env.matching_environment.opt(
        active_classes, [0, 1, 2] * 3
    )
    optimal_matching_rewards.append(optimal_matching_reward)

    expected_matching_rewards.append(np.mean(expected_matching_reward))
    # update bandit

    ts_matching.update(ts_prop_match, ts_matching_reward)

# plot expected social rewards
metrics = compute_metrics(
    np.array(expected_social_rewards),
    np.array([joint_env.social_environment.opt(n_seeds)[0]] * n_exp),
)
plot_metrics(*metrics, model_name="TS", env_name="Joint (Social) TS")

# plot expected matching rewards
expected_matching_rewards = np.array(expected_matching_rewards) / np.array(
    optimal_matching_rewards
)
metrics = compute_metrics(expected_matching_rewards, opt_rewards=np.ones(n_exp))
plot_metrics(*metrics, model_name="TS", env_name="Joint (Matching) TS")

# check if it should maximise rewards (social & matching) separetely or jointly
