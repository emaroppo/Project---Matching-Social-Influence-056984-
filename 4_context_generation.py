import numpy as np
import matplotlib.pyplot as plt
from learners.ts_learners.matching_ts import TSMatching4
from learners.ts_learners.ts_learner import TSProbLearner
from environments.joint_environment import JointEnvironment
from environments.matching_environment import MatchingEnvironment2
from context import ContextGenerationAlgorithm
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

node_features = np.random.binomial(1, 0.5, (30, 2))
#class mapping
class_0= [0,1]
class_1= [1,1]

#features to class labels
node_classes = []
for i in range(n_nodes):
    if list(node_features[i]) == class_0:
        node_classes.append(0)
    elif list(node_features[i]) == class_1:
        node_classes.append(1)

    else:
        node_classes.append(2)

print(node_classes)



# initialise bandit
ts_bandit = TSProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)
ts_matching = TSMatching4(n_product_classes*1, 1, n_product_classes, products_per_class)

# initialise environment
joint_env = JointEnvironment(
    graph_probabilities, (reward_means, reward_std_dev), node_classes
)
matching_env = MatchingEnvironment2((reward_means, reward_std_dev))

expected_social_rewards = []
# opt_social_reward = joint_env.social_environment.opt(n_seeds)[1]
expected_matching_rewards = []
optimal_matching_rewards = []

context_generator = ContextGenerationAlgorithm(2, [0,1,2])
dataset = []
for m in tqdm(range(n_exp)[5:]):

    

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

    if m>13:
        rules= context_generator.context_structures[-1].create_rules()
        mapping=context_generator.context_structures[-1].create_mapping(node_features, rules)
    else:
        mapping = [0]*n_nodes
    
    active_nodes_id = np.array(active_nodes)
    # convert to list of integer indices corresponding to the position of activated nodes in the reward matrix
    active_nodes_id = np.where(active_nodes == 1)[-1]
    active_estimated_classes = [mapping[i] for i in active_nodes_id]

    # convert to list of classes
    active_real_classes = [node_classes[i] for i in active_nodes_id]

    # perform matching assuming customer classes are known but distributions unknown
    ts_prop_match = ts_matching.pull_arms(active_nodes_id, active_estimated_classes)
    # retrieve reward
    ts_matching_reward = matching_env.round(ts_prop_match)

    dataset_entry_features=[node_features[i[0]] for i in ts_matching_reward]
    
    dataset_entry_rewards= [np.zeros(4) for i in ts_matching_reward]
    for i,j in enumerate(dataset_entry_rewards):
        j[ts_prop_match[i][2]]= ts_matching_reward[i][1]

    #merge features and rewards
    #features are the first 3 columns, rewards the last 3
    dataset_entry = np.concatenate((dataset_entry_features, dataset_entry_rewards), axis=1)

    dataset.append(dataset_entry)

    expected_matching_reward = [
        matching_env.round(ts_prop_match) for _ in range(1000)
    ]
    expected_matching_reward = [expected_matching_reward[i][1] for i in range(len(expected_matching_reward))]

    expected_matching_reward = np.mean(expected_matching_reward)

    optimal_matching_reward = joint_env.matching_environment.opt(
        active_real_classes, [0, 1, 2] * 3
    )
    optimal_matching_rewards.append(optimal_matching_reward)

    expected_matching_rewards.append(np.mean(expected_matching_reward))
    # update bandit

    ts_matching.update(ts_prop_match, ts_matching_reward)

    if (m+1) % 14 == 0:
        context_generator.update(dataset)
        ts_matching.resize_arms(context_generator.context_structures[-1])

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

