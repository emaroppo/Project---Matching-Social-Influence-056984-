import numpy as np
from learners.ts_learners.matching_ts import TSMatchingContext
from learners.ts_learners.ts_learner import TSProbLearner
from environments.joint_environment import JointEnvironment
from environments.matching_environment import MatchingEnvironment2
from utils.updated_context import ContextGenerationAlgorithm
from utils.simulation import (
    influence_simulation,
    merged_matching_simulation_with_features,
)
from tqdm import tqdm
from utils.metrics import compute_metrics, plot_metrics
from utils.data_generator import generate_graph, generate_reward_parameters

# init reward matrix, graph probabilities
n_nodes = 30
edge_rate = 0.2

n_seeds = 3
n_customer_classes = 3
n_product_classes = 3
products_per_class = 3
n_exp = 20

reward_means, reward_std_dev = generate_reward_parameters(
    n_customer_classes, n_product_classes
)
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate)
# link node ids to customer classes

node_features = np.random.binomial(1, 0.5, (30, 2))
# class mapping
class_0 = [0, 1]
class_1 = [1, 1]

# features to class labels
node_classes = []
for i in range(n_nodes):
    if list(node_features[i]) == class_0:
        node_classes.append(0)
    elif list(node_features[i]) == class_1:
        node_classes.append(1)

    else:
        node_classes.append(2)

print(node_classes)


# Adapting the step_4_context_generation_with_merged_simulation function to compute and plot metrics


def step_4_context_generation_with_metrics(
    node_features,
    node_classes,
    reward_parameters,
    graph_probabilities,
    graph_structure,
    context_generator,
    n_nodes=30,
    n_seeds=3,
    n_product_classes=3,
    products_per_class=3,
    n_exp=20,
    n_phases=1,
    joint=True,
    context_update_frequency=14,
    use_context_updates=True,
    model_names=None,
    env_name=None,
):
    # Extracting reward parameters
    reward_means, reward_std_dev = reward_parameters

    # initialise bandit
    ts_bandit = TSProbLearner(n_nodes, n_seeds, graph_structure=graph_structure)
    ts_matching = TSMatchingContext(
        n_product_classes * 1, 1, n_product_classes, products_per_class
    )

    # initialise environment
    joint_env = JointEnvironment(
        graph_probabilities, (reward_means, reward_std_dev), node_classes
    )
    matching_env = MatchingEnvironment2((reward_means, reward_std_dev))

    # Influence Simulation
    (
        all_mean_rewards,
        all_optimal_rewards,
        all_models_active_nodes,
    ) = influence_simulation(
        joint_env.social_environment, [ts_bandit], n_exp, n_phases=1, joint=True
    )

    # Compute metrics for Influence Simulation
    (
        rewards_influence,
        opt_rewards_influence,
        inst_regret_influence,
        cum_regret_influence,
        trend_influence,
    ) = compute_metrics(all_mean_rewards, all_optimal_rewards)

    # Matching Simulation using merged_matching_simulation
    (
        all_collected_rewards,
        all_optimal_matching_rewards,
    ) = merged_matching_simulation_with_features(
        matching_env,
        ts_matching,
        n_exp,
        node_features,
        use_context_updates=use_context_updates,
        context_generator=context_generator,
        active_nodes=all_models_active_nodes,
        class_mapping=node_classes,
        product_classes=list(range(n_product_classes)),
        products_per_class=products_per_class,
        context_update_frequency=context_update_frequency,
    )

    # Compute metrics for Matching Simulation
    (
        rewards_matching,
        opt_rewards_matching,
        inst_regret_matching,
        cum_regret_matching,
        trend_matching,
    ) = compute_metrics(all_collected_rewards, all_optimal_matching_rewards)

    # Plot metrics for both simulations
    plot_metrics(
        [rewards_influence, rewards_matching],
        [opt_rewards_influence, opt_rewards_matching],
        [inst_regret_influence, inst_regret_matching],
        [cum_regret_influence, cum_regret_matching],
        model_names=model_names,
        env_name=env_name,
    )

    # Return relevant data/results
    return (
        all_mean_rewards,
        all_optimal_rewards,
        all_models_active_nodes,
        all_collected_rewards,
        all_optimal_matching_rewards,
    )


context_generator = ContextGenerationAlgorithm(node_features, [0, 1, 2])

# The function is updated with metric computations and plotting.
# Note: The function will not be executed here due to the missing dependencies.
# However, it can be executed in an environment where these dependencies are available.

# execute step_4_context_generation_with_metrics
(
    all_mean_rewards,
    all_optimal_rewards,
    all_models_active_nodes,
    all_collected_rewards,
    all_optimal_matching_rewards,
) = step_4_context_generation_with_metrics(
    node_classes=node_classes,
    node_features=node_features,
    reward_parameters=(reward_means, reward_std_dev),
    graph_probabilities=graph_probabilities,
    graph_structure=graph_structure,
    context_generator=context_generator,
    n_nodes=n_nodes,
    n_seeds=n_seeds,
    n_product_classes=n_product_classes,
    products_per_class=products_per_class,
    n_exp=n_exp,
    n_phases=1,
    joint=False,
    context_update_frequency=14,
    use_context_updates=True,
    model_names=["TS", "TS"],
    env_name="Joint (Social) TS",
)
