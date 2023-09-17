from environments.ns_environment import SocialNChanges
from learners.ucb_learners.ucb_learner import UCBProbLearner
from learners.ucb_learners.ns_ucb import SWUCBProbLearner, CDUCBProbLearner
import numpy as np
from data_generator import generate_graph
from metrics import compute_metrics, plot_metrics
from tqdm import tqdm

n_nodes = 30
edge_rate = 0.2
n_phases = 2
n_exp= 365
graph_probabilities, graph_structure = generate_graph(n_nodes, edge_rate, n_phases=3)

# parameters for (gaussian) reward distributions for each node class and product class
means = np.random.uniform(10, 20, (3, 3))
stds = np.random.uniform(1, 3, (3, 3))
reward_parameters = (means, stds)

n_episodes = 365

env=SocialNChanges(graph_probabilities, n_phases=n_phases)
model=SWUCBProbLearner(30,3,121,graph_structure=graph_structure)

mean_rewards = []
max_ = env.opt(3)
optimal_reward = [] 

for i in tqdm(range(n_episodes)):

    
    pulled_arm = model.pull_arm()
    episode, rew, change = env.round(pulled_arm)
    if change:
        print('change at t=', i)
        max_ = env.opt(3)
    optimal_reward.append(max_[0])
    exp_reward = env.expected_reward(pulled_arm, 100)[0]
    mean_rewards.append(exp_reward)

    regret = max_[0] - exp_reward
    if regret > 0.5:
        print("Regret: ", regret)

    model.update(episode)

metrics = compute_metrics(np.array(mean_rewards)/np.array(optimal_reward), [1]*n_episodes)
plot_metrics(*metrics, model_name="SWUCBProbLearner", env_name="Social Environment")