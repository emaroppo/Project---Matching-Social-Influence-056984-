from environments.environment import Environment
from environments.social_environment import SocialEnvironment
from environments.matching_environment import MatchingEnvironment
from environments.ns_environment import NonStationaryEnvironment
from learners.learner import Learner
from learners.ucb_learners.ucb_learner import UCBLearner
from learners.ucb_learners.matching_ucb import UCBMatching


from learners.ucb_learners.CUSUM_CUSUMUCB import CUSUM, CUSUMUCB, CUSUM_Generate_Probability_Estimates
from learners.ucb_learners.SW_UCB import SW_UCBLearner, SW_Generate_Probability_Estimates

from environments.NS_Episode_Simulation import simulate_episode, test_seed, greedy_algorithm
from environments.NS_Matching_Clairvoyant import hungarian_algorithm, get_reward, clairvoyant


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import random


""" Set up the environment parameters """

# assign customers to classes
customer_assignments = np.random.choice([0,1,2], size=30)

# define environment parameters
node_classes = 3
product_classes = 3
products_per_class = 3

means = np.random.uniform(10, 20, (3,3))
std_dev = np.ones((3,3))
rewards_parameters = (means, std_dev)


n_arms = 30
n_phases = 3,
T = 365
window_size = int(T**0.5)
n_experiments = 50

def generate_graph_probabilities(n_nodes, edge_rate):
    graph_structure = np.random.binomial(1, edge_rate, (n_nodes, n_nodes))
    graph_probabilities = np.random.uniform(0.1, 0.9, (n_nodes, n_nodes)) * graph_structure
    return graph_probabilities



n_nodes = 30
edge_rate = 0.05
n_phases = 3

prob_phase1 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))
prob_phase2 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))
prob_phase3 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))

# array containing three (30*30) different probabilities tables.
p = np.stack((prob_phase1, prob_phase2, prob_phase3), axis=0)


# array K will contain 30 arrays containing each 3 rows: row[i] of probability table of phase1, row[i] of the one of phase2, row[i] of the one of phase3.
K = np.array([p[:, i] for i in range(p.shape[1])])





""" Find estimates of edges activation probability with SW_UCB """

SW_rounds_probabilities_for_each_arm = []

for index in range(len(K)):

  print("Learning for row:",index)
  estimates = SW_Generate_Probability_Estimates(K[index])
  SW_rounds_probabilities_for_each_arm.append(estimates)

SW_rounds_probabilities_for_each_arm = np.mean(SW_rounds_probabilities_for_each_arm, axis=1)




""" Find estimates of edges activation probability with CD_UCB """

CUSUM_rounds_probabilities_for_each_arm = []

for index in range(len(K)):

  print("Learning for row:",index)
  estimates = CUSUM_Generate_Probability_Estimates(K[index])
  CUSUM_rounds_probabilities_for_each_arm.append(estimates)

CUSUM_rounds_probabilities_for_each_arm = np.mean(CUSUM_rounds_probabilities_for_each_arm, axis=1)





""" Reshape estimates so to obtain a different probability table for each round. """

def Reshape(LIST):
  # Convert the lists into a NumPy array
  array_of_lists = np.array(LIST)
  # Transpose the array to swap the axes
  transposed_array = array_of_lists.T
  # Split the transposed array into separate arrays along axis=1
  return np.split(transposed_array, transposed_array.shape[1], axis=1)

# The output is a list of 365 NumPy arrays representing the evolution of the estimated probability table at each round.
estimated_tables_SW = Reshape(SW_rounds_probabilities_for_each_arm)
estimated_tables_CUSUM = Reshape(CUSUM_rounds_probabilities_for_each_arm)







"""Simulate the assignments of items to customers through Greedy algorithm and Clairvoyant function based on the SW_UCB estimated probabilities."""

n_exp = 10

SW_mean_rewards_per_round = []
SW_std_dev_rewards_per_round = []
for table in tqdm(range(len(estimated_tables_SW))):
  table = np.reshape(estimated_tables_SW[table],(30,30))
  clairvoyant_output = clairvoyant(table, customer_assignments, rewards_parameters, n_exp)
  SW_mean_rewards_per_round.append(clairvoyant_output[0])
  SW_std_dev_rewards_per_round.append(clairvoyant_output[1])




"""Simulate the assignments of items to customers through Greedy algorithm and Clairvoyant function based on the SW_UCB estimated probabilities."""

CUSUM_mean_rewards_per_round = []
CUSUM_std_dev_rewards_per_round = []
for table in range(len(estimated_tables_CUSUM)):
  table = np.reshape(estimated_tables_CUSUM[table],(30,30))
  clairvoyant_output = clairvoyant(table, customer_assignments, rewards_parameters, n_exp=25)
  CUSUM_mean_rewards_per_round.append(clairvoyant_output[0])
  CUSUM_std_dev_rewards_per_round.append(clairvoyant_output[1])
  


"""Simulate the assignments of items to customers through Greedy algorithm and Clairvoyant function based on the REAL edge probabilities."""

optimum_means = []
optimum_std_dev = []
for table in p:
  clairvoyant_output = clairvoyant(table, customer_assignments, rewards_parameters, 25)
  for i in range(int(T / n_phases)+1):
    optimum_means.append(clairvoyant_output[0])
    optimum_std_dev.append(clairvoyant_output[1])

optimum_means = optimum_means[:-1]
optimum_std_dev = optimum_std_dev[:-1]




""" Plot istantaneous reward for SW_UCB """

time_periods = range(len(SW_mean_rewards_per_round))

for t in time_periods:
    mean = SW_mean_rewards_per_round[t]
    std_dev = SW_std_dev_rewards_per_round[t]
    plt.vlines(t, mean - std_dev, mean + std_dev, color='lightgrey')


plt.plot(time_periods, optimum_means, color='green', linestyle='-')
plt.plot(time_periods, SW_mean_rewards_per_round, color='red', linestyle='-')



plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Sliding Window UCB')

plt.xticks(time_periods[::30])

plt.figure(figsize=(10, 6))

plt.show()




""" Plot istantaneous reward for CUSUM_UCB """

time_periods = range(len(CUSUM_mean_rewards_per_round))

for t in time_periods:
    mean = CUSUM_mean_rewards_per_round[t]
    std_dev = CUSUM_std_dev_rewards_per_round[t]
    plt.vlines(t, mean - std_dev, mean + std_dev, color='lightgrey')


plt.plot(time_periods, optimum_means, color='green', linestyle='-')
plt.plot(time_periods, CUSUM_mean_rewards_per_round, color='blue', linestyle='-')



plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Change Detection CUSUM UCB')

plt.xticks(time_periods[::30])

plt.figure(figsize=(10, 6))

plt.show()




""" Plot istantaneous regrets for both SW_UCB and CUSUM_UCB """

time_periods = range(len(SW_mean_rewards_per_round))

plt.plot(time_periods, [x - y for x, y in zip(optimum_means, SW_mean_rewards_per_round)], color='red', linestyle='-', label = "SW_UCB")
plt.plot(time_periods, [x - y for x, y in zip(optimum_means, CUSUM_mean_rewards_per_round)], color="blue",linestyle="-", label = "CD_UCB")

plt.xlabel('Time')
plt.ylabel('Regret per Round')
plt.title('Istantaneous Regrets')

plt.xticks(time_periods[::30])

plt.figure(figsize=(10, 6))

plt.show()



""" Plot cumulative regrets for both SW_UCB and CUSUM_UCB """

SW_cumulative_regret = [sum([x - y for x, y in zip(optimum_means[:t+1], SW_mean_rewards_per_round[:t+1])]) for t in time_periods]
CD_cumulative_regret = [sum([x - y for x, y in zip(optimum_means[:t+1], CUSUM_mean_rewards_per_round[:t+1])]) for t in time_periods]

plt.plot(time_periods, SW_cumulative_regret, color='red', linestyle='-', label="SW_UCB")
plt.plot(time_periods, CD_cumulative_regret, color="blue", linestyle="-", label="CD_UCB")

plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regrets')

plt.xticks(time_periods[::30])
plt.legend()

plt.figure(figsize=(10, 6))
plt.show()







