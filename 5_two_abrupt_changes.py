import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import random




""" ----------------- Environments ----------------- """
#   1. Environment
#   2. simulate_episode
#   3. test_seed
#   4. greedy_algorithm
#   5. MatchingEnvironment
#   6. NonStationaryEnvironment




class Environment:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward



def simulate_episode(init_prob_matrix, seeds:list, max_steps):
    prob_matrix=init_prob_matrix.copy()
    n_nodes=prob_matrix.shape[0]

    # set up seeds
    active_nodes=np.zeros(n_nodes)

    for seed in seeds:
        active_nodes[seed]=1

    history=np.array([active_nodes])

    newly_active_nodes=active_nodes

    t=0

    while (t<max_steps and np.sum(newly_active_nodes)>0):
        # retrieve probability of edge activations
        p = (prob_matrix.T*active_nodes).T
        activated_edges=p>np.random.rand(p.shape[0], p.shape[1])
        # remove activated edges
        prob_matrix=prob_matrix*((p!=0)==activated_edges)
        # update active nodes
        newly_active_nodes=(np.sum(activated_edges, axis=0)>0)*(1-active_nodes)
        # print(newly_active_nodes)
        active_nodes=np.array(active_nodes+newly_active_nodes)
        # print(active_nodes)
        history=np.concatenate((history, [newly_active_nodes]), axis=0)
        t+=1
    return history, active_nodes




def test_seed(seeds, prob_matrix, k, max_steps):
    reward = 0
    for i in range(k):
        history, active_nodes=simulate_episode(prob_matrix, seeds, max_steps)
        reward+=np.sum(active_nodes)
    return reward/k





def greedy_algorithm(init_prob_matrix, budget, k, max_steps):
    prob_matrix=init_prob_matrix.copy()
    n_nodes=prob_matrix.shape[0]

    seeds=[]
    for j in range(budget):
        print('Choosing seed ', j+1, '...')
        rewards=np.zeros(n_nodes)

        for i in tqdm(range(n_nodes)):
            if i not in seeds:
                rewards[i]=test_seed([i]+seeds, prob_matrix, k, max_steps)
        seeds.append(np.argmax(rewards))


    return seeds





class MatchingEnvironment(Environment):
    def __init__(self, reward_matrix):
        self.reward_matrix = reward_matrix
        # add a row and a column of zeros to the reward matrix to represent the case in which no match is made
        self.reward_matrix = np.hstack((self.reward_matrix, np.zeros((self.reward_matrix.shape[0], 1))))
        self.reward_matrix = np.vstack((self.reward_matrix, np.zeros((1, self.reward_matrix.shape[1]))))

        self.n_arms = reward_matrix.size
        self.t = 0

    def round(self, pulled_arms):
        try:
            rewards = [self.reward_matrix[pulled_arm] for pulled_arm in pulled_arms]
        except:
            print(pulled_arms)
        # iterate through all cells of rewards: if a cell is callable, call it and replace it with the result
        for i in range(len(rewards)):
            if callable(rewards[i]):
                rewards[i] = rewards[i]()

        return np.array(rewards)




class NonStationaryEnvironment(Environment):
    def __init__(self, probabilities, horizon):
        super().__init__(probabilities)
        self.t=0
        n_phases = len(self.probabilities)
        self.phase_size = horizon / n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward






""" ----------------- Learners ----------------- """
#   1. Learner
#   2. UCBLearner
#   3. SW_UCBLearner
#   4. CUSUM
#   5. CUSUMUCB


class Learner:

    def __init__(self, n_arms) -> None:
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.t += 1
        



class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)  # count the number of times each arm has been pulled
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward) / self.n_pulls[pulled_arm]
        for a in range(self.n_arms):

            n_samples = self.n_pulls[a]
            self.confidence[a] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf

        self.update_observations(pulled_arm, reward)



        

class SW_UCBLearner(UCBLearner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        # array showing sequence of pulled arms
        self.pulled_arms = np.array([])
        

    # get the list of unplayed arm in the last time_window
    def get_unplayed_arms(self, pulled_arms, time_window):
      all_arms = [i for i in range(self.n_arms)]
      if len(pulled_arms) < time_window:
        return list(all_arms)
      else:
        last_time_window = pulled_arms[-time_window:]
        played_arms = set(last_time_window)
        all_arms = [i for i in range(self.n_arms)]
        unplayed_arms = set(all_arms) - played_arms
        return list(unplayed_arms)


    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence

        self.upper_confidence_bound = upper_confidence_bound

        arms = [i for i in range(n_arms)] # get a list of all arms
        unplayed_arms_in_window = self.get_unplayed_arms(self.pulled_arms, self.window_size)
        # if there are unplayed arms in the most recent time window, play one of them at random
        if unplayed_arms_in_window != []:
            return random.choice(unplayed_arms_in_window)
        # else play the one with highest confidence bound
        else:
            return np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)

        for arm in range(self.n_arms):
            # count the number of times the arm has been played in the window
            n_samples = np.count_nonzero(np.array(self.pulled_arms[-window_size:]) == arm)
            # get the cumulative reward for the window if the arm was played at least once in the window
            cum_reward_in_window = np.sum(self.rewards_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            # empirical mean is computed
            self.empirical_means[arm] = cum_reward_in_window / n_samples if n_samples > 0 else 0
            # confidence is updated
            self.confidence[arm] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else 1000

        self.update_observations(pulled_arm, reward)

    def expectations(self):
        return self.empirical_means






class CUSUM:
  def __init__(self, M, eps, h):
    self.M = M
    self.eps = eps
    self.h = h # threshold
    self.t = 0
    self.reference = 0 # reference Mean
    self.g_plus = 0
    self.g_minus = 0

  def update(self, sample):
    self.t += 1


    # if time < CD window, update reference mean with new sample and return 0
    if self.t <= self.M:
      self.reference += sample/self.M
      return 0

    # if time > CD window, compute deviations and their cumulative sum
    else:
      s_plus = (sample - self.reference) - self.eps
      s_minus = -(sample - self.reference) - self.eps
      self.g_plus = max(0, self.g_plus + s_plus)
      self.g_minus = max(0, self.g_minus+ s_minus)
      # return 1 if cusum of deviations are over threshold h
      return self.g_plus > self.h or self.g_minus > self.h

  def reset(self):
    # reset the parameters if a detection occurs
    self.t = 0
    self.g_minus = 0
    self.g_plus = 0

  def Rounds_After_Last_Change(self):
    return self.t






class CUSUMUCB(UCBLearner):
    def __init__(self, n_arms, M=10, eps=0.01, h=5, alpha=0.1):
        super().__init__(n_arms)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)] 
        self.detections = [[] for _ in range(n_arms)] # list of lists of detections per arm
        self.alpha = alpha
        self.pulled_arms = np.array([])
        # initialize tau(a) as 0 for all arms.
        self.window_sizes = [0 for i in range(n_arms)]


    def pull_arm(self):
      upper_confidence_bound = self.empirical_means + self.confidence
      upper_confidence_bound[np.isinf(upper_confidence_bound)] = 1e3

      if np.random.binomial(1,1-self.alpha):
        return np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])
      else:
        return random.randint(0,n_arms-1)


    def update(self, pulled_arm, reward):
      self.t += 1
      self.pulled_arms = np.append(self.pulled_arms, pulled_arm)

      # update the change detection for the arm pulled
      if self.change_detection[pulled_arm].update(reward):
        self.change_detection[pulled_arm].reset()


      for arm in range(self.n_arms):
        # update window_sizes for each arm
        self.window_sizes[arm] = self.change_detection[arm].Rounds_After_Last_Change()
        # count the number of times the arm has been played in the window
        n_samples = np.count_nonzero(np.array(self.pulled_arms[-self.window_sizes[arm]:]) == arm)
        # get the cumulative reward for the window if the arm was played at least once in the window
        cum_reward_in_window = np.sum(self.rewards_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
        # empirical mean is computed
        self.empirical_means[arm] = cum_reward_in_window / n_samples if n_samples > 0 else 0
        # confidence is updated
        self.confidence[arm] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else 1000


      self.update_observations(pulled_arm, reward)

    def expectations(self):
      return self.empirical_means




    

""" ----------------- Clairvoyant ----------------- """
#   1. hungarian_algorithm
#   2. get_reward
#   3. clairvoyant


def hungarian_algorithm(matrix):
    m = matrix.copy()
    n_rows, n_cols = m.shape
    max_val = np.max(m)

    if n_rows > n_cols:
        m = np.pad(m, ((0, 0), (0, n_rows - n_cols)), mode='constant', constant_values=max_val)
    elif n_cols > n_rows:
        m = np.pad(m, ((0, n_cols - n_rows), (0, 0)), mode='constant', constant_values=max_val)

    assigned_rows, assigned_cols = linear_sum_assignment(m)

    assignment = np.zeros_like(m, dtype=int)
    assignment[assigned_rows, assigned_cols] = 1

    return assignment[:n_rows, :n_cols] * matrix, assignment[:n_rows, :n_cols]



def get_reward(node_class, product_class, rewards_parameters):
    return 100-np.random.normal(rewards_parameters[0][node_class-1, product_class], rewards_parameters[1][node_class-1, product_class])




def clairvoyant(graph_probabilities, node_classes, rewards_parameters, n_exp):

    rewards_per_exp=np.zeros(n_exp)
    opt_seeds=greedy_algorithm(graph_probabilities, 3, 1000, 50)

    for i in tqdm(range(n_exp)):
        active_nodes=simulate_episode(graph_probabilities, opt_seeds, 50)[1]*node_classes

        # remove all zeros from array (nodes that were not activated)
        active_nodes=active_nodes[active_nodes!=0]

        # compute rewards
        hun_matrix_dim=max(len(active_nodes), 9)

        rewards=np.zeros((hun_matrix_dim, hun_matrix_dim))

        for j in range(hun_matrix_dim):
            for l in range(hun_matrix_dim):
                if j<len(active_nodes) and l<9:
                    rewards[j,l]=get_reward(int(active_nodes[j]), l//3, rewards_parameters)
                else:
                    rewards[j,l]=0

        optimum= hungarian_algorithm(rewards)[0]
        optimum= 100-optimum # convert to reward
        optimum[optimum==100]=0
        rewards_per_exp[i]=np.sum(optimum)
    return np.mean(rewards_per_exp), np.std(rewards_per_exp)




""" ----------------- SCENARIO SET UP ----------------- """


# define scenario parameters
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

# edge activation probabilities for the three phases.
prob_phase1 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))
prob_phase2 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))
prob_phase3 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))

# array containing three (30*30) different probabilities tables.
p = np.stack((prob_phase1, prob_phase2, prob_phase3), axis=0)


# array K will contain 30 arrays containing each 3 rows. The 3 rows are row[i] of probability table of phase1,
# row[i] of the one of phase2, row[i] of the one of phase3.
K = np.array([p[:, i] for i in range(p.shape[1])])






""" ----------------- Probability Estimators ----------------- """
#   1. SW_Generate_Probability_Estimates
#   2. CUSUM_Generate_Probability_Estimates

def SW_Generate_Probability_Estimates(p, n_arms=n_arms, n_phases=n_phases, T=T, window_size=window_size, n_experiments=n_experiments):
    phases_len = int(T / n_phases)
    swucb_rewards_per_experiment = []

    experimentS_means_at_each_round = np.empty((n_experiments, T, n_arms))

    for e in tqdm(range(0, n_experiments)):
        # initialize environment
        swucb_env = NonStationaryEnvironment(probabilities=p, horizon=T)
        # initialize learner
        swucb_learner = SW_UCBLearner(n_arms=n_arms, window_size=window_size)

        for t in range(0, T):
            # SW-UCB Learner
            pulled_arm = swucb_learner.pull_arm()
            reward = swucb_env.round(pulled_arm)
            swucb_learner.update(pulled_arm, reward)

            # at each round memorize a copy of the means of each arm
            expected_rew = swucb_learner.expectations()
            experimentS_means_at_each_round[e, t] = expected_rew.copy()

    return experimentS_means_at_each_round




def CUSUM_Generate_Probability_Estimates(p, n_arms=n_arms, n_phases=n_phases, T=T, n_experiments=n_experiments):
    phases_len = int(T / n_phases)
    cusum_rewards_per_experiment = []

    experimentS_means_at_each_round = np.empty((n_experiments, T, n_arms))

    for e in tqdm(range(0, n_experiments)):
        # initialize environment
        cusum_env = NonStationaryEnvironment(probabilities=p, horizon=T)
        # initialize learner
        cusum_learner = CUSUMUCB(n_arms=n_arms)

        for t in range(0, T):
            # SW-UCB Learner
            pulled_arm = cusum_learner.pull_arm()
            reward = cusum_env.round(pulled_arm)
            cusum_learner.update(pulled_arm, reward)

            # at each round memorize a copy of the means of each arm
            expected_rew = cusum_learner.expectations()
            experimentS_means_at_each_round[e, t] = expected_rew.copy()

    return experimentS_means_at_each_round




""" --- Estimating edges' probabilities with SW --- """

SW_rounds_probabilities_for_each_arm = []

for index in range(len(K)):

  print("Learning for row:",index)
  estimates = SW_Generate_Probability_Estimates(K[index])
  SW_rounds_probabilities_for_each_arm.append(estimates)

SW_rounds_probabilities_for_each_arm = np.mean(SW_rounds_probabilities_for_each_arm, axis=1)




""" --- Estimating edges' probabilities with CD --- """

CUSUM_rounds_probabilities_for_each_arm = []

for index in range(len(K)):

  print("Learning for row:",index)
  estimates = CUSUM_Generate_Probability_Estimates(K[index])
  CUSUM_rounds_probabilities_for_each_arm.append(estimates)

CUSUM_rounds_probabilities_for_each_arm = np.mean(CUSUM_rounds_probabilities_for_each_arm, axis=1)

def Reshape(LIST):
  # convert the lists into a np.array
  array_of_lists = np.array(LIST)
  # transpose the array to swap the axes
  transposed_array = array_of_lists.T
  # split the transposed array into separate arrays along axis=1
  return np.split(transposed_array, transposed_array.shape[1], axis=1)


# the output is a list of 365 NumPy arrays representing the evolution of the estimated probability table at each round.
estimated_tables_SW = Reshape(SW_rounds_probabilities_for_each_arm)
estimated_tables_CUSUM = Reshape(CUSUM_rounds_probabilities_for_each_arm)





""" ----------------- COMPUTING REWARDS ----------------- """

n_exp = 10

SW_mean_rewards_per_round = []
SW_std_dev_rewards_per_round = []
for table in tqdm(range(len(estimated_tables_SW))):
  table = np.reshape(estimated_tables_SW[table],(30,30))
  clairvoyant_output = clairvoyant(table, node_classes, rewards_parameters, n_exp)
  SW_mean_rewards_per_round.append(clairvoyant_output[0])
  SW_std_dev_rewards_per_round.append(clairvoyant_output[1])


  

""" --- Reward - CUSUM_UCB --- """

CUSUM_mean_rewards_per_round = []
CUSUM_std_dev_rewards_per_round = []
for table in range(len(estimated_tables_CUSUM)):
  table = np.reshape(estimated_tables_CUSUM[table],(30,30))
  clairvoyant_output = clairvoyant(table, node_classes, rewards_parameters, n_exp)
  CUSUM_mean_rewards_per_round.append(clairvoyant_output[0])
  CUSUM_std_dev_rewards_per_round.append(clairvoyant_output[1])





""" --- Reward - Clairvoyant (Optimum) --- """

optimum_means = []
optimum_std_dev = []
for table in p:
  clairvoyant_output = clairvoyant(table, node_classes, rewards_parameters, 25)
  for i in range(int(T / n_phases)+1):
    optimum_means.append(clairvoyant_output[0])
    optimum_std_dev.append(clairvoyant_output[1])

optimum_means = optimum_means[:-1]
optimum_std_dev = optimum_std_dev[:-1]


    

""" ----------------- Plotting Results ----------------- """

#----------------------------------------------------------
""" - Istantaneous reward of Sliding Window UCB - """

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

#----------------------------------------------------------

""" - Istantaneous reward of Change Detection CUSUM UCB - """

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

#----------------------------------------------------------


""" - Comparison of Istantaneous Rewards - """

time_periods = range(len(SW_mean_rewards_per_round))

plt.plot(time_periods, SW_mean_rewards_per_round, color='red', linestyle='-', label = "SW_UCB")
plt.plot(time_periods, CUSUM_mean_rewards_per_round, color="blue",linestyle="-", label = "CD_UCB")

plt.xlabel('Time')
plt.ylabel('Mean Reward Per Round')
plt.title('Istantaneous Rewards')

plt.xticks(time_periods[::30])

plt.figure(figsize=(10, 6))

plt.show()

#----------------------------------------------------------


""" - Istantaneous Regrets - """

time_periods = range(len(SW_mean_rewards_per_round))

plt.plot(time_periods, [x - y for x, y in zip(optimum_means, SW_mean_rewards_per_round)], color='red', linestyle='-', label = "SW_UCB")
plt.plot(time_periods, [x - y for x, y in zip(optimum_means, CUSUM_mean_rewards_per_round)], color="blue",linestyle="-", label = "CD_UCB")

plt.xlabel('Time')
plt.ylabel('Regret per Round')
plt.title('Istantaneous Regrets')

plt.xticks(time_periods[::30])

plt.figure(figsize=(10, 6))

plt.show()

#----------------------------------------------------------

""" - Cumulative Regrets - """

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

