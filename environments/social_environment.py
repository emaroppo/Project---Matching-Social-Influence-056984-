from environments.environment import Environment
from tqdm import tqdm
import numpy as np


class SocialEnvironment(Environment):
    def __init__(self, probabilities):
        super().__init__(probabilities)

    def simulate_episode(self, seeds: list, max_steps=100, prob_matrix=None, n_runs=1):
        prob_matrix = (
            prob_matrix.copy() if prob_matrix is not None else self.probabilities.copy()
        )

        if n_runs > 1:
            prob_matrix = np.tile(prob_matrix[np.newaxis, :, :], (n_runs, 1, 1))
        else:
            prob_matrix = prob_matrix[np.newaxis, :, :]

        n_nodes = prob_matrix.shape[1]

        # Initialize the arrays
        susceptible_edges_final = np.zeros((n_runs, n_nodes, n_nodes), dtype=int)
        activated_edges_final = np.zeros((n_runs, n_nodes, n_nodes), dtype=int)

        # set up seeds
        active_nodes = np.zeros((n_runs, n_nodes), dtype=int)
        for seed in seeds:
            active_nodes[:, seed] = True

        newly_active_nodes = active_nodes.copy()
        t = 0
        while t < max_steps and np.sum(newly_active_nodes, axis=1).any():
            susceptible_edges = np.einsum(
                "ij,ik->ijk", newly_active_nodes, 1 - active_nodes
            ).astype(bool)

            susceptible_edges[prob_matrix == 0] = False

            susceptible_prob = prob_matrix * susceptible_edges

            activated_edges = np.random.binomial(1, susceptible_prob).astype(bool)

            susceptible_edges_final |= susceptible_edges
            activated_edges_final |= activated_edges

            # Update the nodes
            newly_active_nodes = np.any(activated_edges, axis=1)
            # print coords i,j,k of newly active nodes
            # print idx of newly active nodes

            active_nodes = np.logical_or(active_nodes, newly_active_nodes)

            t += 1

        # If n_runs is 1, remove the extra axis
        if n_runs == 1:
            active_nodes = np.squeeze(active_nodes, axis=0)
            susceptible_edges_final = np.squeeze(susceptible_edges_final, axis=0)
            activated_edges_final = np.squeeze(activated_edges_final, axis=0)

        episode = (susceptible_edges_final, activated_edges_final)
        return episode, active_nodes

    def round(self, pulled_arms, joint=False):
        episode, active_nodes = self.simulate_episode(pulled_arms, n_runs=1)

        if joint:
            return episode, active_nodes

        reward = np.sum(active_nodes)
        return episode, reward

    def expected_reward(self, arm, n_runs=100):
        _, active_nodes = self.simulate_episode(arm, n_runs=n_runs)

        # Calculate mean and standard deviation
        mean_reward = np.mean(np.sum(active_nodes, axis=1))
        std_reward = np.std(np.sum(active_nodes, axis=1))

        return mean_reward, std_reward

    def opt_arm(self, budget, k=100, max_steps=100):
        prob_matrix = self.probabilities.copy()
        n_nodes = prob_matrix.shape[0]

        seeds = list()

        for j in range(budget):
            rewards = np.zeros(n_nodes)
            std_devs = np.zeros(
                n_nodes
            )  # Store standard deviations for possible future use

            seeds_set = set(seeds)

            # Iterate only over nodes not in seeds
            for i in set(range(n_nodes)) - seeds_set:
                rewards[i], std_devs[i] = self.expected_reward([i] + seeds, n_runs=k)

            chosen_seed = np.argmax(rewards)
            seeds.append(chosen_seed)

        return seeds

    def opt(self, n_seeds, n_exp=1000, exp_per_seed=100, max_steps=100):
        opt_seeds = self.opt_arm(n_seeds, exp_per_seed, max_steps)
        mean_reward, std_dev = self.expected_reward(opt_seeds, n_runs=n_exp)

        self.opt_value = (
            mean_reward,
            std_dev,
            opt_seeds,
        )
        return self.opt_value
