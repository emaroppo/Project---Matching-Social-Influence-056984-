from environments.environment import Environment
from tqdm import tqdm
import numpy as np


class SocialEnvironment(Environment):
    def __init__(self, probabilities):
        super().__init__(probabilities)

    def simulate_episode(self, seeds: list, max_steps=100, prob_matrix=None):
        prob_matrix = (
            prob_matrix.copy() if prob_matrix is not None else self.probabilities.copy()
        )
        n_nodes = prob_matrix.shape[0]

        # Initialize the arrays
        active_nodes_final = np.zeros(n_nodes)
        susceptible_edges_final = np.zeros((n_nodes, n_nodes))
        activated_edges_final = np.zeros((n_nodes, n_nodes))

        # set up seeds
        active_nodes = np.zeros(n_nodes)

        for seed in seeds:
            active_nodes[seed] = 1

        newly_active_nodes = active_nodes
        t = 0

        while t < max_steps and np.sum(newly_active_nodes) > 0:
            # retrieve probability of edge activations
            mask = np.outer(np.ones(prob_matrix.shape[0]), active_nodes)
            modified_prob_matrix = prob_matrix * (1 - mask)

            # Calculate edges that might be activated
            p = (modified_prob_matrix.T * newly_active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])

            # Update final arrays
            activated_edges_final += activated_edges
            susceptible_edges_final += np.outer(newly_active_nodes, 1 - active_nodes)

            # remove activated edges
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)

            # update active nodes
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (
                1 - active_nodes
            )
            active_nodes_final += newly_active_nodes
            active_nodes = np.array(active_nodes + newly_active_nodes)

            t += 1

        # Convert the final arrays to binary (0 or 1)
        active_nodes_final = np.where(active_nodes_final > 0, 1, 0)
        susceptible_edges_final = np.where(susceptible_edges_final > 0, 1, 0)
        activated_edges_final = np.where(activated_edges_final > 0, 1, 0)
        episode = (susceptible_edges_final, activated_edges_final)

        return episode, active_nodes

    def round(self, pulled_arms, joint=False):
        episode, active_nodes = self.simulate_episode(pulled_arms)

        if joint:
            return episode, active_nodes

        reward = np.sum(active_nodes)
        return episode, reward

    def opt_arm(self, budget, k=100, max_steps=100):
        prob_matrix = self.probabilities.copy()
        n_nodes = prob_matrix.shape[0]

        seeds = list()
        for j in range(budget):
            # print("Choosing seed ", j + 1, "...")
            rewards = np.zeros(n_nodes)

            seeds_set = set(seeds)

            # Iterate only over nodes not in seeds
            for i in set(range(n_nodes)) - seeds_set:
                # Inserting the test_seed function here
                reward = 0
                for _ in range(k):
                    history, active_nodes = self.simulate_episode(
                        [i] + seeds, prob_matrix=prob_matrix, max_steps=max_steps
                    )
                    reward += np.sum(active_nodes)
                rewards[i] = reward / k
            chosen_seed = np.argmax(rewards)
            seeds.append(chosen_seed)
            """
            print("Seed ", j + 1, " chosen: ", chosen_seed)
            print("Reward: ", rewards[chosen_seed])
            print("-------------------")
            """

        return seeds

    def opt(self, n_seeds, n_exp=1000, exp_per_seed=100, max_steps=100):
        if self.opt_value is None:
            opt_seeds = self.opt_arm(n_seeds, exp_per_seed, max_steps)
            experiment_rewards = np.zeros(n_exp)

            for i in range(n_exp):
                _, experiment_rewards[i] = self.round(opt_seeds)

            self.opt_value = (
                np.mean(experiment_rewards),
                np.std(experiment_rewards),
                opt_seeds,
            )
        return self.opt_value
