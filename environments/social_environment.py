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

        # set up seeds
        active_nodes = np.zeros(n_nodes)

        for seed in seeds:
            active_nodes[seed] = 1

        history = np.array([active_nodes])

        newly_active_nodes = active_nodes

        t = 0

        while t < max_steps and np.sum(newly_active_nodes) > 0:
            # retrieve probability of edge activations
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            # remove activated edges
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)
            # update active nodes
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (
                1 - active_nodes
            )
            # print(newly_active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)
            # print(active_nodes)
            history = np.concatenate((history, [newly_active_nodes]), axis=0)
            t += 1
        return history, active_nodes

    def round(self, pulled_arms, joint=False):
        seeds = pulled_arms
        history, active_nodes = self.simulate_episode(seeds)

        if joint:
            return active_nodes

        reward = np.sum(active_nodes)
        return reward

    def opt_arm(self, budget, k=100, max_steps=100):
        prob_matrix = self.probabilities.copy()
        n_nodes = prob_matrix.shape[0]

        seeds = []
        for j in range(budget):
            print("Choosing seed ", j + 1, "...")
            rewards = np.zeros(n_nodes)

            for i in tqdm(range(n_nodes)):
                if i not in seeds:
                    # Inserting the test_seed function here
                    reward = 0
                    for _ in range(k):
                        history, active_nodes = self.simulate_episode(
                            [i] + seeds, prob_matrix=prob_matrix, max_steps=max_steps
                        )
                        reward += np.sum(active_nodes)
                    rewards[i] = reward / k
            seeds.append(np.argmax(rewards))
            print("Seed ", j + 1, " chosen: ", seeds[-1])
            print("Reward: ", rewards[seeds[-1]])
            print("-------------------")

        return seeds

    def opt(self, n_seeds, n_exp=1000, exp_per_seed=100, max_steps=100):
        if self.opt_value is None:
            opt_seeds = self.opt_arm(n_seeds, exp_per_seed, max_steps)
            experiment_rewards = np.zeros(n_exp)

            for i in tqdm(range(n_exp)):
                experiment_rewards[i] = self.round(opt_seeds)

            self.opt_value = (
                np.mean(experiment_rewards),
                np.std(experiment_rewards),
                opt_seeds,
            )
        return self.opt_value
