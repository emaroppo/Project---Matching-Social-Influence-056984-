import numpy as np

class Environment:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])       
        return reward

    
class SocialEnvironment(Environment):
    def __init__(self, probabilities):
        super().__init__(probabilities)
    
    def simulate_episode(self, seeds:list, max_steps=100):
        prob_matrix=self.probabilities.copy()
        n_nodes=prob_matrix.shape[0]
        
        #set up seeds
        active_nodes=np.zeros(n_nodes)
        
        for seed in seeds:
            active_nodes[seed]=1
        
        history=np.array([active_nodes])

        newly_active_nodes=active_nodes

        t=0
        
        while (t<max_steps and np.sum(newly_active_nodes)>0):
            #retrieve probability of edge activations
            p = (prob_matrix.T*active_nodes).T
            activated_edges=p>np.random.rand(p.shape[0], p.shape[1])
            #remove activated edges
            prob_matrix=prob_matrix*((p!=0)==activated_edges)
            #update active nodes
            newly_active_nodes=(np.sum(activated_edges, axis=0)>0)*(1-active_nodes)
            #print(newly_active_nodes)
            active_nodes=np.array(active_nodes+newly_active_nodes)
            #print(active_nodes)
            history=np.concatenate((history, [newly_active_nodes]), axis=0)
            t+=1
        return history, active_nodes
    
    def round(self, pulled_arms):
        seeds= pulled_arms
        history, active_nodes=self.simulate_episode(seeds)
        reward = np.sum(active_nodes)
        return reward

class MatchingEnvironment(Environment):
    def __init__(self, reward_matrix):
        self.reward_matrix = reward_matrix
        # add a row and a column of zeros to the reward matrix to represent the case in which no match is made
        self.reward_matrix = np.hstack((self.reward_matrix, np.zeros((self.reward_matrix.shape[0], 1))))
        self.reward_matrix = np.vstack((self.reward_matrix, np.zeros((1, self.reward_matrix.shape[1]))))

        self.n_arms = reward_matrix.size
        self.t = 0

    def round(self, pulled_arms):
        rewards = [self.reward_matrix[pulled_arm] for pulled_arm in pulled_arms]
        
        #iterate through all cells of rewards, if a cell is callable, call it and replace it with the result
        for i in range(len(rewards)):
            if callable(rewards[i]):
                rewards[i] = rewards[i]()
        
        return np.array(rewards)



