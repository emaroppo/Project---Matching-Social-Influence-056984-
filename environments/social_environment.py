from environments.environment import Environment
import numpy as np

    
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
    
    def round(self, pulled_arms, joint=False):
        seeds= pulled_arms
        history, active_nodes=self.simulate_episode(seeds)
        
        if joint:
            return active_nodes
        
        reward = np.sum(active_nodes)
        return reward
