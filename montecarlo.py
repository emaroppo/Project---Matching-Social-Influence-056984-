import numpy as np


n_nodes=30
edge_rate=0.1
graph_structure=np.random.binomial(1, 0.1, (30,30))
graph_probabilities=np.random.uniform(0.001, 0.07, (30, 30))


def simulate_episode(init_prob_matrix, seeds:list, max_steps):
    prob_matrix=init_prob_matrix.copy()
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

def test_seed(seed, active_nodes, prob_matrix, k, max_steps):
    reward = 0
    for i in range(k):
        history, active_nodes=simulate_episode(prob_matrix, seed, max_steps)
        reward+=np.sum(active_nodes)
    return reward/k

def greedy_algorithm(init_prob_matrix, budget, k, max_steps):
    prob_matrix=init_prob_matrix.copy()
    n_nodes=prob_matrix.shape[0]
    
    seeds=[]
    for j in range(budget):
        rewards=np.zeros(n_nodes)
        for i in range(n_nodes):
            if i not in seeds:
                rewards[i]=test_seed([i], np.zeros(n_nodes), prob_matrix, k, max_steps)
        seeds.append(np.argmax(rewards))
    return seeds
    


print(greedy_algorithm(graph_probabilities, 5, 100, 1000))




    

