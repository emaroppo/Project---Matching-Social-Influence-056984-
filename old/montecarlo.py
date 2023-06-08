import numpy as np
from tqdm import tqdm
from hungarian_algorithm import hungarian_algorithm


n_nodes=30
edge_rate=0.1
graph_structure=np.random.binomial(1, 0.1, (30,30))
graph_probabilities=np.random.uniform(0.1, 0.2, (30, 30))*graph_structure


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
        print('Seed ', j+1, ' chosen: ', seeds[-1])
        print('Reward: ', rewards[seeds[-1]])
        print('-------------------')

    return seeds
    


def get_reward(node_class, product_class):
    return 100-np.random.normal(means[node_class-1, product_class], stds[node_class-1, product_class])

import numpy as np
from scipy.optimize import linear_sum_assignment

# con la versione dell'assistente si impantanava, fatto questo con chatgpt per ora, cerco di riparare l'altra
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


def exp_optimum(graph_probabilities, opt_seeds, n_exp):

    rewards_per_exp=np.zeros(n_exp)

    for i in tqdm(range(n_exp)):
        active_nodes=simulate_episode(graph_probabilities, opt_seeds, 100)[1]*node_classes
        #remove all zeros from array (nodes that were not activated)
        active_nodes=active_nodes[active_nodes!=0]
        #print(active_nodes)
        #compute rewards
        hun_matrix_dim=max(len(active_nodes), 9)
        rewards=np.zeros((hun_matrix_dim, hun_matrix_dim))

        for j in range(hun_matrix_dim):
            for l in range(hun_matrix_dim):
                if j<len(active_nodes) and l<9:
                    rewards[j,l]=get_reward(int(active_nodes[j]), l//3)
                else:
                    rewards[j,l]=0

        optimum= hungarian_algorithm(rewards)[0]
        optimum= 100-optimum #convert to reward
        #set all 100s to 0s
        optimum[optimum==100]=0
        #print(optimum)
        rewards_per_exp[i]=np.sum(optimum)
        #print('Optimum reward: ', np.sum(optimum))
        #print('-------------------')
    return np.mean(rewards_per_exp), np.std(rewards_per_exp)


node_classes=np.random.randint(1,4, graph_probabilities.shape[0])
#parameters for gaussian distribution
means=np.random.uniform(10, 20, (3,3))
stds=np.random.uniform(1, 3, (3,3))

opt_seeds=greedy_algorithm(graph_probabilities, 3, 1000, 100)
print(exp_optimum(graph_probabilities, opt_seeds, 1000))