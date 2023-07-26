from environments.environment import Environment
from environments.social_environment import SocialEnvironment
from environments.matching_environment import MatchingEnvironment
import numpy as np

class JointEnvironment(Environment):
    def __init__(self, probabilities, reward_parameters, node_classes):
        self.social_environment = SocialEnvironment(probabilities)
        self.matching_environment = MatchingEnvironment(reward_parameters)
        self.node_classes = node_classes
        self.t = 0

    def round(self, pulled_arms):
        social_reward = self.social_environment.round(pulled_arms)
        matching_reward = self.matching_environment.round(pulled_arms)
        return matching_reward
    
    def opt(self, n_seeds=3, prod_classes=[0,1,2]*3):
        opt_seeds = self.social_environment.opt(n_seeds)[2]
        print("opt seeds: ", opt_seeds, type(opt_seeds))

        active_nodes=self.social_environment.round(opt_seeds, joint=True)
        active_nodes=np.where(active_nodes==1)[0]
        print("active nodes: ", active_nodes)
        customer_classes=[self.node_classes[i] for i in active_nodes]
        matching_reward=self.matching_environment.opt(customer_classes, prod_classes)

        print("opt reward: ", matching_reward)

        return matching_reward