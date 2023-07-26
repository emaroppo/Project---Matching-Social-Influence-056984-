import numpy as np

class Environment:
    def __init__(self, probabilities, opt_value=None):
        self.probabilities = probabilities
        self.opt_value = opt_value

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])       
        return reward
    
    def opt(self):
        if self.opt_value is None:
            self.opt_value=np.max(self.probabilities)
        return self.opt_value




