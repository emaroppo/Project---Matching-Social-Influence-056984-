from environments.environment import Environment

class JointEnvironment(Environment):
    def __init__(self, social_environment, matching_environment):
        self.social_environment = social_environment
        self.matching_environment = matching_environment
        self.t = 0

    def round(self, pulled_arms):
        social_reward = self.social_environment.round(pulled_arms)
        matching_reward = self.matching_environment.round(pulled_arms)
        return matching_reward