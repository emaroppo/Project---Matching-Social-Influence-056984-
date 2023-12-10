from environments.matching_environment import MatchingEnvironment, generate_reward


class MatchingEnvironmentContext(MatchingEnvironment):
    def __init__(self, reward_parameters, class_mapping):
        super().__init__(reward_parameters)
        self.class_mapping = class_mapping

    def round(self, pulled_arms):
        # replace features with class label
        pulled_arms = [
            (self.class_mapping(pulled_arm[0]), pulled_arm[1])
            for pulled_arm in pulled_arms
        ]

        return super().round(pulled_arms)

    def opt(self, customer_features, product_classes):
        customer_classes = [
            self.class_mapping(customer_feature)
            for customer_feature in customer_features
        ]
        return super().opt(customer_classes, product_classes)

    def expected_reward(self, pulled_arm):
        # replace features with class label
        pulled_arm = [
            (self.class_mapping(pulled_arm[0]), pulled_arm[1])
            for pulled_arm in pulled_arm
        ]
        return super().expected_reward(pulled_arm)
