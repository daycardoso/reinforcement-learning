from numpy import random, argmax
class Environment:
    def __init__(self, k_arm=10, base_reward=0., seed=None):
        self.k = k_arm
        self.base_reward = base_reward
        self.seed = seed
        self.reset()

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
        # Real reward for each action
        self.q_true = random.randn(self.k) + self.base_reward
        self.best_action = argmax(self.q_true)

    def get_reward(self, action):
        # Generate the reward under N(real reward, 1)
        return random.randn() + self.q_true[action]