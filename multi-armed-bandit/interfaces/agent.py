from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self, k_arm=10, initial_estimate=0.):
        self.k_arm = k_arm
        # Keep both names for compatibility with subclasses
        self.initial = initial_estimate
        self.initial_estimate = initial_estimate
        self.reset()

    def reset(self):
        # Estimation for each action (default implementation)
        self.q_estimation = np.zeros(self.k_arm) + self.initial

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self, action, reward):
        pass