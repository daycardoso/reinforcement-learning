from interfaces.agent import Agent
import numpy as np
class EpsilonGreedySampleAverageAgent(Agent):
    def __init__(self, k_arm=10, epsilon=0.1, initial_estimate=0):
        super().__init__(k_arm, initial_estimate)
        self.epsilon = epsilon
        self.k_arm = k_arm
        self.initial_estimate = initial_estimate
        self.q_estimation = np.full(k_arm, initial_estimate)
        self.action_count = np.zeros(k_arm)

    def act(self):
        # Sortear valor randomico de 0 a 1
        if np.random.rand() <= self.epsilon:
            # Exploração: escolhe aleatoriamente
            return np.random.randint(self.k_arm)
        else:
            # Exploitação: escolhe a melhor estimativa até agora
            return np.argmax(self.q_estimation)

    def reset(self):
        self.q_estimation = np.full(self.k_arm, self.initial_estimate)
        self.action_count = np.zeros(self.k_arm)

    def update(self, action, reward):
        
        # Incrementa o contador da açao que foi selecionada
        self.action_count[action] += 1
        
        # Atualiza a estimativa de recompensa esperada q(a) para a ação selecionada
        # Q[a] ← Q[a] + (1 / N[a]) * (reward - Q[a])
        # step_size = (1 / N[a])
        step_size = (1 / self.action_count[action])
        
        # Q[a] ← Q[a] + step_size * (reward - Q[a])
        
        self.q_estimation[action] += step_size * (reward - self.q_estimation[action])

        