from interfaces.agent import Agent
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

class GradientAgent(Agent):
    def __init__(self, k_arm=10, initial_estimate=0., alpha=0.01):
        super().__init__(k_arm, initial_estimate)
        self.alpha = alpha
        self.preferencias = np.zeros(k_arm)
        self.n_tentativas = np.zeros(k_arm)
        self.media_recompensas = np.zeros(k_arm)

    def act(self):
        probabilidades = softmax(self.preferencias)
        acao = np.random.choice(self.k_arm, p=probabilidades)
        return acao
    
    def update(self, action, reward):
        
        for a in range(self.k_arm):            
            if a == action:
                self.preferencias[a] += self.alpha * (reward - self.media_recompensas[a]) * (1 - softmax(self.preferencias)[a])
            else:
                self.preferencias[a] -= self.alpha * (reward - self.media_recompensas[a]) * softmax(self.preferencias)[a]
                
            self.n_tentativas[a] += 1
            self.media_recompensas[a] += (reward - self.media_recompensas[a]) / self.n_tentativas[a]
            
    def reset(self):
        super().reset()
        self.preferencias = np.zeros(self.k_arm)
        self.n_tentativas = np.zeros(self.k_arm)
        self.media_recompensas = np.zeros(self.k_arm)
        