from interfaces.agent import Agent
import numpy as np
class EpsilonGreedySampleAverageAgent(Agent):
    def __init__(self, k_arm=10, epsilon=0.1, initial_estimate=0.):
        super().__init__(k_arm, initial_estimate)
        self.epsilon = epsilon
        self.valor_estimado = np.zeros(self.k_arm) #ESTIMATIVA DO VALOR DA AÇÃO
        # iniciar com destibuição normal
        # self.valor_estimado = np.random.normal(0, 1, self.k_arm)
        self.n_tentativas = np.zeros(self.k_arm) #NUMERO DE VEZES QUE A AÇÃO FOI TENTADA
        self.reset()
        
    def act(self):
        # Escolha da ação sengundo a politica epsilon-gulosa
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.k_arm)
        else:
            action = np.argmax(self.valor_estimado)
        return action

    def reset(self):
        self.valor_estimado = np.full(self.k_arm, self.initial)
        

    def update(self, action, reward):
        # Atualização incremental do valor da ação escolhida segundo a media amostral
        self.n_tentativas[action] += 1

        # Atualiza a estimativa de recompensa esperada q(a) para a ação selecionada
        # Q[a] ← Q[a] + (1 / N[a]) * (reward - Q[a])
        # step_size = (1 / N[a])
        step_size = (1 / self.n_tentativas[action])

        # Q[a] ← Q[a] + step_size * (reward - Q[a])
        self.valor_estimado[action] += step_size * (reward - self.valor_estimado[action])

class EpsilonGreedyConstantStepsizeAgent(Agent):
    def __init__(self, k_arm=10, epsilon=0.1, initial_estimate=0., step_size=0.1):
        super().__init__(k_arm, initial_estimate)
        self.epsilon = epsilon
        self.step_size = step_size # passo constante
        self.valor_estimado = np.zeros(self.k_arm) #ESTIMATIVA DO VALOR DA AÇÃO
        self.reset()
        
    def act(self):
          # Escolha da ação sengundo a politica epsilon-gulosa
          if np.random.random() < self.epsilon:
              action = np.random.randint(self.k_arm)
          else:
              action = np.argmax(self.valor_estimado)
          return action

    def update(self, action, reward):
        # Atualização do valor da ação escolhida segundo a media amostral
        # Q[a] ← Q[a] + step_size * (reward - Q[a])
        self.valor_estimado[action] += self.step_size * (reward - self.valor_estimado[action])

    def reset(self):
        return super().reset()
    