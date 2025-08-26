from interfaces.agent import Agent
import numpy as np
class EpsilonGreedySampleAverageAgent(Agent):
    def __init__(self, k_arm=10, epsilon=0.1, initial_estimate=0):
        super().__init__(k_arm, initial_estimate)
        self.epsilon = epsilon
        self.k_arm = k_arm
        self.q_valor = np.zeros(self.k_arm) #ESTIMATIVA DO VALOR DA AÇÃO
        self.n_tentativas = np.zeros(self.k_arm) #NUMERO DE VEZES QUE A AÇÃO FOI TENTADA

    def act(self):
        # Escolha da ação sengundo a politica epsilon-gulosa
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.k_arm)
        else:
            action = np.argmax(self.q_valor)
        return action

    def reset(self):
        self.q_valor = np.full(self.k_arm, self.initial_estimate)
        self.n_tentativas = np.zeros(self.k_arm)

    def update(self, action, reward):
        # Atualização incremental do valor da ação escolhida segundo a media amostral
        self.n_tentativas[action] += 1

        # Atualiza a estimativa de recompensa esperada q(a) para a ação selecionada
        # Q[a] ← Q[a] + (1 / N[a]) * (reward - Q[a])
        # step_size = (1 / N[a])
        step_size = (1 / self.n_tentativas[action])

        # Q[a] ← Q[a] + step_size * (reward - Q[a])
        self.q_valor[action] += step_size * (reward - self.q_valor[action])

class EpsilonGreedyConstantStepsizeAgent(Agent):
    def __init__(self, k_arm=10, epsilon=0.1, initial_estimate=0., step_size=0.1):
        super().__init__(k_arm, initial_estimate)
        self.epsilon = epsilon
        self.k_arm = k_arm
        self.step_size = step_size # passo constante
        self.q_valor = np.zeros(self.k_arm) #ESTIMATIVA DO VALOR DA AÇÃO

        def act(self):
          # Escolha da ação sengundo a politica epsilon-gulosa
          if np.random.random() < self.epsilon:
              action = np.random.randint(self.k_arm)
          else:
              action = np.argmax(self.q_valor)
          return action

    def update(self, action, reward):
        # Atualização do valor da ação escolhida segundo a media amostral
        # Q[a] ← Q[a] + step_size * (reward - Q[a])
        self.q_valor[action] += self.step_size * (reward - self.q_valor[action])

