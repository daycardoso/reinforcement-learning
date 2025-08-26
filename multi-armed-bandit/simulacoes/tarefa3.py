# Em um multi-armed bandit com 10 alavancas e recompensa base de 0, gerar um
# gráfico comparando as políticas a seguir (use alpha=0.1 para os agentes que 
# atualizam estimativas usando tamanho de passo constante). Cada política deve
# ser executada 2000 vezes com 1000 tentativas em cada vez:
# epsilon-greedy atualizado com média amostral com epsilon=0.01
# epsilon-greedy atualizado com tamanho de passo constante com epsilon=0.01
# gradiente

import os
import matplotlib.pyplot as plt
from environment import Environment
from agentes.epsilon_greedy import EpsilonGreedySampleAverageAgent, EpsilonGreedyConstantStepsizeAgent
from agentes.gradient import GradientAgent
from simulacoes.simulacao_mab import simulate

def main():
    nome_simulacao="tarefa_3"
    runs = 2000
    time = 1000
    k_arm = 10
    base_reward = 0.0

    env = Environment(k_arm=k_arm, base_reward=base_reward)

    agents = [
        EpsilonGreedySampleAverageAgent(k_arm=k_arm, epsilon=0.01, initial_estimate=0),
        EpsilonGreedyConstantStepsizeAgent(k_arm=k_arm, epsilon=0.01, initial_estimate=0, step_size=0.1),
        GradientAgent(k_arm=k_arm, initial_estimate=0, alpha=0.1)
    ]
    labels = [
        'Epsilon-Greedy Média Amostral (ε=0.01)',
        'Epsilon-Greedy Passo Constante (ε=0.01, α=0.1)',
        'Gradiente (α=0.1)'
    ]

    mean_best_action_counts, mean_rewards = simulate(runs, time, agents, env)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(mean_rewards[i], label=label)
    plt.xlabel('Passos')
    plt.ylabel('Recompensa Média')
    plt.title('Comparação de Políticas - Multi-Armed Bandit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{nome_simulacao}_mean_reward.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(mean_best_action_counts[i], label=label)
    plt.xlabel('Passos')
    plt.ylabel('Frequência da Melhor Ação (%)')
    plt.title('Frequência da Melhor Ação - Multi-Armed Bandit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{nome_simulacao}_best_action_frequency.png')
    plt.show()

if __name__ == '__main__':
    main()

