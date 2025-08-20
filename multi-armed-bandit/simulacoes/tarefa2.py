import os
import matplotlib.pyplot as plt
from environment import Environment
from agentes.epsilon_greedy import EpsilonGreedySampleAverageAgent
from simulacoes.simulacao_mab import simulate
# 1. Criar ambiente
env = Environment(k_arm=10, base_reward=0)

# 2. Criar agentes
agents = [
    EpsilonGreedySampleAverageAgent(k_arm=10, epsilon=0),
    EpsilonGreedySampleAverageAgent(k_arm=10, epsilon=0.1),
    EpsilonGreedySampleAverageAgent(k_arm=10, epsilon=0.01)
]

labels = [r'$\epsilon = 0$', r'$\epsilon = 0.1$', r'$\epsilon = 0.01$']

# 3. Rodar simulação
# Permite reduzir para testes rápidos definindo a variável de ambiente DEBUG_SMALL=1
debug_small = os.environ.get('DEBUG_SMALL') == '1'
if debug_small:
    runs, time_steps = 50, 200
else:
    runs, time_steps = 2000, 1000

mean_best_action_counts, mean_rewards = simulate(runs=runs, time=time_steps, agents=agents, environment=env)

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# 4a. Plotar % de melhor ação escolhida
plt.figure(figsize=(10,5))
for i, label in enumerate(labels):
    plt.plot(mean_best_action_counts[i], label=label)
plt.xlabel('Etapas')
plt.ylabel('% Melhor ação')
plt.legend()
plt.title('Frequência da melhor ação escolhida')
best_action_path = os.path.join(output_dir, 'best_action_frequency.png')
plt.savefig(best_action_path, bbox_inches='tight')
if os.environ.get('SHOW_FIG') == '1':
    plt.show()
plt.close()

# 4b. Plotar recompensa média
plt.figure(figsize=(10,5))
for i, label in enumerate(labels):
    plt.plot(mean_rewards[i], label=label)
plt.xlabel('Etapas')
plt.ylabel('Recompensa média')
plt.legend()
plt.title('Recompensa média ao longo do tempo')
reward_path = os.path.join(output_dir, 'mean_reward.png')
plt.savefig(reward_path, bbox_inches='tight')
if os.environ.get('SHOW_FIG') == '1':
    plt.show()
plt.close()

print(f"Gráficos salvos em: {output_dir}")
