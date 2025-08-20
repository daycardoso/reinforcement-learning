import numpy as np
from tqdm import trange # Importar tqdm para mostrar a barra de progresso

def simulate(runs, time, agents, environment):
    rewards = np.zeros((len(agents), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, agent in enumerate(agents):
        for r in trange(runs):
            environment.reset()
            agent.reset()
            for t in range(time):
                action = agent.act()
                reward = environment.get_reward(action)
                agent.update(action, reward)
                rewards[i, r, t] = reward
                if action == environment.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards