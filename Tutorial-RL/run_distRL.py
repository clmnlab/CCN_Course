import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agents_distRL import QLearningAgent, DistributionalRLAgent

# ---------------- Training loop ---------------- #
def train(agent, env, episodes=100):
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards


# ---------------- Dopamine firing simulation ---------------- #
def simulate_dopamine_firing(quantiles, neurons_per_quantile=20, duration=500):
    """
    Simulate dopamine neuron firing for each quantile.
    Each quantile corresponds to one group of dopamine neurons.
    - quantiles: learned quantile values
    - neurons_per_quantile: number of neurons per group
    - duration: number of milliseconds to simulate
    """
    all_spikes = []
    for q in quantiles:
        # Convert quantile value into firing rate (Hz)
        rate = max(1.0, q + 5.0)  # ensure positive firing rate
        # Generate spikes as a Poisson process
        spikes = np.random.poisson(rate / 1000.0, (neurons_per_quantile, duration))
        all_spikes.append(spikes)
    return all_spikes


# ---------------- Main script ---------------- #
if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")

    # Create agents
    q_agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                             alpha=0.1, gamma=0.99, epsilon=0.1)
    dist_agent = DistributionalRLAgent(env.observation_space.n, env.action_space.n,
                                       num_quantiles=5, alpha=0.1, gamma=0.99, epsilon=0.1)

    # Train agents
    q_rewards = train(q_agent, env, episodes=100)
    dist_rewards = train(dist_agent, env, episodes=100)

    # ---------------- Plot reward curves ---------------- #
    plt.figure(figsize=(10, 5))
    plt.plot(q_rewards, label="Q-learning (expected value)")
    plt.plot(dist_rewards, label="Distributional RL (quantiles)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CliffWalking-v1: Standard RL vs Distributional RL")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------- Visualize quantiles at the start state ---------------- #
    nrow, ncol = env.unwrapped.shape
    start_state = (nrow - 1) * ncol   # bottom-left corner = start state
    best_action = np.argmax(np.mean(dist_agent.Z[start_state], axis=-1))
    quantiles = dist_agent.Z[start_state, best_action]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(quantiles)), quantiles, color="skyblue")
    plt.xticks(range(len(quantiles)),
               [f"τ={tau:.1f}" for tau in dist_agent.quantile_probs])
    plt.ylabel("Value")
    plt.title("Distributional RL: Quantile values at Start State")
    plt.show()

    # ---------------- Dopamine firing simulation ---------------- #
    spikes = simulate_dopamine_firing(quantiles,
                                      neurons_per_quantile=20,
                                      duration=500)

    fig, axes = plt.subplots(len(quantiles), 1, figsize=(8, 8), sharex=True)
    for i, q_spikes in enumerate(spikes):
        neuron_ids, times = np.where(q_spikes == 1)
        axes[i].scatter(times, neuron_ids, s=5)
        axes[i].set_ylabel(f"τ={dist_agent.quantile_probs[i]:.1f}")
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Simulated Dopamine Neuron Firing per Quantile")
    plt.tight_layout()
    plt.show()
