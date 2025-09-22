import numpy as np
import random

class BaseAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        raise NotImplementedError


class QLearningAgent(BaseAgent):
    """Standard Q-learning agent (expected value only)."""
    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next * (1 - int(done))
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])


class DistributionalRLAgent(BaseAgent):
    """Simplified Distributional RL agent using quantile approximation."""
    def __init__(self, n_states, n_actions, num_quantiles=5, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.num_quantiles = num_quantiles
        self.quantile_probs = np.linspace(0, 1, num_quantiles+2)[1:-1]
        # Z[s,a,i] = quantile i value for (s,a)
        self.Z = np.zeros((n_states, n_actions, num_quantiles))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # choose action by mean of quantiles
        return np.argmax(np.mean(self.Z[state], axis=-1))

    def update(self, state, action, reward, next_state, done):
        best_next = np.argmax(np.mean(self.Z[next_state], axis=-1))
        target_samples = reward + self.gamma * self.Z[next_state, best_next] * (1 - int(done))

        # quantile TD update
        for i, tau in enumerate(self.quantile_probs):
            td_error = target_samples.mean() - self.Z[state, action, i]
            self.Z[state, action, i] += self.alpha * td_error
