# This script contains several implementations of reinforcement learning agents,
# specifically for Q-learning and SARSA algorithms.
# These classes are designed to be used in simple discrete state/action environments.

import gymnasium as gym
import numpy as np


class Agent:
    """
    A generic agent class that can be configured to use either Q-learning or SARSA.
    It includes methods for action selection and Q-table updates.
    """
    def __init__(self, env, mode='q_learning', epsilon=0.1, alpha=0.1, gamma=0.99):
        """
        Initializes the agent with hyperparameters and the environment.

        Args:
            env (gym.Env): The environment the agent will learn in.
            mode (str): The learning algorithm to use ('q_learning' or 'sarsa').
            epsilon (float): The exploration rate for the epsilon-greedy policy.
            alpha (float): The learning rate.
            gamma (float): The discount factor for future rewards.
        """
        self.env = env
        self.mode = mode
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # Initialize the Q-table with zeros. The shape should be (number of states, number of actions).
        # Note: The original code had the shape (num_actions, num_states), which is a common error.
        # This implementation assumes the standard (num_states, num_actions) format.
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        
    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploration: Choose a random action.
        else:
            return np.argmax(self.q_table[state])  # Exploitation: Choose the action with the highest Q-value.

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table based on the chosen learning algorithm.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state.
            done (bool): A flag indicating if the episode has terminated.
        """
        if self.mode == 'q_learning':
            # Q-learning update (Off-policy):
            # It updates based on the best possible future reward from the next state,
            # regardless of the action that would be taken by the current policy.
            best_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next_q * (1 - done)
        
        elif self.mode == 'sarsa':
            # SARSA update (On-policy):
            # It updates based on the value of the next action that would be selected
            # by the current policy (which is `self.select_action(next_state)`).
            next_action = self.select_action(next_state)
            next_q = self.q_table[next_state, next_action]
            td_target = reward + self.gamma * next_q * (1 - done)
        
        # Calculate the Temporal Difference (TD) error and update the Q-value.
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


class QLearningAgent:
    """
    A class representing a Q-learning agent with a softmax-like action selection policy.
    This provides a different way to handle the exploration-exploitation trade-off.
    """
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.99, exploration_rate=1):
        """
        Initializes the Q-learning agent.

        Args:
            n_states (int): The number of states in the environment.
            n_actions (int): The number of actions available to the agent.
            lr (float): The learning rate.
            gamma (float): The discount factor for future rewards.
            exploration_rate (float): A value that controls the degree of exploration (higher = more random).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.q_values = np.zeros((n_states, n_actions))

    def choose(self, state):
        """
        Selects an action for the given state using a softmax-like policy.
        The agent chooses actions with probabilities proportional to their Q-values.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The selected action.
        """
        # Compute the probabilities for each action using the softmax function.
        # A higher exploration_rate makes the probabilities more uniform, leading to more exploration.
        probs = np.exp(self.q_values[state] / self.exploration_rate)
        probs /= np.sum(probs)
        
        # Sample an action from the probability distribution.
        action = np.random.choice(self.n_actions, p=probs)
        return action
    
    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value for the current state and action using the Q-learning rule.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state of the environment.
        """
        # Calculate the temporal difference error.
        if next_state is None:
            # Case for a terminal state (no next state).
            td_error = reward - self.q_values[state, action]
        else:
            # Q-learning TD error: R + gamma * max(Q(s', a')) - Q(s, a)
            td_error = reward + self.gamma * np.max(self.q_values[next_state]) - \
                       self.q_values[state, action]
        
        # Update the Q-value.
        self.q_values[state, action] += self.lr * td_error


class SARSA:
    """
    A class representing a SARSA agent with an epsilon-greedy policy.
    This agent's policy and update rule are tightly coupled (on-policy).
    """
    def __init__(self, n_states, n_actions, lr=0.1, gamma=1, epsilon_decay = 0.99, epsilon_min=0.01):
        """
        Initializes the SARSA agent.

        Args:
            n_states (int): The number of states in the environment.
            n_actions (int): The number of actions available to the agent.
            lr (float): The learning rate.
            gamma (float): The discount factor for future rewards.
            epsilon_decay (float): The decay rate of epsilon.
            epsilon_min (float): The minimum value epsilon can decay to.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0 # Initial epsilon value.
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_values = np.zeros((n_states, n_actions))
        
    def choose(self, state):
        """
        Selects an action for the given state using an epsilon-greedy policy.
        Epsilon decays over time to shift from exploration to exploitation.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action.
            action = np.random.randint(self.n_actions)
        else:
            # Exploitation: Choose the best action.
            action = np.argmax(self.q_values[state])
            
        # Decay epsilon after each action selection.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) 
        return action
    
    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value for the current state and action using the SARSA rule.
        This rule is on-policy because it depends on the next action chosen by the policy.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state of the environment.
        """
        # Calculate the temporal difference error.
        if next_state is None:
            # Case for a terminal state.
            td_error = reward - self.q_values[state, action]
        else:
            # SARSA TD error: R + gamma * Q(s', a') - Q(s, a)
            # The next action (`next_action`) is chosen by the same `choose` method.
            next_action = self.choose(next_state)
            td_error = reward + self.gamma * self.q_values[next_state, next_action] - \
                       self.q_values[state, action]
        
        # Update the Q-value.
        self.q_values[state, action] += self.lr * td_error
