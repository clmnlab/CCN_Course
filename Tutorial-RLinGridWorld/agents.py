class QLearningAgent:
    """
    A class representing an agent performing Q-learning with an epsilon-greedy policy on the 
    HierarchicalBanditTask environment.
    """
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.99, exploration_rate=1):
        """
        Initialize the Q-learning agent.

        Args:
            n_actions (int): The number of actions available to the agent.
            n_states (int): The number of states in the environment.
            learning_rate (float): The learning rate for Q-learning updates (default: 0.1).
            discount_factor (float): The discount factor for future rewards (default: 0.99).
            exploration_rate (float): The exploration rate for the epsilon-greedy policy (default: 0.1).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
#        self.epsilon = epsilon
        self.exploration_rate = exploration_rate
        self.q_values = np.zeros((n_states, n_actions))

    
    
    def choose(self, state):
        """
        Select an action for the given state using an epsilon-greedy policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The selected action.
        """
        # Compute the probabilities for each action using the softmax function
        probs = np.exp(self.q_values[state] / self.exploration_rate)
        probs /= np.sum(probs)
        
        # Sample an action from the probability distribution
        action = np.random.choice(self.n_actions, p=probs)
#         if np.random.rand() < self.epsilon:
#             # Choose a random action
#             action = np.random.randint(self.n_actions)
#         else:
#             # Choose the action with the highest Q-value
#             action = np.argmax(self.q_values[state])
        return action
    
    
    def update(self, state, action, reward, next_state):
        """
        Update the Q-value for the current state and action.

        Args:
            next_state (int): The next state of the environment.
            reward (float): The reward received for the current action.

        Returns:
            None
        """
        # Calculate the temporal difference error
        if next_state is None:
            td_error = reward - self.q_values[state, action]
        else:
            td_error = reward + self.gamma * np.max(self.q_values[next_state]) - \
                       self.q_values[state, action]
        
        # Update the Q-value for the current state and action
        self.q_values[state, action] += self.lr * td_error

import numpy as np

class SARSA:
    """
    A class representing an agent performing SARSA with an epsilon-greedy policy on the 
    HierarchicalBanditTask environment.
    """
    def __init__(self, n_states, n_actions, lr=0.1, gamma=1, epsilon_decay = 0.99, epsilon_min=0.01):
        """
        Initialize the Q-learning agent.

        Args:
            n_states (int): The number of states in the environment.
            n_actions (int): The number of actions available to the agent.
            lr (float): The learning rate for Q-learning updates (default: 0.1).
            gamma (float): The discount factor for future rewards (default: 0.99).
            epsilon_decay (float): The decay rate of epsilon value in the epsilon-greedy policy (default: 0.99).
            epsilon_min (float): minumum value of epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_values = np.zeros((n_states, n_actions))
        
    
    
    def choose(self, state):
        """
        Select an action for the given state using an epsilon-greedy policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The selected action.
        """
        
        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = np.random.randint(self.n_actions)
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(self.q_values[state])
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min) 
        return action
    
    
    def update(self, state, action, reward, next_state):
        """
        Update the Q-value for the current state and action.

        Args:
            next_state (int): The next state of the environment.
            reward (float): The reward received for the current action.

        Returns:
            None
        """

        # Calculate the temporal difference error
        if next_state is None:
            td_error = reward - self.q_values[state, action]
        else:
            # sample the next action given next state
            next_action = self.choose(next_state)
            td_error = reward + self.gamma * self.q_values[next_state, next_action] - \
                       self.q_values[state, action]
        
        # Update the Q-value for the current state and action
        self.q_values[state, action] += self.lr * td_error
        
        