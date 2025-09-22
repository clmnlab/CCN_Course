
import gymnasium as gym
import numpy as np


class Agent:
    def __init__(self, env, mode='q_learning', epsilon=0.1, alpha=0.1, gamma=0.99):
        self.env = env
        self.mode = mode
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # Q-테이블을 0으로 초기화 (상태 개수, 행동 개수)
        self.q_table = np.zeros((env.action_space.n, env.observation_space.shape[0]))
        
    def select_action(self, state):
        """Epsilon-Greedy 정책으로 행동 선택"""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # 탐험: 무작위 행동
        else:
            return np.argmax(self.q_table[state])  # 활용: 가장 가치가 높은 행동

    def update(self, state, action, reward, next_state, done):
        """Q-테이블 업데이트"""
        if self.mode == 'q_learning':
            # --- Q-러닝 업데이트 ---
            # 다음 상태에서 가장 좋은 행동의 가치를 가져옴
            best_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next_q * (1 - done)
        
        elif self.mode == 'sarsa':
            # --- SARSA 업데이트 ---
            # 다음 상태에서 실제로 할 행동의 가치를 가져옴
            next_action = self.select_action(next_state)
            next_q = self.q_table[next_state, next_action]
            td_target = reward + self.gamma * next_q * (1 - done)
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


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
        
        