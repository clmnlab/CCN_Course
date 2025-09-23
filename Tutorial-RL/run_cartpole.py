# This script implements and compares four different Reinforcement Learning (RL) agents
# for solving the classic CartPole-v1 environment.
# The agents are: Q-table, Tabular Policy Gradient, Deep Policy Gradient, and DQN.
# The code includes a training loop, real-time plotting of rewards, and a user interface
# to choose the agent and rendering options.

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ---------------- Hyperparameters ---------------- #
# These parameters control the behavior of the agents and the training process.
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 200
GAMMA = 0.99  # Discount factor for future rewards

# Q-table / Tabular Policy Gradient
ALPHA = 0.1  # Learning rate for tabular methods
EPSILON = 0.1  # Epsilon-greedy exploration rate for Q-table
DISCRETE_BINS = 10  # Number of bins to discretize the continuous state space

# Deep Policy Gradient / DQN
LR = 1e-3  # Learning rate for neural network optimizers
BATCH_SIZE = 64  # Number of samples to train on in each DQN step
MEMORY_SIZE = 10000  # Size of the replay buffer for DQN
EPSILON_START = 1.0  # Initial epsilon value for DQN's epsilon-greedy exploration
EPSILON_END = 0.01  # Minimum epsilon value
EPSILON_DECAY = 0.995  # Decay rate for epsilon
TARGET_UPDATE = 10  # Number of episodes after which to update the target Q-network
# ------------------------------------------------ #

# ---------------- Environment Setup ---------------- #
# Initialize the CartPole-v1 environment.
env = gym.make(ENV_NAME, render_mode='human')
state_dim = env.observation_space.shape[0]  # The number of state variables (4 for CartPole)
action_dim = env.action_space.n  # The number of possible actions (2 for CartPole: left/right)

# ---------------- Helper: State Discretization ---------------- #
# This function is used by the Q-table and Tabular Policy Gradient agents
# to convert the continuous state space of CartPole into a discrete one.
def discretize_state(state, bins=DISCRETE_BINS):
    """
    Discretizes a continuous state into a tuple of integers.
    This is necessary for agents that use a discrete state representation (e.g., Q-table).
    The continuous values are mapped to a fixed number of bins.
    """
    upper_bounds = env.observation_space.high
    lower_bounds = env.observation_space.low
    # Clip inf values to avoid overflow and make the state space bounded.
    upper_bounds[1] = min(upper_bounds[1], 5.0)  # Clip cart velocity
    upper_bounds[3] = min(upper_bounds[3], 5.0)  # Clip pole angular velocity
    lower_bounds[1] = max(lower_bounds[1], -5.0)
    lower_bounds[3] = max(lower_bounds[3], -5.0)

    # Calculate the ratio of the state value within its range, then scale to the number of bins.
    ratios = (state - lower_bounds) / (upper_bounds - lower_bounds)
    ratios = np.clip(ratios, 0, 0.999) # Clip to avoid index out of bounds
    discrete = (ratios * bins).astype(int)
    return tuple(discrete)

# ---------------- Q-table Agent ---------------- #
# Implements a Q-learning agent using a tabular approach.
class QTableAgent:
    def __init__(self):
        # Initialize the Q-table with zeros. The dimensions are based on the discretized
        # state space and the number of actions.
        self.q_table = np.zeros((DISCRETE_BINS,) * state_dim + (action_dim,))

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        With a probability of epsilon, a random action is chosen for exploration.
        Otherwise, the action with the highest Q-value is chosen for exploitation.
        """
        if np.random.rand() < EPSILON:
            return env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Q-learning update rule.
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
        """
        best_next = np.max(self.q_table[next_state])
        td_target = reward + GAMMA * best_next * (1 - done)
        self.q_table[state][action] += ALPHA * (td_target - self.q_table[state][action])

# ---------------- Tabular Policy Gradient Agent ---------------- #
# Implements a Policy Gradient agent with a tabular policy.
# This agent learns a probability distribution over actions for each state.
class TabularPolicyAgent:
    def __init__(self):
        # Initialize a policy table with uniform probabilities.
        self.policy_table = np.ones((DISCRETE_BINS,) * state_dim + (action_dim,)) / action_dim
        # Lists to store episode data for updating the policy.
        self.rewards = []
        self.actions = []
        self.states = []

    def select_action(self, state):
        """
        Selects an action based on the probability distribution in the policy table.
        The state and action are saved for later updates.
        """
        probs = self.policy_table[state]
        action = np.random.choice(action_dim, p=probs)
        self.states.append(state)
        self.actions.append(action)
        return action

    def finish_episode(self):
        """
        Updates the policy table at the end of an episode using the policy gradient algorithm.
        Rewards are used to calculate the discounted return (G), which serves as a score.
        The policy is updated to increase the probability of actions that lead to high returns.
        """
        R = 0
        returns = []
        # Calculate discounted returns
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = np.array(returns)
        # Normalize returns (Z-score) to stabilize training.
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()
        
        # Update policy for each step in the episode.
        for state, action, G in zip(self.states, self.actions, returns):
            probs = self.policy_table[state]
            grad = -probs
            grad[action] += 1.0  # The policy gradient update
            self.policy_table[state] += ALPHA * G * grad
            # Clip probabilities to prevent them from becoming too small or large.
            self.policy_table[state] = np.clip(self.policy_table[state], 0.01, 0.99)
            self.policy_table[state] /= self.policy_table[state].sum() # Re-normalize probabilities
        
        # Clear episode data.
        self.rewards, self.actions, self.states = [], [], []

# ---------------- Deep Policy Gradient Agent ---------------- #
# Implements a Policy Gradient agent using a neural network (PyTorch).
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Define a simple feedforward neural network.
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Softmax to get a probability distribution over actions
        )
    def forward(self, x):
        return self.fc(x)

class PolicyGradientAgent:
    def __init__(self):
        # Initialize the policy network and optimizer.
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        # Lists to save log probabilities and rewards for an episode.
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Selects an action based on the neural network's output probability distribution.
        The log probability of the chosen action is saved for later loss calculation.
        """
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        """
        Calculates the loss and updates the network weights at the end of an episode.
        The loss is a sum of the negative log probabilities weighted by the discounted returns.
        """
        R = 0
        returns = []
        # Calculate discounted returns
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # Normalize returns for stable training.
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        loss = 0
        for log_prob, R in zip(self.saved_log_probs, returns):
            loss += -log_prob * R
        
        # Backpropagation and weight update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data.
        self.saved_log_probs = []
        self.rewards = []

# ---------------- DQN Agent ---------------- #
# Implements a Deep Q-Network agent with a replay buffer and target network.
class ReplayBuffer:
    def __init__(self, capacity):
        # A deque (double-ended queue) for efficient appending and popping.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition (s, a, r, s', done) to the replay buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Define a simple feedforward Q-network.
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Output a Q-value for each action
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self):
        # Main Q-network
        self.q_net = QNetwork(state_dim, action_dim)
        # Target Q-network (used to stabilize training by providing a fixed target)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())  # Copy weights
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        The epsilon value decays over time to encourage more exploitation.
        """
        if random.random() < self.epsilon:
            return random.randrange(action_dim)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()  # Exploit

    def train_step(self):
        """
        Performs one training step by sampling a batch from the replay buffer.
        The loss is calculated using the Q-learning update rule and the
        network weights are updated.
        """
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample a batch of transitions.
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        # Convert numpy arrays to PyTorch tensors.
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q-values from the main network.
        q_values = self.q_net(states).gather(1, actions)
        
        # Compute target Q-values from the target network (which is not updated during this step).
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + GAMMA * next_q_values * (1 - dones)
        
        # Calculate loss and perform backpropagation.
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon.
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# ---------------- Training Loop ---------------- #
# This is the main function that runs the training for the selected agent.
def train(agent_type="qtable", render=False):
    total_rewards = []

    # Initialize the selected agent based on the user's input.
    if agent_type == "qtable":
        agent = QTableAgent()
    elif agent_type == "policy_table":
        agent = TabularPolicyAgent()
    elif agent_type == "policy_dnn":
        agent = PolicyGradientAgent()
    elif agent_type == "dqn":
        agent = DQNAgent()
    else:
        raise ValueError("Invalid agent_type")

    # Set up real-time plotting.
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))

    # Main training loop for a fixed number of episodes.
    for ep in range(NUM_EPISODES):
        # Reset the environment at the beginning of each episode.
        state, _ = env.reset()
        if agent_type in ["qtable", "policy_table"]:
            state = discretize_state(state) # Discretize initial state if needed
        done = False
        total_reward = 0

        # Loop for a single episode.
        while not done:
            if render:
                env.render() # Render the environment visually if requested.

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Convert boolean values to avoid numpy type issues.
            terminated = bool(terminated)
            truncated = bool(truncated)
            done = terminated or truncated
            total_reward += reward

            # Update agent based on type.
            if agent_type == "qtable":
                next_discrete = discretize_state(next_state)
                agent.update(state, action, reward, next_discrete, done)
                state = next_discrete
            elif agent_type == "policy_table":
                agent.rewards.append(reward)
                state = discretize_state(next_state)
            elif agent_type == "policy_dnn":
                agent.rewards.append(reward)
                state = next_state
            elif agent_type == "dqn":
                agent.memory.push(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state

        # Update the target network for DQN every N episodes.
        if agent_type == "dqn" and ep % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
        
        # Perform end-of-episode updates for policy gradient agents.
        if agent_type in ["policy_table", "policy_dnn"]:
            agent.finish_episode()

        total_rewards.append(total_reward)

        # Real-time reward plotting.
        clear_output(wait=True)
        ax.clear()
        ax.plot(total_rewards, label='Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'{agent_type} Training')
        ax.legend()
        plt.pause(0.01)

    # Disable interactive plotting and show the final plot.
    plt.ioff()
    plt.show()
    return total_rewards

# ---------------- Main Execution ---------------- #
if __name__ == "__main__":
    # Get user input for agent type and rendering option.
    agent_type = input("Choose agent ('qtable', 'policy_table', 'policy_dnn', 'dqn'): ").strip()
    render = input("Render environment? (y/n): ").strip().lower() == 'y'
    
    # Run the training process.
    rewards = train(agent_type, render)
    
    # Close the environment after training is complete.
    env.close()
