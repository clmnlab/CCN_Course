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
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 200
GAMMA = 0.99

# Q-table / Tabular Policy Gradient
ALPHA = 0.1
EPSILON = 0.1
DISCRETE_BINS = 10

# Deep Policy Gradient / DQN
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
# ------------------------------------------------ #

# ---------------- Environment ---------------- #
env = gym.make(ENV_NAME, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ---------------- Helper: Discretize ---------------- #
def discretize_state(state, bins=DISCRETE_BINS):
    upper_bounds = env.observation_space.high
    lower_bounds = env.observation_space.low
    # Clip inf values to avoid overflow
    upper_bounds[1] = min(upper_bounds[1], 5.0)
    upper_bounds[3] = min(upper_bounds[3], 5.0)
    lower_bounds[1] = max(lower_bounds[1], -5.0)
    lower_bounds[3] = max(lower_bounds[3], -5.0)

    ratios = (state - lower_bounds) / (upper_bounds - lower_bounds)
    ratios = np.clip(ratios, 0, 0.999)
    discrete = (ratios * bins).astype(int)
    return tuple(discrete)

# ---------------- Q-table Agent ---------------- #
class QTableAgent:
    def __init__(self):
        self.q_table = np.zeros((DISCRETE_BINS,) * state_dim + (action_dim,))
    def select_action(self, state):
        if np.random.rand() < EPSILON:
            return env.action_space.sample()
        return np.argmax(self.q_table[state])
    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + GAMMA * best_next * (1 - done)
        self.q_table[state][action] += ALPHA * (td_target - self.q_table[state][action])

# ---------------- Tabular Policy Gradient Agent ---------------- #
class TabularPolicyAgent:
    def __init__(self):
        self.policy_table = np.ones((DISCRETE_BINS,) * state_dim + (action_dim,)) / action_dim
        self.rewards = []
        self.actions = []
        self.states = []
    def select_action(self, state):
        probs = self.policy_table[state]
        action = np.random.choice(action_dim, p=probs)
        self.states.append(state)
        self.actions.append(action)
        return action
    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = np.array(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()
        for state, action, G in zip(self.states, self.actions, returns):
            probs = self.policy_table[state]
            grad = -probs
            grad[action] += 1.0
            self.policy_table[state] += ALPHA * G * grad
            self.policy_table[state] = np.clip(self.policy_table[state], 0.01, 0.99)
            self.policy_table[state] /= self.policy_table[state].sum()
        self.rewards, self.actions, self.states = [], [], []

# ---------------- Deep Policy Gradient Agent ---------------- #
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

class PolicyGradientAgent:
    def __init__(self):
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.saved_log_probs = []
        self.rewards = []
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = 0
        for log_prob, R in zip(self.saved_log_probs, returns):
            loss += -log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

# ---------------- DQN Agent ---------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + GAMMA * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# ---------------- Training Loop ---------------- #
def train(agent_type="qtable", render=False):
    total_rewards = []

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

    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))

    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        if agent_type in ["qtable", "policy_table"]:
            state = discretize_state(state)
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Convert to Python bool to avoid np.bool8 issue
            terminated = bool(terminated)
            truncated = bool(truncated)
            done = terminated or truncated
            total_reward += reward

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

        if agent_type == "dqn" and ep % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
        if agent_type in ["policy_table", "policy_dnn"]:
            agent.finish_episode()

        total_rewards.append(total_reward)

        # Real-time reward plot
        clear_output(wait=True)
        ax.clear()
        ax.plot(total_rewards, label='Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'{agent_type} Training')
        ax.legend()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    return total_rewards

# ---------------- Main ---------------- #
if __name__ == "__main__":
    agent_type = input("Choose agent ('qtable', 'policy_table', 'policy_dnn', 'dqn'): ").strip()
    render = input("Render environment? (y/n): ").strip().lower() == 'y'
    rewards = train(agent_type, render)
    env.close()
