import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ---------------- Hyperparameters ---------------- #
ENV_NAME = "BreakoutNoFrameskip-v4"
NUM_EPISODES = 500
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 1000
STACK_SIZE = 4
FRAME_SKIP = 4
# ------------------------------------------------ #

# ---------------- Replay Buffer ---------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # stack -> ensures a proper ndim array instead of object-dtype
        states = np.stack(states)            # shape: (batch, C, H, W)  or (batch, state_dim)
        next_states = np.stack(next_states)

        actions = np.array(actions, dtype=np.int64)          # (batch,)
        rewards = np.array(rewards, dtype=np.float32)       # (batch,)
        dones = np.array(dones, dtype=np.float32)           # (batch,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---------------- Q-Network ---------------- #
class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        # conv feature extractor (no FC)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out((c, h, w))  # compute flattened conv output
        # fully connected head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # Pass a dummy tensor through conv to determine flattened size
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.shape[1:])))  # channels * H * W

    def forward(self, x):
        x = self.conv(x)   # (B, C_out, H_out, W_out)
        x = self.fc(x)     # (B, n_actions)
        return x

# ---------------- Preprocessing ---------------- #
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess(frame)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84,84), dtype=np.float32) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    return np.array(stacked_frames), stacked_frames

# ---------------- DQN Agent ---------------- #
class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(input_shape, n_actions).to(self.device)
        self.target_net = QNetwork(input_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.n_actions = n_actions

    def select_action(self, state):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + GAMMA * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ---------------- Visualization ---------------- #
def visualize_frame(frame, q_values, action):
    """Display frame with Q-value bar and current action"""
    plt.figure(figsize=(4,4))
    plt.imshow(frame)
    plt.axis('off')
    plt.title(f"Action: {action}, Q-values: {np.round(q_values, 2)}")
    plt.show()
    clear_output(wait=True)

# ---------------- Training Loop ---------------- #
def train():
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    n_actions = env.action_space.n
    agent = DQNAgent((STACK_SIZE, 84, 84), n_actions)
    stacked_frames = deque(maxlen=STACK_SIZE)
    total_rewards = []

    for ep in range(NUM_EPISODES):
        frame, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, frame, True)
        done = False
        ep_reward = 0

        while not done:
            # Frame skip
            frames = []
            total_reward = 0
            for _ in range(FRAME_SKIP):
                action = agent.select_action(state)
                next_frame, reward, terminated, truncated, _ = env.step(action)
                terminated, truncated = bool(terminated), bool(truncated)
                done = terminated or truncated
                frames.append(next_frame)
                total_reward += reward
                if done:
                    break
            max_frame = np.maximum(frames[0], frames[-1])
            next_state, stacked_frames = stack_frames(stacked_frames, max_frame, False)
            reward_clipped = np.sign(total_reward)
            agent.memory.push(state, action, reward_clipped, next_state, done)
            agent.train_step()

            # Visualization
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.q_net(state_tensor).cpu().numpy()[0]
            visualize_frame(max_frame, q_vals, action)

            state = next_state
            ep_reward += total_reward

            if agent.steps_done % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.q_net.state_dict())
            agent.steps_done += 1

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return total_rewards

# ---------------- Main ---------------- #
if __name__ == "__main__":
    rewards = train()
