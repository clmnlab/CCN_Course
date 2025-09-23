# This script implements a Deep Q-Learning (DQN) agent for training on an Atari environment,
# specifically "ALE/Breakout-v5". It includes standard practices for training on
# visual input, such as frame preprocessing, frame stacking, experience replay,
# and using a target network.

import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2 # OpenCV for image preprocessing
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ---------------- Hyperparameters ---------------- #
# These parameters control the training process and the agent's architecture.
ENV_NAME = "ALE/Breakout-v5"  # The name of the Atari environment.

NUM_EPISODES = 500             # Total number of episodes to train.
GAMMA = 0.99                   # Discount factor for future rewards.
LR = 1e-4                       # Learning rate for the Adam optimizer.
BATCH_SIZE = 32                 # Mini-batch size for training from the replay buffer.
MEMORY_SIZE = 100000            # Maximum size of the experience replay buffer.
EPSILON_START = 1.0             # Initial value of epsilon for the epsilon-greedy policy.
EPSILON_END = 0.01              # Minimum epsilon value.
EPSILON_DECAY = 0.995           # Multiplicative decay factor for epsilon per step.
TARGET_UPDATE = 1000            # Number of steps after which to update the target Q-network.
STACK_SIZE = 4                  # Number of frames to stack to represent a single state.
FRAME_SKIP = 4                  # Number of frames to skip, taking the same action for each.
# ------------------------------------------------ #

# ---------------- Replay Buffer ---------------- #
class ReplayBuffer:
    """
    Experience replay buffer for storing transitions and sampling mini-batches.
    It stores tuples of (state, action, reward, next_state, done).
    A deque is used for efficient memory management with a fixed size.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Adds a single transition (experience) to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Samples a random mini-batch of transitions from the buffer.
        This breaks the correlation between consecutive experiences.
        
        Returns:
            A tuple of numpy arrays for states, actions, rewards, next states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of arrays to single numpy arrays for efficient tensor conversion.
        states = np.stack(states)            
        next_states = np.stack(next_states)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current number of transitions stored."""
        return len(self.buffer)

# ---------------- Q-Network ---------------- #
class QNetwork(nn.Module):
    """
    Deep Q-Network with convolutional layers designed for visual input (Atari frames).
    The network takes a stacked frame tensor and outputs Q-values for each possible action.
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        # Convolutional layers to extract features from the image input.
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Compute the flattened size of the convolutional output to connect to linear layers.
        conv_out_size = self._get_conv_out((c, h, w))  
        # Fully connected layers that map the extracted features to Q-values.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions) # Final layer outputs one Q-value per action.
        )

    def _get_conv_out(self, shape):
        """
        Helper function to calculate the output size of the convolutional layers.
        It passes a dummy tensor through the layers to determine the flattened size.
        """
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.shape[1:])))  # channels * H * W

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = self.conv(x)   # Pass through convolutional layers
        x = self.fc(x)     # Pass through fully connected layers
        return x

# ---------------- Preprocessing ---------------- #
def preprocess(frame):
    """
    Preprocesses a raw RGB Atari frame.
    It converts the frame to grayscale, resizes it to 84x84, and normalizes pixel values.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, frame, is_new_episode):
    """
    Stacks frames to provide temporal context, which is essential for understanding
    motion in the game. The state is a tensor of the last `STACK_SIZE` frames.
    
    Args:
        stacked_frames (deque): The current deque of frames.
        frame (np.array): The current preprocessed frame.
        is_new_episode (bool): Flag to reset the stack at the start of an episode.
    Returns:
        np.array: The new stacked state.
        deque: The updated stacked frames deque.
    """
    frame = preprocess(frame)
    if is_new_episode:
        # Initialize the stack with black frames at the start of a new episode.
        stacked_frames = deque([np.zeros((84,84), dtype=np.float32) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    # Convert deque to a numpy array to be used as input for the network.
    return np.array(stacked_frames), stacked_frames

# ---------------- DQN Agent ---------------- #
class DQNAgent:
    """
    The main Deep Q-Learning agent. It manages the two Q-networks (online and target),
    the optimizer, the replay buffer, and the epsilon-greedy policy.
    """
    def __init__(self, input_shape, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The main Q-network (online network) for action selection.
        self.q_net = QNetwork(input_shape, n_actions).to(self.device)
        # The target Q-network, used to compute stable Q-targets.
        self.target_net = QNetwork(input_shape, n_actions).to(self.device)
        # Copy initial weights from the online network to the target network.
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.n_actions = n_actions

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        Epsilon decays multiplicatively over time.
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions) # Exploration
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item() # Exploitation

    def train_step(self):
        """
        Performs a single training step.
        It samples a batch, computes the loss, and updates the online network's weights.
        """
        if len(self.memory) < BATCH_SIZE:
            return
        # Sample a random mini-batch from the replay buffer.
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Convert numpy arrays to PyTorch tensors and move to the appropriate device.
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute the Q-values for the states and actions in the batch.
        q_values = self.q_net(states).gather(1, actions)
        
        # Compute the target Q-values using the target network (which is not updated during this step).
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + GAMMA * next_q_values * (1 - dones)
            
        # Compute the Mean Squared Error loss between the computed Q-values and the target Q-values.
        loss = nn.MSELoss()(q_values, target)
        
        # Backpropagate the loss and update the online network's weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ---------------- Visualization ---------------- #
def visualize_frame(frame, q_values, action):
    """
    Displays a single frame along with the Q-values and the action taken.
    This helps in understanding what the agent is "seeing" and what it's choosing.
    """
    plt.figure(figsize=(4,4))
    plt.imshow(frame)
    plt.axis('off')
    plt.title(f"Action: {action}, Q-values: {np.round(q_values, 2)}")
    plt.show()
    clear_output(wait=True)

# ---------------- Training Loop ---------------- #
def train():
    """
    The main training loop. It sets up the environment and the agent,
    and then runs a loop for the specified number of episodes.
    """
    # Create the environment with a render mode that provides RGB frames.
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    n_actions = env.action_space.n
    agent = DQNAgent((STACK_SIZE, 84, 84), n_actions)
    stacked_frames = deque(maxlen=STACK_SIZE)
    total_rewards = []

    for ep in range(NUM_EPISODES):
        # Reset the environment at the start of each episode.
        frame, _ = env.reset()
        # Initialize the stacked frames for the new episode.
        state, stacked_frames = stack_frames(stacked_frames, frame, True)
        done = False
        ep_reward = 0

        while not done:
            # Frame skipping: take the same action for multiple frames.
            # This makes the training faster.
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
            
            # Max-pooling across skipped frames to capture the ball's position.
            # This is a common technique in Atari environments.
            max_frame = np.maximum(frames[0], frames[-1])
            # Stack the new frame to form the next state.
            next_state, stacked_frames = stack_frames(stacked_frames, max_frame, False)
            
            # Reward clipping: simplifies the reward signal to -1, 0, or +1.
            reward_clipped = np.sign(total_reward)
            
            # Push the transition to the replay buffer.
            agent.memory.push(state, action, reward_clipped, next_state, done)
            # Perform a training step.
            agent.train_step()

            # Visualization of the current frame and Q-values.
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.q_net(state_tensor).cpu().numpy()[0]
            visualize_frame(max_frame, q_vals, action)

            state = next_state
            ep_reward += total_reward

            # Update the target network periodically.
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.q_net.state_dict())
            agent.steps_done += 1

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return total_rewards


# ---------------- Main ---------------- #
if __name__ == "__main__":
    rewards = train()
