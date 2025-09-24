import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# --- Hyperparameters ---
LEARNING_RATE = 0.01
GAMMA = 0.99  # Discount factor
NUM_EPISODES = 1000

# --- Policy Network ---
# A simple neural network to approximate the policy.
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)  # Output: Logits for each action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Use softmax to get the probability for each action.
        return F.softmax(self.fc2(x), dim=1)

def main():
    # --- Environment Setup ---
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- Agent Setup ---
    policy = Policy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        
        # --- Data Collection ---
        # Lists to store log probabilities and rewards from the episode.
        log_probs = []
        rewards = []
        done = False

        # 1. Rollout: Run one episode to collect data.
        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            
            # Get action probabilities from the policy network.
            action_probs = policy(state_tensor)
            
            # Create a distribution and sample an action from it.
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Store the log probability of the chosen action. This is log π(a|s).
            log_probs.append(dist.log_prob(action))

            # Interact with the environment.
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            rewards.append(reward)
            state = next_state

        # --- Policy Update ---
        # 2. After the episode, calculate returns and update the policy.
        
        # Calculate discounted returns (G_t) for each timestep.
        # We iterate backward through the rewards to calculate the return from each step.
        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            discounted_returns.insert(0, R)
            
        # Convert to tensor and normalize for stability.
        returns_tensor = torch.FloatTensor(discounted_returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        # 3. Calculate the loss and update the network.
        loss = []
        for log_prob, R in zip(log_probs, returns_tensor):
            # This is the core of REINFORCE:
            # Multiply the log probability of an action by the return that followed.
            # The negative sign is because optimizers minimize, but we want to maximize the objective.
            loss.append(-log_prob * R)

        optimizer.zero_grad()
        # Sum the losses for all timesteps and perform backpropagation.
        policy_loss = torch.cat(loss).sum()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            print(f'Episode {episode + 1}, Total Reward: {sum(rewards)}')

    env.close()

if __name__ == '__main__':
    main()