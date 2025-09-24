import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
NUM_EPISODES = 1000

# --- Actor-Critic Network ---
# A single network with two heads: one for the policy (Actor) and one for the value (Critic).
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.shared_layer = nn.Linear(state_dim, 128)
        
        # Actor-specific layer (outputs action probabilities)
        self.actor_head = nn.Linear(128, action_dim)
        
        # Critic-specific layer (outputs a single state value)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.shared_layer(x))
        
        # Calculate action probabilities (for the Actor)
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Calculate the state value (for the Critic)
        state_value = self.critic_head(x)
        
        return action_probs, state_value

def main():
    # --- Environment Setup ---
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- Agent Setup ---
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Unlike REINFORCE, Actor-Critic updates at every step.
        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            
            # 1. Get action probabilities and state value from the network.
            action_probs, state_value = model(state_tensor)

            # 2. Sample an action from the distribution.
            dist = Categorical(action_probs)
            action = dist.sample()

            # 3. Interact with the environment.
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            # 4. Calculate the Advantage and the Target Value.
            with torch.no_grad():
                # Get the value of the next state from the critic.
                _, next_state_value = model(torch.FloatTensor(np.array(next_state)).unsqueeze(0))
                # If the episode is done, the value of the next state is 0.
                target_value = reward + GAMMA * next_state_value * (1 - done)

            # Advantage: How much better was the action than the critic's expectation?
            # A(s, a) = Q(s, a) - V(s) ≈ (r + γV(s')) - V(s)
            advantage = target_value - state_value

            # 5. Calculate the two losses.
            # Actor Loss (Policy Loss): Encourages actions that led to a positive advantage.
            # We use .detach() on advantage to prevent gradients from flowing into the critic's weights.
            actor_loss = -dist.log_prob(action) * advantage.detach()

            # Critic Loss (Value Loss): Aims to make the critic's value estimate more accurate.
            # It's the mean squared error between the predicted value and the target value.
            critic_loss = F.mse_loss(state_value, target_value)
            
            # Total loss is the sum of both.
            total_loss = actor_loss + critic_loss

            # 6. Update the network.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            state = next_state

        if (episode + 1) % 50 == 0:
            print(f'Episode {episode + 1}, Total Reward: {total_reward}')

    env.close()

if __name__ == '__main__':
    main()