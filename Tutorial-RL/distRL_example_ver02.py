"""
Title: Policy Gradient agent on a tabular Temporal-Difference (TD) based Linear MDP

Description:
------------
This script implements a simplified policy gradient (PG) agent that learns 
to predict the timing of rewards in a synthetic Linear Markov Decision Process (MDP). 
The environment pre-computes state values using a TD-learning procedure 
across multiple gamma-discount channels. 
The PG agent observes these values and learns to predict the correct reward time.

The setup is inspired by the computational model introduced in:
- Gershman, S. J. et al. (2025). "A neural model of temporal reward prediction."
  Nature, 626, 123–130. https://doi.org/10.1038/s41586-025-08929-9

Key Components:
---------------
1. **Decoder (Policy Network)**: 
   - A small MLP mapping environment-provided "value features" to logits over possible targets.
   - Outputs a categorical distribution for action selection.

2. **LinearMDPEnv (Environment)**:
   - A synthetic environment where rewards occur at random times with random magnitudes.
   - TD learning simulates how an internal value representation might be updated.
   - Observations are the value estimates at the cue time, across discount factors.

3. **PGAgent (Policy Gradient Agent)**:
   - Samples actions from its policy (or greedily selects for evaluation).
   - Trains by maximizing log-likelihood of the correct target weighted by returns (reward-to-go).
   - Uses REINFORCE with return-to-go baseline.

4. **Training Loop**:
   - Collects trajectories of (observation, action, reward).
   - Computes reward-to-go and updates the policy.
   - Tracks agent’s accuracy in predicting the correct reward time.

Outputs:
--------
- Trains the PG agent for multiple epochs.
- Logs performance (accuracy).
- Saves a training curve plot as "training_performance.png".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt


# ---------- Policy Network ----------
class Decoder(nn.Module):
    """
    Policy Network (a.k.a. Decoder)
    --------------------------------
    - Maps the observation (vector of TD values at cue for each gamma)
      into logits over possible discrete targets.
    - This is a feedforward MLP with hidden layers.
    - Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape),
        )

    def forward(self, x):
        return self.fc(x)


# ---------- Environment ----------
class LinearMDPEnv:
    """
    Synthetic Linear MDP Environment
    --------------------------------
    - Simulates reward timing tasks in a simplified environment.
    - Rewards occur at a randomly sampled time-step with a random magnitude.
    - For each episode:
        * TD learning is run multiple iterations to estimate state values
          across different gamma-discount channels.
        * The agent only observes the "cue values" (first time-step values).
    - Noise can be added to observations and reward timing perception.

    Parameters:
    -----------
    - total_time : int, length of the episode in time steps
    - gammas : list of discount factors
    - max_reward_magn : max reward magnitude
    - noise_values : noise variance for observations
    - random_reward : if True, reward magnitude is sampled randomly
    - noise_perceived_time : variability in perceived reward timing
    - min_num_td_steps, max_num_td_steps : number of TD iterations to run
    """

    def __init__(self, total_time=12, gammas=None, max_reward_magn=6,
                 noise_values=0, random_reward=True, noise_perceived_time=0,
                 min_num_td_steps=59, max_num_td_steps=99):
        self.total_time = total_time
        self.gammas = gammas
        self.max_reward_magn = max_reward_magn
        self.noise_values = noise_values
        self.random_reward = random_reward
        self.noise_perceived_time = noise_perceived_time
        self.min_num_td_steps = min_num_td_steps
        self.max_num_td_steps = max_num_td_steps

    def reset(self):
        """
        Initialize a new episode.
        - Sample reward time and magnitude.
        - Run TD learning to compute value functions.
        - Return initial observation (cue values).
        """
        # learning rate for tabular TD
        self.alpha = np.random.normal(loc=0.1, scale=0.001)
        # number of TD updates to run
        self.td_it = np.random.choice(range(self.min_num_td_steps, self.max_num_td_steps))
        # randomly choose reward time (not first or last time step)
        self.rew_time = np.random.choice(range(1, self.total_time - 1))
        # reward magnitude sampled randomly
        self.rew_magn = 1 + np.random.choice(self.max_reward_magn) if self.random_reward else 1

        # value table: [time_step, gamma_channel]
        self.values = np.zeros((self.total_time, len(self.gammas)))

        # Run TD learning updates
        for _ in range(self.td_it):
            # noisy perception of reward time
            perceived_rew_time = int(abs(
                self.rew_time + np.random.normal(0, self.rew_time * self.noise_perceived_time)
            ))
            # backward TD sweep
            for i in reversed(range(self.total_time - 1)):
                for j, gamma in enumerate(self.gammas):
                    rew = self.rew_magn if i == perceived_rew_time else 0
                    if i == (self.total_time - 2):
                        # terminal update
                        self.values[i, j] += self.alpha * (rew - self.values[i, j])
                    else:
                        # TD backup
                        self.values[i, j] += self.alpha * (
                            rew + gamma * self.values[i + 1, j] - self.values[i, j]
                        )

        # Add Gaussian noise to the observation values
        for j in range(len(self.gammas)):
            noise = np.random.normal(scale=self.noise_values)
            self.values[0, j] += noise

        return self.get_obs(), self.rew_time, self.rew_magn

    def get_obs(self):
        """Return the current observation (cue values)."""
        return self.values[0, :]


# ---------- Agent ----------
class PGAgent:
    """
    Policy Gradient Agent (REINFORCE)
    ---------------------------------
    - Learns a mapping from environment observations to target predictions.
    - Uses REINFORCE update with reward-to-go as weights.
    - Supports:
        * Exploration with epsilon-greedy.
        * Greedy planning action (argmax).
    """

    def __init__(self, gammas, possible_targets, target_fn, lr=1e-3, eps=0.3):
        self.gammas = gammas
        self.possible_targets = possible_targets
        self.target_fn = target_fn
        self.eps = eps

        self.net = Decoder(len(gammas), len(possible_targets))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def get_policy(self, obs):
        """
        Given an observation, return a categorical policy distribution.
        - obs can be a numpy array or tensor.
        - Converts obs to torch tensor, forwards through network to get logits.
        """
        logits = self.net(torch.as_tensor(obs, dtype=torch.float32))
        return Categorical(logits=logits)

    def select_action(self, obs):
        """
        Sample an action (with epsilon-greedy exploration).
        - With prob eps: random action
        - Otherwise: sample from policy distribution
        """
        if np.random.rand() < self.eps:
            return np.random.choice(len(self.possible_targets))
        return self.get_policy(obs).sample().item()

    def plan_action(self, obs):
        """
        Greedy action selection (for evaluation).
        - Picks action with maximum probability from policy.
        """
        return self.get_policy(obs).probs.argmax().item()

    def update_policy(self, batch_obs, batch_acts, batch_weights):
        """
        Update policy using one batch of experience.
        - batch_obs: list of observations
        - batch_acts: list of actions taken
        - batch_weights: reward-to-go for each action
        """
        self.optimizer.zero_grad()
        obs = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        acts = torch.as_tensor(np.array(batch_acts), dtype=torch.int64)
        weights = torch.as_tensor(np.array(batch_weights), dtype=torch.float32)

        # log probability of actions
        logp = self.get_policy(obs).log_prob(acts)
        # loss is negative weighted log likelihood (sum version)
        loss = -(logp * weights).sum()
        loss.backward()
        self.optimizer.step()


# ---------- Utility ----------
def reward_to_go(rews):
    """
    Compute reward-to-go for a sequence of rewards.
    - For each time-step i, compute sum of rewards from i to the end.
    - This is exactly the REINFORCE 'return-to-go' baseline.
    """
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


# ---------- Training Loop ----------
def train_pg(agent, env, epochs=2000, batch_size=100):
    """
    Train the policy gradient agent in the environment.

    Steps per epoch:
    ----------------
    - Collect batch of trajectories (obs, action, reward).
    - Compute reward-to-go for each episode.
    - Update policy parameters.
    - Track accuracy (whether greedy action = true target).

    Returns:
    --------
    - performance: list of accuracy values per epoch
    """

    performance = []
    for ep in range(epochs):
        batch_obs, batch_acts, batch_weights, corrects = [], [], [], []

        while True:
            # Reset environment, get observation, reward time & magnitude
            obs, rew_time, rew_magn = env.reset()
            obs = obs.flatten()

            # Agent action (exploration)
            act = agent.select_action(obs)
            # Planning (greedy evaluation)
            plan = agent.plan_action(obs)

            # Assign reward: correct if predicted target matches ground truth
            rew = 1 if agent.possible_targets[act] == agent.target_fn(rew_time, rew_magn) else 0
            correct = 1 if agent.possible_targets[plan] == agent.target_fn(rew_time, rew_magn) else 0

            # Store episode experience
            batch_obs.append(obs)
            batch_acts.append(act)
            batch_weights += list(reward_to_go([rew]))
            corrects.append(correct)

            if len(batch_obs) > batch_size:
                break

        # Policy update
        agent.update_policy(batch_obs, batch_acts, batch_weights)
        performance.append(np.mean(corrects))

        if ep % 50 == 0:
            print(f"Epoch {ep}: Performance {np.mean(corrects):.3f}")

    return performance


# ---------- Target Definition ----------
class Target:
    """
    Target function definition
    --------------------------
    - Defines how the "correct" target is computed for each episode.
    - Two modes:
        * delta: correct target is simply the reward time.
        * hyperbolic: correct target is scaled by hyperbolic discount of time.
    """

    def __init__(self, discount_type="delta", discount_param=0.9):
        self.discount_type = discount_type
        self.discount_param = discount_param

    def compute_target(self, rew_time, rew_magn):
        if self.discount_type == "hyperbolic":
            return rew_magn * (1 / (1 + self.discount_param * rew_time))
        elif self.discount_type == "delta":
            return rew_time

    def possible_targets(self, max_rew_time, max_rew_magn=6):
        """
        Enumerate all possible target values.
        - For delta: just reward times [0..max_time)
        - For hyperbolic: all combinations of reward time and magnitude
        """
        targets = []
        if self.discount_type == "delta":
            for t in range(max_rew_time):
                targets.append(self.compute_target(t, 1))
        else:
            for t in range(max_rew_time):
                for r in range(max_rew_magn):
                    targets.append(self.compute_target(t, 1+r))
        return targets


# ---------- Main ----------
if __name__ == "__main__":
    # Discount factors used in parallel TD channels
    gammas = [0.6, 0.9, 0.99]
    # Define target function (delta = use reward time directly)
    target = Target(discount_type="delta")
    # Construct possible targets (all possible reward times)
    possible_targets = target.possible_targets(max_rew_time=15, max_rew_magn=15)

    # Initialize environment and agent
    env = LinearMDPEnv(total_time=15, gammas=gammas, max_reward_magn=15)
    agent = PGAgent(gammas, possible_targets, target.compute_target)

    # Train agent
    performance = train_pg(agent, env, epochs=500, batch_size=100)

    # Plot training performance curve
    plt.plot(performance)
    plt.xlabel("Epoch")
    plt.ylabel("Performance (accuracy)")
    plt.title("Training Performance")
    plt.savefig("training_performance.png")
    plt.show()
