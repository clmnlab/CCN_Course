import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from IPython.display import clear_output

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
NUM_EPISODES = 300
FINAL_VIS_EPISODES = [50, 100, 300]

class Agent:
    def __init__(self, env, mode='q_learning'):
        """
        Initialize the agent with Q-table and environment.
        """
        self.env = env
        self.mode = mode
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.grid_height = 4
        self.grid_width = 12

        # Font setting for Korean/English visualization
        import platform
        if 'Darwin' in platform.system():
            plt.rcParams['font.family'] = 'AppleGothic'
        elif 'Windows' in platform.system():
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < EPSILON:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Q-Learning or SARSA update rule.
        """
        if self.mode == 'q_learning':
            best_next_q = np.max(self.q_table[next_state])
            td_target = reward + GAMMA * best_next_q * (1 - done)
        elif self.mode == 'sarsa':
            next_q = self.q_table[next_state, next_action]
            td_target = reward + GAMMA * next_q * (1 - done)

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += ALPHA * td_error

    def visualize_q_table(self, episode=None):
        """
        Visualize Q-table as a heatmap with arrows.
        Cliff, Goal, Start included.
        Use min Q-value for color mapping to highlight extreme negative rewards (Cliff).
        """
        fig, ax = plt.subplots(figsize=(self.grid_width, self.grid_height))

        # Colormap normalization
        q_min = np.min(self.q_table)
        q_max = np.max(self.q_table)
        if q_max == q_min:
            norm = mcolors.Normalize(vmin=-100, vmax=0)
        else:
            norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
        cmap = plt.cm.RdYlGn  # Red (low) -> Yellow -> Green (high)

        # Draw grid
        ax.set_xticks(np.arange(self.grid_width + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.grid_height + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                state_idx = r * self.grid_width + c

                # Determine Q-value for color mapping
                # Use min Q-value to emphasize extreme negative rewards for Cliff
                q_val_for_color = np.min(self.q_table[state_idx])
                rect_color = cmap(norm(q_val_for_color))

                # Draw rectangle
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=rect_color, edgecolor='black', linewidth=0.5))

                # Overlay text for special cells
                cell_text = ''
                if state_idx == self.grid_height * self.grid_width - self.grid_width:
                    cell_text = 'S'  # Start
                elif state_idx == self.grid_height * self.grid_width - 1:
                    cell_text = 'G'  # Goal
                elif r == self.grid_height - 1 and 0 < c < self.grid_width - 1:
                    cell_text = 'C'  # Cliff
                if cell_text:
                    ax.text(c, r, cell_text, ha='center', va='center', fontsize=12, weight='bold')

                # Draw arrow for the action with the highest Q-value (except Goal)
                if state_idx != self.grid_height * self.grid_width - 1:
                    max_action = np.argmax(self.q_table[state_idx])
                    dx, dy = 0, 0
                    if max_action == 0: dy = 0.3  # Up
                    elif max_action == 1: dx = 0.3  # Right
                    elif max_action == 2: dy = -0.3  # Down
                    elif max_action == 3: dx = -0.3  # Left
                    ax.arrow(c, r, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')

        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(-0.5, self.grid_height - 0.5)
        ax.invert_yaxis()
        ax.set_title(f"{self.mode.upper()} Q-Table (Episode: {episode + 1 if episode is not None else 'Final'})")
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Q-value')
        plt.tight_layout()
        plt.show()

    def visualize_path(self, path, episode=None):
        """
        Visualize agent path on the grid.
        """
        fig, ax = plt.subplots(figsize=(self.grid_width, self.grid_height))

        # Draw grid with start/goal/cliff
        ax.set_xticks(np.arange(self.grid_width + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.grid_height + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                state_idx = r * self.grid_width + c
                rect_color = 'lightgray'
                cell_text = ''
                if state_idx == self.grid_height * self.grid_width - self.grid_width:
                    rect_color = 'skyblue'
                    cell_text = 'S'
                elif state_idx == self.grid_height * self.grid_width - 1:
                    rect_color = 'lightgreen'
                    cell_text = 'G'
                elif r == self.grid_height - 1 and 0 < c < self.grid_width - 1:
                    rect_color = 'darkred'
                    cell_text = 'C'
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=rect_color, edgecolor='black', linewidth=0.5))
                if cell_text:
                    ax.text(c, r, cell_text, ha='center', va='center', fontsize=12, weight='bold')

        # Draw path
        for i in range(len(path) - 1):
            r1, c1 = divmod(path[i], self.grid_width)
            r2, c2 = divmod(path[i + 1], self.grid_width)
            ax.plot([c1, c2], [r1, r2], color='blue', linewidth=2, marker='o', markersize=5)
            ax.text((c1 + c2) / 2, (r1 + r2) / 2, str(i), color='red', fontsize=8, ha='center', va='center')

        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(-0.5, self.grid_height - 0.5)
        ax.invert_yaxis()
        ax.set_title(f"{self.mode.upper()} Path (Episode: {episode + 1 if episode is not None else 'Final'})")
        plt.tight_layout()
        plt.show()


def train(agent, env):
    """
    Train the agent on the environment.
    """
    rewards = []
    print(f"--- {agent.mode.upper()} Training Start ---")
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        action = agent.select_action(state)
        done = False
        total_reward = 0
        path = [state]

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            total_reward += reward
            path.append(state)

        rewards.append(total_reward)

        if (episode + 1) in FINAL_VIS_EPISODES:
            clear_output(wait=True)
            print(f"--- {agent.mode.upper()} Episode {episode + 1} Visualization ---")
            agent.visualize_q_table(episode)
            agent.visualize_path(path, episode)

    return rewards


def plot_rewards(q_rewards, sarsa_rewards):
    """
    Plot total rewards for Q-Learning and SARSA.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(q_rewards, label='Q-Learning')
    plt.plot(sarsa_rewards, label='SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs SARSA on CliffWalking')
    plt.axhline(y=-13, color='gray', linestyle='--', label='Optimal Path Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    q_agent = Agent(env, 'q_learning')
    sarsa_agent = Agent(env, 'sarsa')

    q_rewards = train(q_agent, env)
    sarsa_rewards = train(sarsa_agent, env)

    env.close()
    plot_rewards(q_rewards, sarsa_rewards)
