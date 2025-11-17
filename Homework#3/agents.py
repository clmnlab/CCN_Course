import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
...
...
...
...

TOTAL_TRIALS = 100
REWARD_BUDGET_PER_ARM = 25
TARGET_ACTION = 0  # 목표 행동 (0: 왼쪽, 1: 오른쪽)
ALT_ACTION = 1 # 목표 행동이 아닌 쪽
# TARGET_ACTION = 0
# ALT_ACTION = 1
# QLearner 설정
LEARNER_ALPHA = 0.1
LEARNER_GAMMA = 0.9
LEARNER_EPSILON = 0.1


# Adversary (DQN) 
# State: learner_state(2) + trial_num(1) + rewards_left(2) = 5
ADV_STATE_SIZE = 5
ADV_ACTION_SIZE = 4 
ADV_HIDDEN_SIZE = 128
ADV_BATCH_SIZE = 64
ADV_GAMMA = 0.99
ADV_EPS_START = 0.9
ADV_EPS_END = 0.05
ADV_EPS_DECAY = 1000
ADV_TAU = 0.005
ADV_LR = 1e-4
ADV_TRAINING_EPISODES = 2000
ADV_MEMORY_CAPACITY = 30000
REWARD_SCALE_FACTOR = 1e4


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLearner:
    """Simple Q-learning agent. Attack target of Adversary RL"""
    def __init__(self, alpha, gamma, epsilon):
        self.q_table = np.zeros(2, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_action = None

    def select_action(self):
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            # 1. Find all indices with the maximum Q-value
            max_q_indices = np.flatnonzero(self.q_table == self.q_table.max())
            # 2. Choose randomly among them
            action = np.random.choice(max_q_indices)
            
        self.last_action = action
        return action

    def update(self, reward):
        if self.last_action is None: return
        old_value = self.q_table[self.last_action]
        next_max = 0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[self.last_action] = new_value

    def get_observable_state(self):
        return self.q_table
        
    def reset(self):
        self.q_table = np.zeros(2, dtype=np.float32)
        self.last_action = None



# --- 2. Adversary Agent:  DQN Agent ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state',))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AdversaryDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(AdversaryDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, ADV_HIDDEN_SIZE)
        self.layer2 = nn.Linear(ADV_HIDDEN_SIZE, ADV_HIDDEN_SIZE)
        self.layer3 = nn.Linear(ADV_HIDDEN_SIZE, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)   #  raw Q-values

class AdversaryAgent:
    def __init__(self, learner, n_episodes=10000):
        self.learner = learner # target learner to attack
        self.n_episodes = n_episodes
        self.policy_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ADV_LR)
        self.memory = ReplayBuffer(ADV_MEMORY_CAPACITY)
        self.rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        self.reward_assignment = [0, 0]
        self.steps_done = 0
        self.n_trials = 100
    def reset(self):
        self.reward_assignment = [0, 0]
        self.rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        learner_state = self.learner.get_observable_state()
        norm_t = 0
        norm_rewards_left = [1, 1]
        state = torch.tensor(
            np.concatenate([learner_state, [norm_t], norm_rewards_left]),
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        return state
    def load_adversary(self, path="adversary_policy.pth"):
        """Loading policy_net weights and apply them to Adversary"""
        try:
            print(f"\n--- loading Adversary weights from {path}  ---")
            # policy_net과 target_net 둘 다에 로드해야 합니다.
            state_dict = torch.load(path, map_location=device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            self.policy_net.eval()
            self.target_net.eval()
            print("--- Loading complete  ---")
        except FileNotFoundError:
            print(f"*** Warning: {path} file is not found. ***")
        except Exception as e:
            print(f"*** Errors: Failure to load weights: {e} ***")
    def save_adversary(self, path="adversary_policy.pth"):
        """Trained Adversary의 policy_net weights are saved."""
        print(f"\n--- Saving Adversary weights to {path}... ---")
        torch.save(self.policy_net.state_dict(), path)
        print("--- Save complete ---")

    def select_action(self, state, is_training=True):
        if is_training:
            sample = random.random()
            eps_threshold = ADV_EPS_END + (ADV_EPS_START - ADV_EPS_END) * np.exp(-1. * self.steps_done / ADV_EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                ...
            else:
                ...
        else:
            with torch.no_grad():
                ...
            
        adv_action_raw = adv_action_tensor.item() # 0, 1, 2, 3               
        # adv_action_raw: # 0: (R:0, L:0) # 1: (R:1, L:0) # 2: (R:0, L:1) # 3: (R:1, L:1)
        # (TARGET_ACTION = 0 (Left), ALT_ACTION = 1 (Right))
        reward_intent = [0, 0]
        if adv_action_raw == 1:
            reward_intent[TARGET_ACTION] = 1
        elif adv_action_raw == 2:
            reward_intent[ALT_ACTION] = 1
        elif adv_action_raw == 3:
            reward_intent[TARGET_ACTION] = 1
            reward_intent[ALT_ACTION] = 1
        
        # # Constratinted Reward Assignment, if budget is exhausted, force no reward  
        # Is there budget left to reward the TARGET_ACTION (Left)?
        if reward_intent[TARGET_ACTION] == 1 and self.rewards_left[TARGET_ACTION] > 0:
            ...
            ...
        
        # Is there budget left to reward the ALT_ACTION (Right)?
        if reward_intent[ALT_ACTION] == 1 and self.rewards_left[ALT_ACTION] > 0:
            ...
            ...

        adv_action = 0  # Default: [0, 0] No reward assigned
        ...
            adv_action = 1  # [1, 0] Reward assigned to left only
        ...
            adv_action = 2  # [0, 1] Reward assigned to right only
        ...
            adv_action = 3  # [1, 1]

        return adv_action    
    def optimize_model(self):
        if len(self.memory) < ADV_BATCH_SIZE: return
        transitions = self.memory.sample(ADV_BATCH_SIZE)
        batch = self.memory.Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(ADV_BATCH_SIZE, device=device)
        ...
        ...
        ...
        ...
        ...
        ...
        return loss.item()
    
    def train(self, is_training=True):        
        losses = []
        mean_rewards_all = []
        total_adv_rewards_all = []
        total_adv_actions_all = []
        # self.n_episodes = n_episodes
        fig, ax = plt.subplots(figsize=(10, 5))
        for i_episode in tqdm(range(self.n_episodes), desc="Adversary Training"):
            self.learner.reset()
            current_episode_experience = []
            total_adv_rewards = 0
            total_adv_actions = []
            state = self.reset()
            for t in range(self.n_trials):
                # 1. adversary action: assigning rewards
                ...
                ...
                # 2. learner selects action
                ...
                # 3. reward to learner based on adversary's assignment
                ...
                ...
                ...
                ...
                # 4. learner updates its Q-values
                ...
                # 5. reward to adversary: did learner choose TARGET_ACTION?
                ...
                ...
                ...
                # 5. record experience
                ...
                ...
                ...
                if t < TOTAL_TRIALS - 1:
                    norm_next_t = (t + 1) / (TOTAL_TRIALS - 1)
                    norm_next_rewards = [
                        self.rewards_left[0] / REWARD_BUDGET_PER_ARM, 
                        self.rewards_left[1] / REWARD_BUDGET_PER_ARM
                    ]
                    next_state = torch.tensor(
                        np.concatenate([next_learner_state, [norm_next_t], norm_next_rewards]),
                        dtype=torch.float32, device=device
                    ).unsqueeze(0)
                self.memory.push(state, adv_action_tensor, adv_reward_tensor, next_state)
                ## Important, don't miss this line to update state
                ...  # Reset for next trial (Really important!!)
                ...  # Update for next state
                total_adv_actions.append(adv_action)
                # End of an episode
            total_adv_rewards_all.append(total_adv_rewards)
            total_adv_actions_all.append(total_adv_actions)
            loss = self.optimize_model()
            losses.append(loss)
            if (i_episode+1) % 10 == 0 and is_training:
                mean_rewards_all.append(np.mean(total_adv_rewards_all[-10:-1]))
                # plot the mean of last 10 adv_rewards
                # --- Real-time update of plot ---
                clear_output(wait=True) # 1. Clear previous output
                ax.clear()              # 2. Clear previous plot
                
                # 3. Plot new data
                ax.plot(mean_rewards_all, 'b-')
                ax.set_xlabel('Episodes (x10)')
                ax.set_ylabel('Mean Adversary Rewards')
                ax.set_title('Adversary Training Progress')
                display(fig)            # 5. Plot display update
            # if (i_episode+1) % 100 == 0:
                print(f"Episode {i_episode+1}, Loss: {loss:.4f}, Reward: {total_adv_rewards}")
                self.update_target_net()
            if (i_episode+1) % 100 == 0:
                # self.save_adversary(path=f"./advRL/adv_QLearner{i_episode+1}.pth")
                self.save_adversary(path=f"./advRL/adv_Q_RNN{i_episode+1}.pth")
        return total_adv_rewards_all, total_adv_actions_all, current_episode_experience
    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
        self.target_net.load_state_dict(target_net_state_dict)
