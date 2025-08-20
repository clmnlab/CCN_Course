import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

# --- 하이퍼파라미터 설정 ---
# 실험 환경 설정
TOTAL_TRIALS = 100
REWARD_BUDGET_PER_ARM = 25
TARGET_ACTION = 0  # 목표 행동 (0: 왼쪽, 1: 오른쪽)

# QLearner 설정
LEARNER_ALPHA = 0.1
LEARNER_GAMMA = 0.9
LEARNER_EPSILON = 0.1

# Adversary (DQN) 설정
ADV_STATE_SIZE = 5 # Q-table(2) + trial_num(1) + rewards_left(2)
ADV_ACTION_SIZE = 2 # 0: 왼쪽 보상 할당, 1: 오른쪽 보상 할당
ADV_HIDDEN_SIZE = 128
ADV_BATCH_SIZE = 64
ADV_GAMMA = 0.99
ADV_EPS_START = 0.9
ADV_EPS_END = 0.05
ADV_EPS_DECAY = 1000
ADV_TAU = 0.005
ADV_LR = 1e-4
ADV_TRAINING_EPISODES = 500
ADV_MEMORY_CAPACITY = 10000

# 평가 설정
EVALUATION_EPISODES = 100

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Learner Agent: Q-Learning 기반 Bandit 에이전트 ---
class QLearner:
    """간단한 Q-learning 에이전트. Adversary의 공격 대상."""
    def __init__(self, alpha, gamma, epsilon):
        self.q_table = np.zeros(2, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_action = None

    def choose_action(self):
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(self.q_table)
        self.last_action = action
        return action

    def update(self, reward):
        if self.last_action is None: return
        old_value = self.q_table[self.last_action]
        next_max = 0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[self.last_action] = new_value

# --- 2. Adversary Agent: 보상 전략을 학습하는 DQN 에이전트 ---
class ReplayBuffer:
    """DQN을 위한 리플레이 버퍼 (캡슐화 개선)."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        # Transition 데이터 구조를 클래스 내부에서 정의
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """전환(transition)을 메모리에 저장합니다."""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        """메모리에서 무작위로 배치 크기만큼 샘플링합니다."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AdversaryDQN(nn.Module):
    """보상 제공 전략을 학습하는 DQN 모델."""
    def __init__(self, n_observations, n_actions):
        super(AdversaryDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, ADV_HIDDEN_SIZE)
        self.layer2 = nn.Linear(ADV_HIDDEN_SIZE, ADV_HIDDEN_SIZE)
        self.layer3 = nn.Linear(ADV_HIDDEN_SIZE, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class AdversaryAgent:
    """Adversary의 정책, 학습, 메모리를 모두 캡슐화하는 클래스."""
    def __init__(self):
        self.policy_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ADV_LR)
        self.memory = ReplayBuffer(ADV_MEMORY_CAPACITY)
        self.steps_done = 0

    def select_action(self, state, is_training=True):
        """Epsilon-Greedy 전략에 따라 행동을 선택합니다."""
        if is_training:
            sample = random.random()
            eps_threshold = ADV_EPS_END + (ADV_EPS_START - ADV_EPS_END) * \
                np.exp(-1. * self.steps_done / ADV_EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(ADV_ACTION_SIZE)]], device=device, dtype=torch.long)
        else: # 평가 시에는 탐욕적(greedy) 정책 사용
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def store_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        """리플레이 버퍼에서 샘플링하여 모델을 한 스텝 최적화합니다."""
        if len(self.memory) < ADV_BATCH_SIZE:
            return

        transitions = self.memory.sample(ADV_BATCH_SIZE)
        batch = self.memory.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(ADV_BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * ADV_GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """타겟 네트워크를 정책 네트워크의 가중치로 부드럽게 업데이트합니다."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
        self.target_net.load_state_dict(target_net_state_dict)

# --- 3. 실험 전체를 관리하는 클래스 ---
class Experiment:
    """훈련과 평가 과정을 총괄하는 클래스."""
    def __init__(self):
        print(f"Using device: {device}")
        self.q_learner_template = QLearner(alpha=LEARNER_ALPHA, gamma=LEARNER_GAMMA, epsilon=LEARNER_EPSILON)
        self.adversary_agent = AdversaryAgent()

    def train(self):
        """Adversary Agent를 훈련시킵니다."""
        print("--- Adversary 훈련 시작 ---")
        for i_episode in tqdm(range(ADV_TRAINING_EPISODES), desc="Adversary Training"):
            q_learner = copy.deepcopy(self.q_learner_template)
            rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
            
            for t in range(TOTAL_TRIALS):
                state = torch.tensor(
                    np.concatenate([q_learner.q_table, [t, rewards_left[0], rewards_left[1]]]),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)

                adv_action_tensor = self.adversary_agent.select_action(state)
                adv_action = adv_action_tensor.item()
                
                if rewards_left[adv_action] <= 0: adv_action = 1 - adv_action
                
                reward_assignment = [0, 0]
                if rewards_left[adv_action] > 0:
                    reward_assignment[adv_action] = 1
                    rewards_left[adv_action] -= 1

                learner_action = q_learner.choose_action()
                reward_to_learner = reward_assignment[learner_action]
                q_learner.update(reward_to_learner)

                adversary_reward = 1.0 if learner_action == TARGET_ACTION else 0.0
                adversary_reward_tensor = torch.tensor([adversary_reward], device=device)
                
                next_state = None
                if t < TOTAL_TRIALS - 1:
                    next_state = torch.tensor(
                        np.concatenate([q_learner.q_table, [t + 1, rewards_left[0], rewards_left[1]]]),
                        dtype=torch.float32, device=device
                    ).unsqueeze(0)

                self.adversary_agent.store_transition(state, adv_action_tensor, next_state, adversary_reward_tensor)
                self.adversary_agent.optimize_model()
                self.adversary_agent.update_target_net()

        print("--- Adversary 훈련 완료 ---\n")

    def evaluate(self):
        """훈련된 Adversary를 평가합니다."""
        print("--- 평가 시작 ---")
        target_action_counts = []
        for _ in tqdm(range(EVALUATION_EPISODES), desc="Evaluation"):
            q_learner = copy.deepcopy(self.q_learner_template)
            rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
            count = 0

            for t in range(TOTAL_TRIALS):
                state = torch.tensor(
                    np.concatenate([q_learner.q_table, [t, rewards_left[0], rewards_left[1]]]),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)

                adv_action = self.adversary_agent.select_action(state, is_training=False).item()
                
                if rewards_left[adv_action] <= 0: adv_action = 1 - adv_action
                
                reward_assignment = [0, 0]
                if rewards_left[adv_action] > 0:
                    reward_assignment[adv_action] = 1
                    rewards_left[adv_action] -= 1

                learner_action = q_learner.choose_action()
                reward_to_learner = reward_assignment[learner_action]
                q_learner.update(reward_to_learner)

                if learner_action == TARGET_ACTION:
                    count += 1
            target_action_counts.append(count)
        
        self.plot_results(target_action_counts)

    def plot_results(self, target_action_counts):
        """평가 결과를 시각화합니다."""
        bias = (np.mean(target_action_counts) / TOTAL_TRIALS) * 100
        print(f"\n평가 완료. 평균 목표 행동({TARGET_ACTION}) 선택 비율: {bias:.2f}%")

        plt.figure(figsize=(10, 6))
        plt.hist(target_action_counts, bins=np.arange(0, TOTAL_TRIALS + 2) - 0.5, alpha=0.7, label=f'평균: {np.mean(target_action_counts):.1f}회')
        plt.axvline(TOTAL_TRIALS / 2, color='r', linestyle='--', label='무작위 선택 기준 (50회)')
        plt.axvline(np.mean(target_action_counts), color='b', linestyle='-', label='평균 선택 횟수')
        plt.title(f'Adversary 대결 후 목표 행동({TARGET_ACTION}) 선택 횟수 분포', fontsize=16)
        plt.xlabel(f'{TOTAL_TRIALS}회 중 목표 행동 선택 횟수', fontsize=12)
        plt.ylabel('에피소드 수', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

if __name__ == '__main__':
    experiment = Experiment()
    experiment.train()
    experiment.evaluate()
