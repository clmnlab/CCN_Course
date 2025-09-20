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

# LearnerModel (RNN) 설정
MODEL_HIDDEN_SIZE = 8
MODEL_EPOCHS = 50 # 효율 개선으로 에포크 수 줄임
MODEL_LR = 0.005
MODEL_TRAINING_SAMPLES = 1000
MODEL_BATCH_SIZE = 64

# Adversary (DQN) 설정
ADV_HIDDEN_SIZE = 128
ADV_BATCH_SIZE = 64
ADV_GAMMA = 0.99
ADV_EPS_START = 0.9
ADV_EPS_END = 0.05
ADV_EPS_DECAY = 1000
ADV_TAU = 0.005
ADV_LR = 1e-4
ADV_TRAINING_EPISODES = 500

# 평가 설정
EVALUATION_EPISODES = 100

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# --- 2. Learner Model: Learner의 행동을 모방하는 RNN ---
# 이 예제에서는 LearnerModel을 직접 사용하지 않고, QLearner를 바로 공격합니다.
# 논문의 개념을 더 단순화하여, Adversary가 QLearner의 내부 상태(Q-table)를 직접 보고
# 보상 전략을 학습하는 것으로 문제를 재구성했습니다.

# --- 3. Adversary Agent: 보상 전략을 학습하는 DQN 에이전트 ---
ReplayMemory = namedtuple('ReplayMemory', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    """DQN을 위한 리플레이 버퍼."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(ReplayMemory(*args))

    def sample(self, batch_size):
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

# --- 훈련 및 평가 로직 ---
def train_adversary(adversary_policy_net, adversary_target_net, q_learner_template):
    """QLearner를 상대로 Adversary를 훈련."""
    print("--- Adversary 훈련 시작 ---")
    optimizer = optim.Adam(adversary_policy_net.parameters(), lr=ADV_LR)
    memory = ReplayBuffer(10000)
    steps_done = 0

    def select_adversary_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = ADV_EPS_END + (ADV_EPS_START - ADV_EPS_END) * \
            np.exp(-1. * steps_done / ADV_EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return adversary_policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < ADV_BATCH_SIZE:
            return
        transitions = memory.sample(ADV_BATCH_SIZE)
        batch = ReplayMemory(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = adversary_policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(ADV_BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = adversary_target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * ADV_GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(adversary_policy_net.parameters(), 100)
        optimizer.step()

    for i_episode in tqdm(range(ADV_TRAINING_EPISODES), desc="Adversary Training"):
        q_learner = copy.deepcopy(q_learner_template)
        rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        
        for t in range(TOTAL_TRIALS):
            # Adversary의 state: [q_val_0, q_val_1, trial_num, r_left_0, r_left_1]
            state = torch.tensor(
                np.concatenate([q_learner.q_table, [t, rewards_left[0], rewards_left[1]]]),
                dtype=torch.float32, device=device
            ).unsqueeze(0)

            # 1. Adversary가 보상 할당 결정
            adv_action_tensor = select_adversary_action(state)
            adv_action = adv_action_tensor.item()
            
            # 예산 제약 조건 적용
            if rewards_left[adv_action] <= 0:
                adv_action = 1 - adv_action
            
            reward_assignment = [0, 0]
            if rewards_left[adv_action] > 0:
                reward_assignment[adv_action] = 1
                rewards_left[adv_action] -= 1

            # 2. QLearner가 행동하고 학습
            learner_action = q_learner.choose_action()
            reward_to_learner = reward_assignment[learner_action]
            q_learner.update(reward_to_learner)

            # 3. Adversary에 대한 보상 계산
            adversary_reward = 1.0 if learner_action == TARGET_ACTION else 0.0
            adversary_reward = torch.tensor([adversary_reward], device=device)
            
            # 4. 다음 상태 준비 및 메모리에 저장
            if t == TOTAL_TRIALS - 1:
                next_state = None
            else:
                next_state = torch.tensor(
                    np.concatenate([q_learner.q_table, [t + 1, rewards_left[0], rewards_left[1]]]),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)

            memory.push(state, adv_action_tensor, next_state, adversary_reward)

            # 5. DQN 모델 최적화
            optimize_model()

            # Target network 업데이트
            target_net_state_dict = adversary_target_net.state_dict()
            policy_net_state_dict = adversary_policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
            adversary_target_net.load_state_dict(target_net_state_dict)

    print("--- Adversary 훈련 완료 ---\n")

def evaluate(adversary, q_learner_template):
    """훈련된 Adversary를 실제 QLearner와 대결시켜 평가."""
    print("--- 평가 시작 ---")
    target_action_counts = []

    for _ in tqdm(range(EVALUATION_EPISODES), desc="Evaluation"):
        q_learner = copy.deepcopy(q_learner_template)
        rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        count = 0

        for t in range(TOTAL_TRIALS):
            # Adversary의 state: [q_val_0, q_val_1, trial_num, r_left_0, r_left_1]
            state = torch.tensor(
                np.concatenate([q_learner.q_table, [t, rewards_left[0], rewards_left[1]]]),
                dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                # 평가 시에는 탐험(epsilon-greedy) 없이 최적의 행동만 선택
                adv_action = adversary(state).max(1)[1].item()

            # 예산 제약 조건 적용
            if rewards_left[adv_action] <= 0:
                adv_action = 1 - adv_action
            
            reward_assignment = [0, 0]
            if rewards_left[adv_action] > 0:
                reward_assignment[adv_action] = 1
                rewards_left[adv_action] -= 1

            # QLearner가 행동하고 학습
            learner_action = q_learner.choose_action()
            reward_to_learner = reward_assignment[learner_action]
            q_learner.update(reward_to_learner)

            if learner_action == TARGET_ACTION:
                count += 1
        
        target_action_counts.append(count)

    # 결과 시각화
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
    # 1. 기본 QLearner 템플릿 생성
    base_q_learner = QLearner(alpha=LEARNER_ALPHA, gamma=LEARNER_GAMMA, epsilon=LEARNER_EPSILON)

    # 2. Adversary DQN 네트워크 생성
    # State: Q-table(2) + trial_num(1) + rewards_left(2) = 5
    adv_state_size = 5 
    adv_action_size = 2 # 0: 왼쪽 보상 할당, 1: 오른쪽 보상 할당
    
    adversary_policy_net = AdversaryDQN(adv_state_size, adv_action_size).to(device)
    adversary_target_net = AdversaryDQN(adv_state_size, adv_action_size).to(device)
    adversary_target_net.load_state_dict(adversary_policy_net.state_dict())

    # 3. 훈련 및 평가 실행
    train_adversary(adversary_policy_net, adversary_target_net, base_q_learner)
    evaluate(adversary_policy_net, base_q_learner)
