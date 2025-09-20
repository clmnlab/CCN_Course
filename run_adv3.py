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

# BehavioralLearner (RNN) 설정
BH_MODEL_INPUT_SIZE = 2 # [action, reward]
BH_MODEL_HIDDEN_SIZE = 8
BH_MODEL_EPOCHS = 50
BH_MODEL_LR = 0.005

# Adversary (DQN) 설정
# State: learner_state(2) + trial_num(1) + rewards_left(2) = 5
ADV_STATE_SIZE = 5
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

# --- 1. Learner Agent 들 ---
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

    def get_observable_state(self):
        return self.q_table
        
    def reset(self):
        self.q_table = np.zeros(2, dtype=np.float32)
        self.last_action = None


class BehavioralLearner(nn.Module):
    """데이터로부터 행동 패턴을 학습하는 GRU 기반 에이전트."""
    def __init__(self, input_size, hidden_size, output_size):
        super(BehavioralLearner, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.last_action = None
        self.to(device)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :])
        return F.softmax(out, dim=1), h

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def choose_action(self):
        # 현재 hidden state를 기반으로 행동 확률 계산
        # 입력은 마지막 행동과 보상. 첫 턴에는 [0,0]을 사용.
        prev_action = self.last_action if self.last_action is not None else 0
        # update가 먼저 호출되지 않으므로, 보상은 0으로 가정
        prev_reward = 0 
        
        with torch.no_grad():
            input_tensor = torch.tensor([[prev_action, prev_reward]], dtype=torch.float32, device=device).unsqueeze(1)
            action_probs, self.hidden = self.forward(input_tensor, self.hidden)
        
        action = torch.multinomial(action_probs, 1).item()
        self.last_action = action
        return action

    def update(self, reward):
        # 실제 학습이 아닌, 다음 행동 예측을 위한 hidden state 업데이트
        if self.last_action is None: return
        with torch.no_grad():
            input_tensor = torch.tensor([[self.last_action, reward]], dtype=torch.float32, device=device).unsqueeze(1)
            _, self.hidden = self.forward(input_tensor, self.hidden)

    def get_observable_state(self):
        # 현재 hidden state에서 예측되는 행동 확률을 반환
        with torch.no_grad():
            action_probs, _ = self.forward(torch.zeros(1, 1, BH_MODEL_INPUT_SIZE, device=device), self.hidden)
        return action_probs.cpu().numpy().flatten()

    def reset(self):
        self.hidden = self.init_hidden()
        self.last_action = None

    def train_from_data(self, data):
        print("\n--- BehavioralLearner 사전 학습 시작 ---")
        optimizer = optim.Adam(self.parameters(), lr=BH_MODEL_LR)
        criterion = nn.CrossEntropyLoss()
        
        # 데이터 전처리: (입력 시퀀스, 타겟 행동)
        inputs, targets = [], []
        for episode in data:
            for t in range(len(episode) - 1):
                action, reward = episode[t]
                next_action, _ = episode[t+1]
                inputs.append([action, reward])
                targets.append(next_action)

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
        targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

        for epoch in tqdm(range(BH_MODEL_EPOCHS), desc="BehavioralLearner Training"):
            # 매 에포크마다 hidden state 초기화
            hidden = self.init_hidden(batch_size=len(inputs_tensor))
            # 입력 형태: (batch, seq_len, input_size) -> (batch, 1, 2)
            output, _ = self.forward(inputs_tensor.unsqueeze(1), hidden)
            loss = criterion(output, targets_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"--- BehavioralLearner 사전 학습 완료 (Final Loss: {loss.item():.4f}) ---")


# --- 2. Adversary Agent: 보상 전략을 학습하는 DQN 에이전트 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
        return F.relu(self.layer3(F.relu(self.layer2(F.relu(self.layer1(x))))))

class AdversaryAgent:
    def __init__(self):
        self.policy_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ADV_LR)
        self.memory = ReplayBuffer(ADV_MEMORY_CAPACITY)
        self.steps_done = 0

    def select_action(self, state, is_training=True):
        if is_training:
            sample = random.random()
            eps_threshold = ADV_EPS_END + (ADV_EPS_START - ADV_EPS_END) * np.exp(-1. * self.steps_done / ADV_EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(ADV_ACTION_SIZE)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

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
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
        self.target_net.load_state_dict(target_net_state_dict)

# --- 3. 실험 전체를 관리하는 클래스 ---
class Experiment:
    def __init__(self, learner_template):
        print(f"Using device: {device}")
        self.learner_template = learner_template
        self.adversary_agent = AdversaryAgent()

    def run(self, is_training=True):
        num_episodes = ADV_TRAINING_EPISODES if is_training else EVALUATION_EPISODES
        desc = "Adversary Training" if is_training else "Evaluation"
        print(f"\n--- {desc} 시작 (Learner: {self.learner_template.__class__.__name__}) ---")
        
        target_action_counts = []
        for _ in tqdm(range(num_episodes), desc=desc):
            learner = copy.deepcopy(self.learner_template)
            learner.reset()
            rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
            count = 0

            for t in range(TOTAL_TRIALS):
                learner_state = learner.get_observable_state()
                state = torch.tensor(
                    np.concatenate([learner_state, [t, rewards_left[0], rewards_left[1]]]),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)

                adv_action_tensor = self.adversary_agent.select_action(state, is_training=is_training)
                adv_action = adv_action_tensor.item()
                
                if rewards_left[adv_action] <= 0: adv_action = 1 - adv_action
                
                reward_assignment = [0, 0]
                if rewards_left[adv_action] > 0:
                    reward_assignment[adv_action] = 1
                    rewards_left[adv_action] -= 1

                learner_action = learner.choose_action()
                reward_to_learner = reward_assignment[learner_action]
                learner.update(reward_to_learner)

                if learner_action == TARGET_ACTION: count += 1
                
                if is_training:
                    adversary_reward = 1.0 if learner_action == TARGET_ACTION else 0.0
                    adversary_reward_tensor = torch.tensor([adversary_reward], device=device)
                    
                    next_learner_state = learner.get_observable_state()
                    next_state = None
                    if t < TOTAL_TRIALS - 1:
                        next_state = torch.tensor(
                            np.concatenate([next_learner_state, [t + 1, rewards_left[0], rewards_left[1]]]),
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)

                    self.adversary_agent.memory.push(state, adv_action_tensor, next_state, adversary_reward_tensor)
                    self.adversary_agent.optimize_model()
                    self.adversary_agent.update_target_net()

            target_action_counts.append(count)
        
        if not is_training:
            self.plot_results(target_action_counts)

    def plot_results(self, target_action_counts):
        bias = (np.mean(target_action_counts) / TOTAL_TRIALS) * 100
        print(f"\n평가 완료. 평균 목표 행동({TARGET_ACTION}) 선택 비율: {bias:.2f}%")
        plt.figure(figsize=(10, 6))
        plt.hist(target_action_counts, bins=np.arange(0, TOTAL_TRIALS + 2) - 0.5, alpha=0.7, label=f'평균: {np.mean(target_action_counts):.1f}회')
        plt.axvline(TOTAL_TRIALS / 2, color='r', linestyle='--', label='무작위 선택 기준 (50회)')
        plt.axvline(np.mean(target_action_counts), color='b', linestyle='-', label='평균 선택 횟수')
        plt.title(f'Adversary 대결 후 목표 행동({TARGET_ACTION}) 선택 횟수 분포\n(Learner: {self.learner_template.__class__.__name__})', fontsize=16)
        plt.xlabel(f'{TOTAL_TRIALS}회 중 목표 행동 선택 횟수', fontsize=12)
        plt.ylabel('에피소드 수', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

# --- 4. 메인 실행 로직 ---
def generate_dummy_behavioral_data(num_episodes, episode_length):
    """BehavioralLearner 학습을 위한 가상 데이터 생성."""
    data = []
    for _ in range(num_episodes):
        episode = []
        # 예시: 초반에는 0번을 선호하다가, 보상을 못 받으면 1번으로 바꾸는 경향
        prefer_action = 0
        for t in range(episode_length):
            action = prefer_action if random.random() < 0.8 else 1 - prefer_action
            reward = 1 if (action == 1 and t > episode_length / 2) else 0
            if action == prefer_action and reward == 0 and random.random() < 0.2:
                prefer_action = 1 - prefer_action
            episode.append((action, reward))
        data.append(episode)
    return data

if __name__ == '__main__':
    # --- 시나리오 1: QLearner를 상대로 Adversary 훈련 및 평가 ---
    q_learner_template = QLearner(alpha=LEARNER_ALPHA, gamma=LEARNER_GAMMA, epsilon=LEARNER_EPSILON)
    experiment_q_learner = Experiment(learner_template=q_learner_template)
    experiment_q_learner.run(is_training=True)
    experiment_q_learner.run(is_training=False)

    # --- 시나리오 2: BehavioralLearner를 상대로 Adversary 훈련 및 평가 ---
    # 1. 가상 피험자 데이터 생성
    dummy_data = generate_dummy_behavioral_data(num_episodes=50, episode_length=TOTAL_TRIALS)
    
    # 2. BehavioralLearner 생성 및 사전 학습
    behavioral_learner_template = BehavioralLearner(
        input_size=BH_MODEL_INPUT_SIZE, 
        hidden_size=BH_MODEL_HIDDEN_SIZE, 
        output_size=ADV_ACTION_SIZE
    )
    behavioral_learner_template.train_from_data(dummy_data)
    
    # 3. 실험 진행
    experiment_bh_learner = Experiment(learner_template=behavioral_learner_template)
    experiment_bh_learner.run(is_training=True)
    experiment_bh_learner.run(is_training=False)

