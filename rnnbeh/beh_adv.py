import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import copy

# --- 하이퍼파라미터 설정 ---
# 실험 환경 설정
TOTAL_TRIALS = 100
REWARD_BUDGET_PER_ARM = 25
TARGET_ACTION = 0  # 목표 행동 (0: 왼쪽, 1: 오른쪽)

# QLearner 설정
LEARNER_ALPHA = 0.1  # 학습률
LEARNER_GAMMA = 0.9  # 할인율
LEARNER_EPSILON = 0.1 # 탐험 확률

# LearnerModel (RNN) 설정
MODEL_HIDDEN_SIZE = 8
MODEL_EPOCHS = 300
MODEL_LR = 0.005
MODEL_TRAINING_SAMPLES = 500

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

# --- 1. Learner Agent: Q-Learning 기반 Bandit 에이전트 ---
class QLearner:
    """간단한 Q-learning 에이전트. Adversary의 공격 대상."""
    def __init__(self, alpha, gamma, epsilon):
        self.q_table = np.zeros(2)  # [Q(action_0), Q(action_1)]
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
        old_value = self.q_table[self.last_action]
        # Bandit 문제는 상태가 없으므로, 다음 상태의 가치는 0으로 간주
        next_max = 0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[self.last_action] = new_value

# --- 2. Learner Model: Learner의 행동을 모방하는 RNN ---
class LearnerModel(nn.Module):
    """Learner 에이전트의 행동을 모방하는 GRU 모델."""
    def __init__(self, input_size, hidden_size, output_size):
        super(LearnerModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :])
        return F.softmax(out, dim=1), h

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

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

def train_learner_model(learner_model, q_learner):
    """QLearner의 데이터를 생성하고, 이를 이용해 LearnerModel을 훈련."""
    print("--- 1단계: LearnerModel 훈련 시작 ---")
    optimizer = optim.Adam(learner_model.parameters(), lr=MODEL_LR)
    criterion = nn.CrossEntropyLoss()

    # 데이터 생성
    inputs = []
    targets = []
    for _ in range(MODEL_TRAINING_SAMPLES):
        learner = copy.deepcopy(q_learner)
        hidden = learner_model.init_hidden()
        for _ in range(TOTAL_TRIALS):
            # 랜덤 환경에서 행동 및 학습
            action = learner.choose_action()
            reward = 1 if random.random() < 0.5 else 0
            
            # 이전 행동과 보상을 입력으로, 현재 행동을 타겟으로 설정
            # 입력 형태: [이전 행동, 이전 보상]
            # one-hot encoding: action 0 -> [1,0], action 1 -> [0,1]
            prev_action_vec = [1, 0] if learner.last_action == 0 else [0, 1]
            prev_reward = [reward]
            
            # 첫 trial은 이전 정보가 없으므로 스킵
            if learner.last_action is not None:
                model_input = torch.tensor([prev_action_vec + prev_reward], dtype=torch.float32)
                inputs.append(model_input)
                targets.append(action)

            learner.update(reward)

    # LearnerModel 훈련
    for epoch in range(MODEL_EPOCHS):
        total_loss = 0
        for i in range(len(inputs)):
            optimizer.zero_grad()
            hidden = learner_model.init_hidden() # 매 샘플마다 hidden state 초기화
            
            # 모델 입력 형태: (batch, seq_len, input_size)
            model_input = inputs[i].unsqueeze(0).to(device)
            target = torch.tensor([targets[i]], dtype=torch.long).to(device)
            
            output, hidden = learner_model(model_input, hidden)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{MODEL_EPOCHS}, Loss: {total_loss/len(inputs):.4f}")
    print("--- LearnerModel 훈련 완료 ---\n")


def train_adversary(adversary_policy_net, adversary_target_net, learner_model):
    """LearnerModel을 상대로 Adversary를 훈련."""
    print("--- 2단계: Adversary 훈련 시작 ---")
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
            # Adversary 행동: 0 (왼쪽 보상), 1 (오른쪽 보상)
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


    for i_episode in range(ADV_TRAINING_EPISODES):
        rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        learner_model_hidden = learner_model.init_hidden()
        
        # Adversary의 state: [learner_hidden_state, trial_num, r_left_0, r_left_1]
        state = torch.cat([
            learner_model_hidden.view(-1), 
            torch.tensor([0, rewards_left[0], rewards_left[1]], dtype=torch.float32, device=device)
        ]).unsqueeze(0)
        
        last_learner_action = 0
        last_reward_to_learner = 0

        for t in range(TOTAL_TRIALS):
            # 1. Adversary가 보상 할당 결정
            adv_action_tensor = select_adversary_action(state)
            adv_action = adv_action_tensor.item()
            
            # 예산 제약 조건 강제 적용
            # 만약 예산이 없으면, 반대편에 강제 할당
            if rewards_left[adv_action] == 0:
                adv_action = 1 - adv_action
            # 만약 남은 trial 수와 예산 수가 같으면, 남은 기간 동안 계속 보상 강제 할당
            trials_left = TOTAL_TRIALS - t
            if rewards_left[0] == trials_left: adv_action = 0
            if rewards_left[1] == trials_left: adv_action = 1

            reward_assignment = [0, 0]
            reward_assignment[adv_action] = 1
            rewards_left[adv_action] -= 1

            # 2. LearnerModel이 행동 결정
            prev_action_vec = [1, 0] if last_learner_action == 0 else [0, 1]
            model_input = torch.tensor([[prev_action_vec + [last_reward_to_learner]]], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action_probs, next_learner_model_hidden = learner_model(model_input, learner_model_hidden)
            
            learner_action = torch.multinomial(action_probs, 1).item()
            reward_to_learner = reward_assignment[learner_action]

            # 3. Adversary에 대한 보상 계산
            adversary_reward = 1.0 if learner_action == TARGET_ACTION else 0.0
            adversary_reward = torch.tensor([adversary_reward], device=device)
            
            # 4. 다음 상태 준비 및 메모리에 저장
            last_learner_action = learner_action
            last_reward_to_learner = reward_to_learner
            learner_model_hidden = next_learner_model_hidden

            if t == TOTAL_TRIALS - 1:
                next_state = None
            else:
                next_state = torch.cat([
                    learner_model_hidden.view(-1), 
                    torch.tensor([t+1, rewards_left[0], rewards_left[1]], dtype=torch.float32, device=device)
                ]).unsqueeze(0)

            memory.push(state, adv_action_tensor, next_state, adversary_reward)
            state = next_state

            # 5. DQN 모델 최적화
            optimize_model()

            # Target network 업데이트
            target_net_state_dict = adversary_target_net.state_dict()
            policy_net_state_dict = adversary_policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
            adversary_target_net.load_state_dict(target_net_state_dict)

            if state is None:
                break
        
        if (i_episode + 1) % 50 == 0:
            print(f"Episode {i_episode+1}/{ADV_TRAINING_EPISODES}")

    print("--- Adversary 훈련 완료 ---\n")


def evaluate(adversary, q_learner_template):
    """훈련된 Adversary를 실제 QLearner와 대결시켜 평가."""
    print("--- 3단계: 평가 시작 ---")
    target_action_counts = []

    for _ in range(EVALUATION_EPISODES):
        q_learner = copy.deepcopy(q_learner_template)
        rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        
        # 평가 시에는 LearnerModel이 아닌, Adversary의 state 추적용 '가짜' hidden state 사용
        # 실제 Learner의 내부 상태는 알 수 없기 때문 (논문의 closed-loop와 유사한 개념)
        # 하지만 이 예제에서는 QLearner의 내부 상태(q_table)를 Adversary의 state로 사용해 단순화
        
        count = 0
        for t in range(TOTAL_TRIALS):
            # Adversary의 state: [q_val_0, q_val_1, trial_num, r_left_0, r_left_1]
            # LearnerModel의 hidden state 대신 Q-table을 사용
            state = torch.tensor([q_learner.q_table[0], q_learner.q_table[1], t, rewards_left[0], rewards_left[1]], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                adv_action = adversary(state).max(1)[1].item()

            # 예산 제약 조건 강제 적용
            if rewards_left[adv_action] == 0: adv_action = 1 - adv_action
            trials_left = TOTAL_TRIALS - t
            if rewards_left[0] == trials_left: adv_action = 0
            if rewards_left[1] == trials_left: adv_action = 1
            
            reward_assignment = [0, 0]
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
    print(f"평가 완료. 평균 목표 행동 선택 비율: {bias:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.hist(target_action_counts, bins=np.arange(0, TOTAL_TRIALS + 2) - 0.5, alpha=0.7, label=f'평균: {np.mean(target_action_counts):.1f}회')
    plt.axvline(TOTAL_TRIALS / 2, color='r', linestyle='--', label='무작위 선택 기준 (50회)')
    plt.title(f'Adversary 대결 후 목표 행동({TARGET_ACTION}) 선택 횟수 분포', fontsize=16)
    plt.xlabel('100회 중 목표 행동 선택 횟수', fontsize=12)
    plt.ylabel('에피소드 수', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    # 1. 모델 및 에이전트 초기화
    # Learner
    base_q_learner = QLearner(alpha=LEARNER_ALPHA, gamma=LEARNER_GAMMA, epsilon=LEARNER_EPSILON)
    
    # LearnerModel (RNN)
    # 입력: [이전 행동(one-hot), 이전 보상] -> 3차원
    learner_model = LearnerModel(input_size=3, hidden_size=MODEL_HIDDEN_SIZE, output_size=2).to(device)

    # Adversary (DQN)
    # 상태: [learner_hidden, trial, r_left_0, r_left_1] -> MODEL_HIDDEN_SIZE + 3 차원
    # 평가 시에는 [q_val_0, q_val_1, trial, r_left_0, r_left_1] -> 5차원
    # 더 일반적인 사용을 위해 더 큰 차원으로 설정
    adv_state_size = MODEL_HIDDEN_SIZE + 3
    adv_action_size = 2 # 0: 왼쪽 보상 할당, 1: 오른쪽 보상 할당
    
    adversary_policy_net = AdversaryDQN(adv_state_size, adv_action_size).to(device)
    adversary_target_net = AdversaryDQN(adv_state_size, adv_action_size).to(device)
    adversary_target_net.load_state_dict(adversary_policy_net.state_dict())

    # 2. 훈련 단계 실행
    train_learner_model(learner_model, base_q_learner)
    train_adversary(adversary_policy_net, adversary_target_net, learner_model)

    # 3. 평가 단계 실행
    # 평가 시에는 Learner의 Q-table을 state로 사용하므로, DQN 입력 크기를 맞추어 새로 생성
    eval_adv_state_size = 5 # q_val_0, q_val_1, trial, r_left_0, r_left_1
    evaluation_adversary = AdversaryDQN(eval_adv_state_size, adv_action_size).to(device)
    
    # 훈련된 가중치에서 필요한 부분만 복사 (입력 레이어 가중치 모양이 다르므로)
    # 이 예제에서는 단순화를 위해 훈련된 policy net을 그대로 사용. 
    # 입력 차원이 다르므로 실제로는 호환되지 않지만, 여기서는 개념 증명을 위해
    # 훈련 단계와 평가 단계의 Adversary state 정의를 통일해야 함.
    # 아래는 Q-table을 hidden state 대신 사용하는 방식으로 재정의하여 훈련/평가 통일
    
    # --- 재정의 및 재훈련 (훈련/평가 통일 버전) ---
    print("\n--- 훈련/평가 State 통일 후 재실행 ---\n")
    adv_state_size = 5 # [q_val_0, q_val_1, trial, r_left_0, r_left_1]
    adversary_policy_net = AdversaryDQN(adv_state_size, adv_action_size).to(device)
    adversary_target_net = AdversaryDQN(adv_state_size, adv_action_size).to(device)
    adversary_target_net.load_state_dict(adversary_policy_net.state_dict())

    # Adversary 훈련 (QLearner를 직접 시뮬레이션)
    # 이 방식은 LearnerModel을 생략하고, 실제 Learner의 내부 상태를 안다고 가정
    # 논문의 아이디어를 더 직접적으로 구현
    print("--- 2-2단계: QLearner 직접 시뮬레이션으로 Adversary 훈련 시작 ---")
    optimizer = optim.Adam(adversary_policy_net.parameters(), lr=ADV_LR)
    memory = ReplayBuffer(10000)
    steps_done = 0
    
    for i_episode in range(ADV_TRAINING_EPISODES):
        q_learner = copy.deepcopy(base_q_learner)
        rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        
        for t in range(TOTAL_TRIALS):
            state = torch.tensor([q_learner.q_table[0], q_learner.q_table[1], t, rewards_left[0], rewards_left[1]], dtype=torch.float32, device=device).unsqueeze(0)
            
            # Adversary 행동 선택 (위의 함수 재사용)
            adv_action_tensor = select_adversary_action(state)
            adv_action = adv_action_tensor.item()

            if rewards_left[adv_action] == 0: adv_action = 1 - adv_action
            trials_left = TOTAL_TRIALS - t
            if rewards_left[0] == trials_left: adv_action = 0
            if rewards_left[1] == trials_left: adv_action = 1

            reward_assignment = [0, 0]
            reward_assignment[adv_action] = 1
            rewards_left[adv_action] -= 1

            learner_action = q_learner.choose_action()
            reward_to_learner = reward_assignment[learner_action]
            q_learner.update(reward_to_learner)
            
            adversary_reward = 1.0 if learner_action == TARGET_ACTION else 0.0
            adversary_reward = torch.tensor([adversary_reward], device=device)

            if t == TOTAL_TRIALS - 1:
                next_state = None
            else:
                next_state = torch.tensor([q_learner.q_table[0], q_learner.q_table[1], t+1, rewards_left[0], rewards_left[1]], dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, adv_action_tensor, next_state, adversary_reward)
            
            optimize_model()

            target_net_state_dict = adversary_target_net.state_dict()
            policy_net_state_dict = adversary_policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
            adversary_target_net.load_state_dict(target_net_state_dict)

            if next_state is None:
                break
        
        if (i_episode + 1) % 50 == 0:
            print(f"Episode {i_episode+1}/{ADV_TRAINING_EPISODES}")
    print("--- Adversary 훈련 완료 ---\n")

    # 최종 평가
    evaluate(adversary_policy_net, base_q_learner)
