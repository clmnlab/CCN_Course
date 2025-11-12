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
import statistics
from torch.utils.data import TensorDataset, DataLoader
# --- 하이퍼파라미터 설정 ---
# 실험 환경 설정
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

# BehavioralLearner (RNN) 설정
BH_MODEL_INPUT_SIZE = 2 # [action, reward]
BH_MODEL_HIDDEN_SIZE = 8
BH_MODEL_EPOCHS = 500
BH_MODEL_LR = 0.005

# Adversary (DQN) 설정
# State: learner_state(2) + trial_num(1) + rewards_left(2) = 5
ADV_STATE_SIZE = 5
ADV_ACTION_SIZE = 4 # 0: 왼쪽에 보상, 1: 오른쪽에 보상, 2: 양쪽에 보상, 3: 아무데도 보상 없음
ADV_HIDDEN_SIZE = 128
ADV_BATCH_SIZE = 64
ADV_GAMMA = 0.99
ADV_EPS_START = 0.9
ADV_EPS_END = 0.05
ADV_EPS_DECAY = 1000
ADV_TAU = 0.005
ADV_LR = 1e-4
ADV_TRAINING_EPISODES = 5000
ADV_MEMORY_CAPACITY = 10000
REWARD_SCALE_FACTOR = 1e4

# 평가 설정
EVALUATION_EPISODES = 100

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions
def choose_randomly():
  """ Return either of the alternatives with 50% chance """
  if random.random() < 0.5:
      choice = TARGET_ACTION
  else:
      choice = ALT_ACTION
  return choice


def other_alternative(a):
  """
  Given alternative a return the other alternative:
    TARGET_ACTION --> ALT_ACTION
    ALT_ACTION --> TARGET_ACTION
  """
  if a is TARGET_ACTION:
      return ALT_ACTION
  if a is ALT_ACTION:
      return TARGET_ACTION
  

# --- 1. Learner Agent 들 ---
class QLearner:
    """간단한 Q-learning 에이전트. Adversary의 공격 대상."""
    def __init__(self, alpha, gamma, epsilon):
        self.q_table = 0.25*np.ones(2, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_action = None

    def select_action(self):
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
    def __init__(self, input_size, hidden_size, output_size, n_epochs=100, lr=1e-3):
        super(BehavioralLearner, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.last_action = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.to(device)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out) # out shape: (batch, seq_len, output_size)
        return out, h
        # return F.softmax(out, dim=2), h

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def select_action(self):
        # (이전 코드와 동일)
        prev_action = self.last_action if self.last_action is not None else 0
        prev_reward = 0 
        
        with torch.no_grad():
            input_tensor = torch.tensor([[prev_action, prev_reward]], dtype=torch.float32, device=device).unsqueeze(1)
            logits, self.hidden = self.forward(input_tensor, self.hidden)
        
        # .squeeze(0)으로 (1, 1, 2) -> (1, 2)
        # .multinomial(1) -> (1, 1) -> .item()
        probs = F.softmax(logits.squeeze(0), dim=1)  # (1, output_size)
        action = torch.multinomial(probs, 1).item()
        self.last_action = action
        return action

    def update(self, reward):
        # (이전 코드와 동일)
        if self.last_action is None: return
        with torch.no_grad():
            input_tensor = torch.tensor([[self.last_action, reward]], dtype=torch.float32, device=device).unsqueeze(1)
            _, self.hidden = self.forward(input_tensor, self.hidden)

    def get_observable_state(self):
        # (이전 코드와 동일)
        with torch.no_grad():
            # (1, 1, 2) -> (2,)
            action_probs, _ = self.forward(torch.zeros(1, 1, BH_MODEL_INPUT_SIZE, device=device), self.hidden)
        return action_probs.squeeze().cpu().numpy().flatten()

    def reset(self):
        self.hidden = self.init_hidden()
        self.last_action = None

    # --- [신규] 데이터 전처리 헬퍼 함수 ---
    def _preprocess_data(self, data):
        """원시 데이터 리스트를 (입력 텐서, 타겟 텐서)로 변환합니다."""
        all_inputs = []
        all_targets = []
        expected_seq_len = len(data[0])-1  # 100 스텝 -> 99개의 (입력, 타겟) 쌍

        for episode in data:
            if len(episode) != 100: 
                continue 

            inputs, targets = [], []
            for t in range(expected_seq_len):
                action, reward = episode[t]
                next_action, _ = episode[t+1]
                inputs.append([action, reward])
                targets.append(next_action)
            
            all_inputs.append(inputs)
            all_targets.append(targets)

        if not all_inputs:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.long)

        inputs_tensor = torch.tensor(all_inputs, dtype=torch.float32) 
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)
        
        return inputs_tensor, targets_tensor

    # --- [수정] train_from_data (전처리 로직 분리) ---
    def train_from_data(self, data, batch_size=32):
        print(f"\n--- BehavioralLearner 사전 학습 시작 (Batch Size: {batch_size}) ---")
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # 1. 전처리 헬퍼 함수 호출
        inputs_tensor, targets_tensor = self._preprocess_data(data)
        
        if inputs_tensor.nelement() == 0:
            print("오류: 학습 데이터가 없습니다.")
            return

        # 2. TensorDataset과 DataLoader 생성
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train() # 모델을 학습 모드로 설정

        for epoch in tqdm(range(self.n_epochs), desc="BehavioralLearner Training"):
            total_loss = 0
            total_items = 0
            
            for inputs_batch, targets_batch in loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                current_batch_size = inputs_batch.size(0)

                hidden = self.init_hidden(batch_size=current_batch_size)
                optimizer.zero_grad()
                output_probs, _ = self.forward(inputs_batch, hidden)
                
                loss = criterion(
                    output_probs.view(-1, output_probs.size(2)), # (batch*99, 2)
                    targets_batch.view(-1)                        # (batch*99)
                )

                loss.backward()
                optimizer.step()

                num_timesteps_in_batch = targets_batch.numel() 
                total_loss += loss.item() * num_timesteps_in_batch
                total_items += num_timesteps_in_batch

            avg_loss = total_loss / total_items if total_items > 0 else 0
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Avg Loss: {avg_loss:.4f}")

        print(f"--- BehavioralLearner 사전 학습 완료 (Final Loss: {avg_loss:.4f}) ---")
        self.eval()

    # --- [신규] evaluate (전처리 로직 분리) ---
    def evaluate(self, test_data, batch_size=32):
        print(f"\n--- BehavioralLearner 평가 시작 (Batch Size: {batch_size}) ---")
        self.eval() 

        # 1. 전처리 헬퍼 함수 호출
        inputs_tensor, targets_tensor = self._preprocess_data(test_data)

        if inputs_tensor.nelement() == 0:
            print("오류: 평가 데이터가 없습니다.")
            return 0.0
        
        # 2. DataLoader 생성 (평가 시에는 shuffle=False)
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_correct = 0
        total_items = 0
        all_predictions = [] # 모든 배치의 예측을 저장하기 위한 리스트
        # 3. 그래디언트 계산 비활성화
        with torch.no_grad():
            for inputs_batch, targets_batch in loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                current_batch_size = inputs_batch.size(0)
                hidden = self.init_hidden(batch_size=current_batch_size)
                
                output_probs, _ = self.forward(inputs_batch, hidden)
                
                # 4. Accuracy 계산
                predictions = torch.argmax(output_probs, dim=2)
                correct_predictions = (predictions == targets_batch).sum().item()
                total_items_in_batch = targets_batch.numel()
                
                total_correct += correct_predictions
                total_items += total_items_in_batch
                all_predictions.append(predictions.cpu())
        # 5. 최종 정확도 계산 및 출력
        accuracy = (total_correct / total_items) * 100 if total_items > 0 else 0
        print(f"--- 평가 완료: Accuracy = {accuracy:.2f}% ({total_correct} / {total_items} 맞음) ---")
        final_predictions = torch.cat(all_predictions, dim=0)

        return accuracy, final_predictions
    

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
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)   # 마지막은 raw Q-values
    # def forward(self, x):
    #     return F.relu(self.layer3(F.relu(self.layer2(F.relu(self.layer1(x))))))


class AdversaryAgent:
    def __init__(self, learner):
        self.learner = learner # target learner to attack
        self.policy_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net = AdversaryDQN(ADV_STATE_SIZE, ADV_ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ADV_LR)
        self.memory = ReplayBuffer(ADV_MEMORY_CAPACITY)
        self.rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        self.reward_assignment = [0, 0]
        self.steps_done = 0

    def reset(self):
        learner_state = self.learner.get_observable_state()
        norm_t = 0
        norm_rewards_left = [1, 1]
        state = torch.tensor(
            np.concatenate([learner_state, [norm_t], norm_rewards_left]),
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        return state
    
    def select_action(self, state, is_training=True):
        if is_training:
            sample = random.random()
            eps_threshold = ADV_EPS_END + (ADV_EPS_START - ADV_EPS_END) * np.exp(-1. * self.steps_done / ADV_EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    adv_action_tensor = self.policy_net(state).max(1)[1].view(1, 1)
            else:
                adv_action_tensor= torch.tensor([[random.randrange(ADV_ACTION_SIZE)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                adv_action_tensor = self.policy_net(state).max(1)[1].view(1, 1)
            
        adv_action_raw = adv_action_tensor.item() # 0, 1, 2, 3 중 하나                
        # --- [핵심 수정] 4가지 행동을 '의도'로 번역 ---
        # adv_action_raw의 의미: # 0: (R:0, L:0) # 1: (R:1, L:0) # 2: (R:0, L:1) # 3: (R:1, L:1)
        # (참고: TARGET_ACTION = 0 (Left), ALT_ACTION = 1 (Right))
        reward_intent = [0, 0]
        if adv_action_raw == 1:
            reward_intent[TARGET_ACTION] = 1
        elif adv_action_raw == 2:
            reward_intent[ALT_ACTION] = 1
        elif adv_action_raw == 3:
            reward_intent[TARGET_ACTION] = 1
            reward_intent[ALT_ACTION] = 1
        # if rewards_left[adv_action] <= 0: adv_action = 1 - adv_action
        
        # --- 예산 제약(Budget)을 '현실'로 적용, 예산이 없으면 보상을 0으로 강제합니다.        
        # 0번 팔(Left)에 보상을 주려 했고 & 예산이 있는가?
        if reward_intent[TARGET_ACTION] == 1 and self.rewards_left[TARGET_ACTION] > 0:
            self.reward_assignment[TARGET_ACTION] = 1
            self.rewards_left[TARGET_ACTION] -= 1
        
        # 1번 팔(Right)에 보상을 주려 했고 & 예산이 있는가?
        if reward_intent[ALT_ACTION] == 1 and self.rewards_left[ALT_ACTION] > 0:
            self.reward_assignment[ALT_ACTION] = 1
            self.rewards_left[ALT_ACTION] -= 1

        adv_action = 0  # 기본값: [0, 0] (보상 없음)
        if self.reward_assignment[TARGET_ACTION] == 1 and self.reward_assignment[ALT_ACTION] == 0:
            adv_action = 1  # [1, 0]
        elif self.reward_assignment[TARGET_ACTION] == 0 and self.reward_assignment[ALT_ACTION] == 1:
            adv_action = 2  # [0, 1]
        elif self.reward_assignment[TARGET_ACTION] == 1 and self.reward_assignment[ALT_ACTION] == 1:
            adv_action = 3  # [1, 1]

        return adv_action    
    # def step(self, action):


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
        return loss.item()
    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
        self.target_net.load_state_dict(target_net_state_dict)


class KBandit:
    def __init__(self, n_arms, reward_type='binary', stationary=True):
        self.n_arms = n_arms
        self.reward_type = reward_type
        self.stationary = stationary
    def reset(self):    
        if self.reward_type == "continuous":
            self.action_values = np.random.normal(size=self.n_arms)
            self.reward_func = self._continuous_reward
        elif self.reward_type == "binary":
            self.action_values = np.random.rand(self.n_arms)
            self.reward_func = self._binary_reward
        else:
            raise ValueError("Reward type not recognized")
    
    def _binary_reward(self, action):
        # Return a binary reward with probability equal to the reward_probs for that action
        return np.random.choice([0, 1], p=[1-self.action_values[action], self.action_values[action]])
    
    def step(self, action):
        # Take an action and return the corresponding reward
        reward = self.reward_func(action)
        self.optimal = np.argmax(self.action_values)
        if not self.stationary:
            # Update action values and reward probabilities using random walks
            self.action_values += np.random.normal(0, 0.01, self.n_arms)
            if self.reward_type=='binary':
                self.action_values = np.clip(self.action_values, 0, 1)
        return reward
class RandEnv:
    def __init__(self, n_arms):
        self.n_arms = n_arms
    def reset(self):    
        pass
    def step(self, action):
        reward = np.random.choice([0, 1], p=[0.5, 0.5])
        return reward
        
def generate_behavioral_data(agent, env, num_episodes=10000, episode_length=100, seed=None):
    """
    실제 에이전트(Q-learning 또는 CatieAgent)를 사용하여 행동 데이터를 생성.
    
    Args:
        agent_class: QLearner 또는 CatieAgent 클래스
        num_episodes (int): 에피소드 수
        episode_length (int): 각 에피소드 길이
        seed (int or None): 재현 가능성을 위한 랜덤 시드

    Returns:
        data (list): [[(action, reward), ...], ...] 형태의 BehavioralLearner 학습용 데이터
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = []

    for _ in range(num_episodes):
        # 에이전트 초기화
        episode = []
        # rewards_left = [REWARD_BUDGET_PER_ARM, REWARD_BUDGET_PER_ARM]
        env.reset()
        for _ in range(episode_length):
            # 행동 선택
            action = agent.select_action()
            reward = env.step(action)
            # # 단순한 보상 규칙: TARGET_ACTION에 가까운 행동에 보상 집중
            # # reward = 1 if random.random() < 0.5 else 0
            # if rewards_left[action] > 0 and random.random() < 0.5:
            #     reward = 1
            #     rewards_left[action] -= 1
            # else:
            #     reward = 0

            # 보상 반영
            agent.update(reward)
            episode.append((action, reward))

        data.append(episode)

    # print(f"✅ {agent_class.__name__} 기반 Behavioral 데이터 {num_episodes}개 생성 완료.")
    return data


# --- 3. 실험 전체를 관리하는 클래스 ---
# class Experiment:
   