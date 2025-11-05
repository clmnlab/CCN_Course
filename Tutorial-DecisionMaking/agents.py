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
        self.q_table = np.zeros(2, dtype=np.float32)
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



class CatieAgent:
    def __init__(self, K=2, tau=0.2, phi=0.1, epsilon=0.1, seed=None):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.tau = float(tau)
        self.phi = float(phi)
        self.epsilon = float(epsilon)
        self.K = int(K)
        # choose a k in [0, K] to represent the current contingency depth used internally
        self.k = random.randint(0, max(0, self.K))
        self.trial_number = 0
        self.choices = []
        self.surprises = []
        self.outcomes = []
        self.outcomes_biased = []
        self.outcomes_anti_biased = []

    def get_p_explore(self):
        if len(self.surprises) == 0:
            p_explore = self.epsilon * 1/3.0
        else:
            p_explore = self.epsilon * (1.0 + self.surprises[-1] + np.mean(self.surprises)) / 3.0
        return max(0.0, min(1.0, p_explore))

    def get_trend(self):
        if (len(self.choices) < 2 or self.choices[-1] != self.choices[-2]):
            return "INVALID"
        elif self.outcomes[-1] > self.outcomes[-2]:
            return "POSITIVE"
        else:
            return "NON_POSITIVE"

    def __contingent_average_specific_contingency(self, k, alternative, contingency):
        ca_outcomes = []
        if k == 0:
            return []
        for i in range(len(self.outcomes) - k):
            if (self.outcomes[i:i + k] == contingency) and (self.choices[i + k] == alternative):
                ca_outcomes.append(self.outcomes[i + k])
        return ca_outcomes

    def __contingent_average(self, k, alternative):
        if k <= 0:
            ca_outcomes = self.outcomes_biased if alternative == TARGET_ACTION else self.outcomes_anti_biased
        else:
            if k > len(self.outcomes):
                ca_outcomes = []
            else:
                current_contingency = self.outcomes[-k:]
                ca_outcomes = self.__contingent_average_specific_contingency(k, alternative, current_contingency)
            if not ca_outcomes:
                alternative_choice_indices = [i for i in range((k-1), len(self.choices)) if self.choices[i] == alternative]
                all_k_contingencies = [self.outcomes[i:i+k] for i in alternative_choice_indices if i+k <= len(self.outcomes)]
                if alternative_choice_indices and all_k_contingencies:
                    ca_outcomes = self.__contingent_average_specific_contingency(k, alternative, random.choice(all_k_contingencies))
                else:
                    ca_outcomes = self.__contingent_average(k - 1, alternative)
        if not ca_outcomes:
            return None
        else:
            return np.mean(ca_outcomes)

    def get_contingent_average(self):
        return (self.__contingent_average(self.k, TARGET_ACTION),
                self.__contingent_average(self.k, ALT_ACTION))

    def update(self, choice, outcome):
        self.choices.append(choice)
        if choice == TARGET_ACTION:
            obs_list = self.outcomes_biased
        else:
            obs_list = self.outcomes_anti_biased
        obs_sd = 0 if len(obs_list) < 2 else np.std(obs_list)
        if choice == TARGET_ACTION:
            self.outcomes_biased.append(outcome)
        else:
            self.outcomes_anti_biased.append(outcome)
        self.outcomes.append(outcome)
        if obs_sd > 0:
            exp_t_i = self.__contingent_average(self.k, choice)
            if exp_t_i is None:
                surprise_t = 0.0
            else:
                expected_actual_reward_diff = abs(exp_t_i - outcome)
                surprise_t = expected_actual_reward_diff / (obs_sd + expected_actual_reward_diff)
        else:
            surprise_t = 0.0
        self.surprises.append(surprise_t)
        self.trial_number += 1

    def select_action(self):
        if self.trial_number == 0:
            return random.choice([0, 1])
        elif self.trial_number == 1:
            if self.choices[0] == TARGET_ACTION:
                return ALT_ACTION
            else:
                return TARGET_ACTION

        if ((self.choices[-1] == self.choices[-2]) and
            (self.outcomes[-1] != self.outcomes[-2]) and
            (random.random() < self.tau)):
            if self.outcomes[-1] > self.outcomes[-2]:
                return self.choices[-1]
            else:
                return other_alternative(self.choices[-1])

        if random.random() < self.get_p_explore():
            return random.choice([0, 1])

        if random.random() < self.phi and len(self.choices) > 0:
            return random.choice([0, 1])

        ca_biased = self.__contingent_average(self.k, TARGET_ACTION)
        ca_anti_biased = self.__contingent_average(self.k, ALT_ACTION)
        if (ca_biased is None) or (ca_anti_biased is None) or (ca_biased == ca_anti_biased):
            return choose_randomly()
        elif ca_biased > ca_anti_biased:
            return TARGET_ACTION
        else:
            return ALT_ACTION

    def prob_choose(self, a):
        if self.trial_number == 0:
            return 0.5 if a in (TARGET_ACTION, ALT_ACTION) else 0.0
        if self.trial_number == 1:
            chosen = ALT_ACTION if self.choices[0] == TARGET_ACTION else TARGET_ACTION
            return 1.0 if a == chosen else 0.0

        trend_cond = ((self.choices[-1] == self.choices[-2]) and (self.outcomes[-1] != self.outcomes[-2]))
        p_trend = self.tau if trend_cond else 0.0

        if trend_cond:
            if self.outcomes[-1] > self.outcomes[-2]:
                trend_choice = self.choices[-1]
            else:
                trend_choice = other_alternative(self.choices[-1])
        else:
            trend_choice = None

        p_explore = self.get_p_explore()
        p_inertia = self.phi if len(self.choices) > 0 else 0.0

        ca_biased = self.__contingent_average(self.k, TARGET_ACTION)
        ca_anti_biased = self.__contingent_average(self.k, ALT_ACTION)
        if (ca_biased is None) or (ca_anti_biased is None) or (ca_biased == ca_anti_biased):
            p_contingent_choose_biased = 0.5
        elif ca_biased > ca_anti_biased:
            p_contingent_choose_biased = 1.0
        else:
            p_contingent_choose_biased = 0.0

        p = 0.0
        if p_trend > 0.0:
            p += p_trend * (1.0 if a == trend_choice else 0.0)
        remainder = (1.0 - p_trend)
        p += remainder * (p_explore * 0.5 if a in (TARGET_ACTION, ALT_ACTION) else 0.0)
        rem2 = remainder * (1.0 - p_explore)
        last_choice = self.choices[-1] if len(self.choices) > 0 else None
        inertia_prob = rem2 * p_inertia * (1.0 if a == last_choice else 0.0)
        contingent_prob = rem2 * (1.0 - p_inertia) * (p_contingent_choose_biased if a == TARGET_ACTION else (1.0 - p_contingent_choose_biased))
        p += inertia_prob + contingent_prob
        return max(0.0, min(1.0, p))


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
        return F.softmax(out, dim=2), h

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def select_action(self):
        # (이전 코드와 동일)
        prev_action = self.last_action if self.last_action is not None else 0
        prev_reward = 0 
        
        with torch.no_grad():
            input_tensor = torch.tensor([[prev_action, prev_reward]], dtype=torch.float32, device=device).unsqueeze(1)
            action_probs, self.hidden = self.forward(input_tensor, self.hidden)
        
        # .squeeze(0)으로 (1, 1, 2) -> (1, 2)
        # .multinomial(1) -> (1, 1) -> .item()
        action = torch.multinomial(action_probs.squeeze(0), 1).item()
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
        expected_seq_len = 99  # 100 스텝 -> 99개의 (입력, 타겟) 쌍

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
        return loss.item()
    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*ADV_TAU + target_net_state_dict[key]*(1-ADV_TAU)
        self.target_net.load_state_dict(target_net_state_dict)

# --- 3. 실험 전체를 관리하는 클래스 ---
class Experiment:
   