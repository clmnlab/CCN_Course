# -*- coding: utf-8 -*-
"""
REINFORCE와 Actor-Critic(A2C) 메타 강화학습 에이전트 비교
- 태스크: 확률적 투-암드 밴딧(Two-Armed Bandit)
- 목표: 두 알고리즘의 학습 효율성과 안정성을 시각적으로 비교 분석
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. 공통 구성 요소: 하이퍼파라미터 및 환경 ---

class Config:
    """훈련을 위한 하이퍼파라미터 설정"""
    # 훈련 관련
    NUM_EPISODES = 200  # 훈련할 총 에피소드(밴딧 문제) 수
    EPISODE_LENGTH = 100 # 하나의 에피소드(밴딧 문제) 내에서 행동할 횟수
    LEARNING_RATE = 0.002 # 옵티마이저의 학습률

    # 신경망 구조
    INPUT_SIZE = 2      # 입력 크기 (이전 행동 one-hot)
    HIDDEN_SIZE = 32    # RNN의 은닉 상태 크기
    OUTPUT_SIZE = 2     # 출력 크기 (각 팔에 대한 행동 확률)

    # A2C 전용 하이퍼파라미터
    GAMMA = 0.98                    # 할인율(Discount Factor)
    VALUE_LOSS_COEF = 0.5           # Critic 손실의 가중치
    ENTROPY_COEF = 0.01             # 엔트로피 보너스 가중치 (탐험 장려)
    
    # 시각화 관련
    MOVING_AVG_WINDOW = 50 # 결과 그래프를 부드럽게 만들기 위한 이동 평균 윈도우

class BanditEnv:
    """
    확률적 투-암드 밴딧 환경.
    객체가 생성될 때마다 각 팔의 보상 확률이 무작위로 새로 설정된다.
    이는 메타 학습의 '새로운 태스크'에 해당한다.
    """
    def __init__(self):
        # 0과 1 사이의 균등 분포에서 두 팔의 보상 확률을 무작위로 선택
        self.reward_probs = np.random.uniform(low=0.0, high=1.0, size=2)

    def step(self, action):
        """에이전트의 행동을 받아 보상을 반환"""
        # 선택된 팔의 보상 확률에 따라 1(성공) 또는 0(실패)의 보상을 반환
        return 1 if np.random.random() < self.reward_probs[action] else 0

def moving_average(data, window_size):
    """데이터를 부드럽게 만들기 위한 이동 평균 계산"""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# --- 2. REINFORCE 알고리즘 구현 ---
import numpy as np
import gym
from gym import spaces
import numpy as np
import gym
from gym.spaces import Discrete, Box



class DynamicBandit(gym.Env):
    """
    논문 'Prefrontal cortex as a meta-reinforcement learning system'의
    Simulation 1 (Tsutsui et al. task)을 구현한 환경입니다.

    Reference:
        Wang, J. X., Kurth-Nelson, Z., Kumaran, D., Tirumala, D., Soyer, H.,
        Leibo, J. Z., Hassabis, D., & Botvinick, M. (2018).
        Prefrontal cortex as a meta-reinforcement learning system.
        Nature Neuroscience, 21(6), 860–868.
        Methods Section -> Simulation 1 [cite: 914-926]
    """
    def __init__(self):
        super(DynamicBandit, self).__init__()
        # 액션은 '왼쪽(0)'과 '오른쪽(1)' 두 가지입니다.
        self.action_space = spaces.Discrete(2)
        # 관찰 공간은 단순하며, 현재 상태를 나타내는 단일 값입니다.
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        # 이 환경에서 관찰은 상태 정보를 제공하지 않으므로 고정값을 반환합니다.
        # 에이전트는 이전 행동과 보상 이력을 통해 학습해야 합니다.
        return np.array([0.0], dtype=np.float32)

    def _get_info(self):
        # 디버깅 및 분석을 위한 추가 정보
        return {
            "p0": self.p0,
            "current_probabilities": self.current_probabilities,
            "trials_since_chosen": self.trials_since_chosen
        }

    def reset(self, seed=None, options=None, training=True):
        """
        새로운 에피소드를 위해 환경을 초기화합니다.
        """
        super().reset(seed=seed)

        # 에피소드 길이를 50에서 100 사이로 무작위 설정 
        self.episode_length = self.np_random.integers(50, 101)
        self.current_step = 0

        # 기본 보상 확률(p0) 설정
        if training:
            # 학습 시에는 특정 구간을 제외하고 샘플링 
            valid_ranges = [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5)]
            range_index = self.np_random.integers(0, len(valid_ranges))
            p0_left = self.np_random.uniform(*valid_ranges[range_index])
        else:
            # 테스트 시에는 전체 구간 [0.0, 0.5]에서 샘플링
            p0_left = self.np_random.uniform(0.0, 0.5)

        p0_right = 0.5 - p0_left # p0 합은 항상 0.5 
        self.p0 = np.array([p0_left, p0_right])

        # 각 액션이 마지막으로 선택된 후 경과된 시행 횟수 초기화
        self.trials_since_chosen = np.zeros(2, dtype=int)
        
        # 현재 보상 확률 계산 및 저장
        self.current_probabilities = self._calculate_probabilities()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _calculate_probabilities(self):
        """
        논문의 공식에 따라 현재 보상 확률을 계산합니다.
        p(a) = 1 - (1 - p0(a))^(n(a) + 1) 
        """
        n = self.trials_since_chosen
        return 1 - (1 - self.p0) ** (n + 1)

    def step(self, action):
        """
        선택된 액션을 실행하고 환경 상태를 업데이트합니다.
        """
        if self.current_step >= self.episode_length:
            # 에피소드 길이 초과 시 비정상 종료 처리
            return self._get_obs(), 0, True, False, self._get_info()

        self.current_step += 1

        # 현재 확률에 따라 보상 결정
        self.current_probabilities = self._calculate_probabilities()
        prob = self.current_probabilities[action]
        reward = 1.0 if self.np_random.random() < prob else 0.0

        # trials_since_chosen 카운터 업데이트
        # 선택된 액션은 0으로 리셋, 선택되지 않은 액션은 1 증가
        other_action = 1 - action
        self.trials_since_chosen[action] = 0
        self.trials_since_chosen[other_action] += 1

        # 종료 조건 확인
        terminated = self.current_step >= self.episode_length
        truncated = False  # Truncated는 사용하지 않음

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    

class ReinforceAgent(nn.Module):
    """REINFORCE 에이전트 (Actor-Only 구조)"""
    def __init__(self, input_size, hidden_size, output_size):
        super(ReinforceAgent, self).__init__()
        self.rnn = nn.lstm_cell(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        """순전파: 정책(행동 확률)을 계산"""
        hidden_state = self.rnn(x, hidden_state)
        logits = self.action_head(hidden_state)
        return logits, hidden_state

def train_reinforce(config):
    """REINFORCE 알고리즘으로 에이전트를 훈련시키는 함수"""
    print("REINFORCE 훈련을 시작합니다...")
    
    # 모델과 옵티마이저 초기화
    agent = ReinforceAgent(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE)
    
    total_rewards_history = []

    for i in tqdm(range(config.NUM_EPISODES), desc="REINFORCE Training"):
        # 매 에피소드마다 새로운 밴딧 문제 생성
        env = BanditEnv()
        
        # 에피소드 데이터 저장을 위한 리스트
        log_probs = []
        rewards = []
        
        # RNN을 위한 초기값 설정
        hidden_state = torch.zeros(1, config.HIDDEN_SIZE)
        action_input = torch.zeros(1, config.INPUT_SIZE)

        # 단일 에피소드 진행 (하나의 밴딧 문제 해결 과정)
        for _ in range(config.EPISODE_LENGTH):
            # 1. 행동 선택
            logits, hidden_state = agent(action_input, hidden_state)
            policy = Categorical(logits=logits)
            action = policy.sample()

            # 2. 환경과 상호작용
            _, reward, _, _ = env.step(action.item())

            # 3. 결과 저장
            log_probs.append(policy.log_prob(action))
            rewards.append(reward)

            # 4. 다음 입력을 위해 현재 행동을 원-핫 인코딩
            action_input = nn.functional.one_hot(action, num_classes=config.INPUT_SIZE).float().unsqueeze(0)

        total_rewards_history.append(sum(rewards))

        # 5. 손실 계산 및 파라미터 업데이트 (REINFORCE의 핵심)
        # 에피소드가 끝난 후, 전체 보상을 바탕으로 한 번에 업데이트
        loss = []
        total_reward = sum(rewards)
        
        # 높은 총 보상을 받은 에피소드의 행동 확률은 높이고, 낮은 보상을 받은 행동 확률은 낮춘다.
        for log_p in log_probs:
            loss.append(-log_p * total_reward)

        optimizer.zero_grad()
        # 모든 시간 스텝의 손실을 합산하여 역전파
        total_loss = torch.stack(loss).sum()
        total_loss.backward()
        optimizer.step()

    return total_rewards_history

# --- 3. Actor-Critic (A2C) 알고리즘 구현 ---

class A2CAgent(nn.Module):
    """Actor-Critic 에이전트. 정책(Actor)과 가치(Critic)를 모두 출력한다."""
    def __init__(self, input_size, hidden_size, num_actions):
        super(A2CAgent, self).__init__()
        # Actor와 Critic이 RNN의 일부를 공유
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        # Actor Head: 어떤 행동을 할지 결정 (정책)
        self.action_head = nn.Linear(hidden_size, num_actions)
        # Critic Head: 현재 상태가 얼마나 좋은지 평가 (가치)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_state):
        """순전파: 정책과 가치를 모두 계산"""
        h, c = self.lstm_cell(x, hidden_state)
        logits = self.action_head(h)
        value = self.value_head(h)
        new_hidden_state = (h, c)  # LSTM의 은닉 상태는 (h, c) 튜플로 반환
        # logits는 정책을 나타내고, value는 상태의 가치를 나타냅니다.
        # hidden_state는 다음 시간 스텝을 위해 업데이트된 은닉 상태입니다.
        return logits, value, new_hidden_state
    


def train_a2c(config):
    """Advantage Actor-Critic (A2C) 알고리즘으로 에이전트를 훈련"""
    print("\nActor-Critic (A2C) 훈련을 시작합니다...")

    agent = A2CAgent(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE)

    total_rewards_history = []

    for i in tqdm(range(config.NUM_EPISODES), desc="A2C Training"):
        env = DynamicBandit()
        env.reset(training=True)  # 학습 모드로 초기화
        log_probs = []
        values = []
        rewards = []
        entropies = [] # 탐험을 장려하기 위한 엔트로피

        hidden_state = (torch.zeros(1, config.HIDDEN_SIZE), 
                        torch.zeros(1, config.HIDDEN_SIZE))
        action_input = torch.zeros(1, config.INPUT_SIZE)
        
        for _ in range(config.EPISODE_LENGTH):
            logits, value, hidden_state = agent(action_input, hidden_state)
            
            policy = Categorical(logits=logits)
            action = policy.sample()
            
            _, reward, _, _, _ = env.step(action.item())
            
            log_probs.append(policy.log_prob(action))
            values.append(value)
            # rewards.append(reward)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            entropies.append(policy.entropy())

            # action_input = nn.functional.one_hot(action, num_classes=config.INPUT_SIZE).float().unsqueeze(0)
            action_input = nn.functional.one_hot(action, num_classes=config.INPUT_SIZE).float()

        # total_rewards_history.append(sum(rewards))
        total_rewards_history.append(torch.cat(rewards).sum().item())

        # --- A2C 손실 계산 (REINFORCE와의 핵심적인 차이) ---
        
        # 1. 할인된 미래 보상(Return) 계산
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config.GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # 2. 어드밴티지(Advantage) 계산: A(s,a) = R - V(s)
        # 어드밴티지: 예상보다 실제 보상이 얼마나 더 좋았는지를 나타내는 척도
        values = torch.cat(values).squeeze()
        advantages = returns - values
        
        # 3. 최종 손실 계산: Actor 손실 + Critic 손실 + 엔트로피 보너스
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean() # Critic으로 그래디언트 전파 방지
        critic_loss = nn.functional.mse_loss(returns, values)
        entropy_loss = -torch.stack(entropies).mean()

        total_loss = actor_loss + config.VALUE_LOSS_COEF * critic_loss + config.ENTROPY_COEF * entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return total_rewards_history

# --- 4. 메인 실행 및 결과 시각화 ---

if __name__ == "__main__":
    config = Config()

    # 각 알고리즘 훈련 실행
    # reinforce_rewards = train_reinforce(config)
    a2c_rewards = train_a2c(config)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    
    # 이동 평균을 적용하여 그래프를 부드럽게 표현
    # reinforce_smooth = moving_average(reinforce_rewards, config.MOVING_AVG_WINDOW)
    a2c_smooth = moving_average(a2c_rewards, config.MOVING_AVG_WINDOW)
    
    # plt.plot(reinforce_smooth, label=f'REINFORCE (Smoothed)', color='blue', alpha=0.8)
    plt.plot(a2c_smooth, label=f'Actor-Critic (A2C) (Smoothed)', color='red', alpha=0.8)

    # 원본 데이터도 옅하게 표시
    # plt.plot(reinforce_rewards, color='blue', alpha=0.2)
    plt.plot(a2c_rewards, color='red', alpha=0.2)
    
    plt.title('REINFORCE vs. Actor-Critic (A2C) 성능 비교')
    plt.xlabel('에피소드 (Episode)')
    plt.ylabel(f'총 보상 (Smoothed over {config.MOVING_AVG_WINDOW} episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()