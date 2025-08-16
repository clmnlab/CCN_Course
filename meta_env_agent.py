# -*- coding: utf-8 -*-
"""
REINFORCE와 Actor-Critic(A2C) 메타 강화학습 에이전트 비교
- 태스크: 확률적 투-암드 밴딧(Two-Armed Bandit)
- 목표: 두 알고리즘의 학습 효율성과 안정성을 시각적으로 비교 분석
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from gym import spaces
from gym.spaces import Discrete, Box
from sklearn.decomposition import PCA
# --- 1. 공통 구성 요소: 하이퍼파라미터 및 환경 ---

class Config:
    """훈련을 위한 하이퍼파라미터 설정"""
    # 훈련 관련
    NUM_EPISODES = 10000  # 훈련할 총 에피소드(밴딧 문제) 수
    EPISODE_LENGTH = 100 # 하나의 에피소드(밴딧 문제) 내에서 행동할 횟수
    LEARNING_RATE = 0.001 # 옵티마이저의 학습률

    # 신경망 구조
    INPUT_SIZE = 3      # 입력 크기 (이전 행동 one-hot, 2, 보상 (1))
    HIDDEN_SIZE = 32    # RNN의 은닉 상태 크기
    OUTPUT_SIZE = 2     # 출력 크기 (각 팔에 대한 행동 확률)

    # A2C 전용 하이퍼파라미터
    GAMMA = 0.98                    # 할인율(Discount Factor)
    VALUE_LOSS_COEF = 0.5           # Critic 손실의 가중치
    ENTROPY_COEF = 0.01             # 엔트로피 보너스 가중치 (탐험 장려)
    
    # 시각화 관련
    MOVING_AVG_WINDOW = 50 # 결과 그래프를 부드럽게 만들기 위한 이동 평균 윈도우
    PCA_COMPONENTS = 2 # PCA 차원

def moving_average(data, window_size):
    """데이터를 부드럽게 만들기 위한 이동 평균 계산"""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


class BanditEnv:
    """확률적 투-암드 밴딧 환경"""
    def __init__(self, correlated=False, probs=None):
        if probs is not None:
            self.reward_probs = np.array(probs)
        elif correlated:
            p1 = np.random.uniform(low=0.0, high=1.0)
            self.reward_probs = np.array([p1, 1.0 - p1])
        else:
            self.reward_probs = np.random.uniform(low=0.0, high=1.0, size=2)
        self.optimal_reward = np.max(self.reward_probs)

    def step(self, action):
        """행동을 받아 보상과 후회를 반환"""
        reward = 1 if np.random.random() < self.reward_probs[action] else 0
        regret = self.optimal_reward - self.reward_probs[action]
        return reward, regret

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
    def __init__(self, training=True):
        super(DynamicBandit, self).__init__()
        # 액션은 '왼쪽(0)'과 '오른쪽(1)' 두 가지입니다.
        self.action_space = spaces.Discrete(2)
        # 관찰 공간은 단순하며, 현재 상태를 나타내는 단일 값입니다.
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.training = training  # 학습 모드 여부
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

    def reset(self, seed=None, options=None, episode_length=None, p0_pair=None):
        """
        새로운 에피소드를 위해 환경을 초기화합니다.
        """
        super().reset(seed=seed)

        # 에피소드 길이를 50에서 100 사이로 무작위 설정 
        self.current_step = 0

        # 기본 보상 확률(p0) 설정
        if p0_pair is not None:
            # p0_pair가 주어진 경우 해당 값을 사용
            p0_left = p0_pair[0]
            self.episode_length = 100
        elif self.training:
            # 학습 시에는 특정 구간을 제외하고 샘플링 
            valid_ranges = [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5)]
            range_index = self.np_random.integers(0, len(valid_ranges))
            p0_left = self.np_random.uniform(*valid_ranges[range_index])
            self.episode_length = self.np_random.integers(50, 101)
        else:
            # 테스트 시에는 전체 구간 [0.0, 0.5]에서 샘플링
            p0_left = self.np_random.uniform(0.0, 0.5)
            if episode_length is None:
                self.episode_length = 100
            else:
                # 테스트 시 에피소드 길이를 명시적으로 지정할 수 있습니다.
                self.episode_length = episode_length
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
    


# --- Actor-Critic (A2C) 알고리즘 구현 ---

class A2CAgent(nn.Module):
    """Actor-Critic 에이전트. 학습, 저장, 로드 기능 포함."""
    def __init__(self, input_size, hidden_size, num_actions, lr=0.002, gamma=0.98, value_loss_coef=0.5, entropy_coef=0.01):
        super(A2CAgent, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # LSTMCell
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # 학습 기록
        self.total_rewards_history = []

    def forward(self, x, hidden_state):
        """순전파: 정책과 가치 계산"""
        h, c = self.lstm_cell(x, hidden_state)
        logits = self.action_head(h)
        value = self.value_head(h)
        return logits, value, (h, c)

    def train_episode(self, env, episode_length=100):
        """단일 에피소드 학습"""
        log_probs = []
        values = []
        rewards = []
        entropies = []

        hidden_state = (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))
        # 첫 입력: 이전 행동과 보상이 없으므로 제로 벡터 사용
        prev_action = torch.zeros(1, self.num_actions)
        prev_reward = torch.zeros(1, 1)

        for _ in range(episode_length):
            # 입력 구성: [이전 행동(one-hot), 이전 보상]
            input_tensor = torch.cat([prev_action, prev_reward], dim=1)
            logits, value, hidden_state = self.forward(input_tensor, hidden_state)
            policy = Categorical(logits=logits)
            action = policy.sample()
            
            _, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(policy.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            entropies.append(policy.entropy())

            # 다음 입력을 위해 현재 행동과 보상 저장
            prev_action = nn.functional.one_hot(action, num_classes=self.num_actions).float()
            prev_reward = torch.tensor([[reward]], dtype=torch.float32)

            if terminated:
                break

        # 학습 기록 저장
        episode_reward = torch.cat(rewards).sum().item()
        self.total_rewards_history.append(episode_reward)

        # --- 손실 계산 ---
        returns = []
        R = 0.0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        values = torch.cat(values).squeeze()
        advantages = returns - values

        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        critic_loss = nn.functional.mse_loss(returns, values)
        entropy_loss = -torch.stack(entropies).mean()
        total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def learn(self, env_class, num_episodes=200, episode_length=100, **env_kwargs):
        """주어진 환경으로 여러 에피소드 학습"""
        for _ in tqdm(range(num_episodes), desc="A2C Training"):
            env = env_class(**env_kwargs)
            env.reset()
            loss = self.train_episode(env, episode_length=episode_length)
            if _ % 20 == 0:
                print(f"Episode {_}, Loss: {loss:.4f}")

    def save(self, path):
        """모델과 학습 기록 저장"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_rewards_history': self.total_rewards_history
        }, path)

    def load(self, path):
        """모델과 학습 기록 로드"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_rewards_history = checkpoint['total_rewards_history']

def collect_episode_data(agent, env, episode_length):
    """
    단일 에피소드를 실행하여 hidden state 및 기타 데이터를 수집하는 함수
    """
    # 데이터 저장을 위한 리스트
    hidden_states, values, action_probs, rewards, actions = [], [], [], [], []
    hidden_state = (torch.zeros(1, agent.hidden_size), torch.zeros(1, agent.hidden_size))
    
    prev_action = torch.zeros(1, agent.num_actions)
    prev_reward = torch.zeros(1, 1)

    for _ in range(episode_length):
        # 현재 hidden state(h)를 저장 (분석 대상)
        hidden_states.append(hidden_state[0].detach().numpy())
        input_tensor = torch.cat([prev_action, prev_reward], dim=1)

        # 에이전트 순전파
        logits, value, hidden_state = agent(input_tensor, hidden_state)
        
        # 데이터 저장
        values.append(value.item())
        probs = torch.softmax(logits, dim=-1)
        action_probs.append(probs.detach().numpy())
        
        # 행동 선택 및 환경과 상호작용
        action = Categorical(logits=logits).sample()
        _, reward, terminated, _, _ = env.step(action.item())
        
        actions.append(action.item())
        rewards.append(reward)

        # 다음 입력을 위해 현재 행동을 원-핫 인코딩
        prev_action = nn.functional.one_hot(action, num_classes=agent.num_actions).float()
        prev_reward = torch.tensor([[reward]], dtype=torch.float32)        
        
        if terminated:
            break
    return {
        'hidden_states': np.array(hidden_states).squeeze(),
        'values': np.array(values),
        'action_probs': np.array(action_probs).squeeze(),
        'rewards': np.array(rewards),
        'actions': np.array(actions),
        'p0': env.p0 # 이 에피소드의 p0 값을 저장
    }

def fit_pca_on_task_difference(all_data, p0_pairs):
    """
    개선된 PCA 방법: 두 과제 간 hidden state의 평균 차이에 대해 PCA를 학습
    """
    # 1. 각 과제별로 데이터 분리
    data_task1 = [d['hidden_states'] for d in all_data if np.allclose(d['p0'], p0_pairs[0])]
    data_task2 = [d['hidden_states'] for d in all_data if np.allclose(d['p0'], p0_pairs[1])]

    # --- 에러 핸들링 추가 ---
    # 만약 특정 과제에 대한 데이터가 하나도 없으면, np.mean이 빈 리스트를 받아 nan을 반환하고 ValueError를 발생시킵니다.
    # 이를 방지하고 더 명확한 에러 메시지를 제공합니다.
    if not data_task1 or not data_task2:
        print("디버깅 정보:")
        print(f"Task 1 (p0={p0_pairs[0]})에 대해 수집된 에피소드 수: {len(data_task1)}")
        print(f"Task 2 (p0={p0_pairs[1]})에 대해 수집된 에피소드 수: {len(data_task2)}")
        collected_p0s = [str(d.get('p0', 'N/A').tolist()) for d in all_data]
        print(f"실제로 수집된 p0 값들 (처음 10개): {collected_p0s[:10]}")
        raise ValueError("PCA 분석에 필요한 한 개 또는 두 개 과제에 대한 데이터가 수집되지 않았습니다. 데이터 수집 과정이나 p0 값 비교 로직을 확인해주세요.")
    # --- 에러 핸들링 종료 ---

    # 2. 각 과제별로 시간 스텝에 따른 평균 hidden state 계산
    mean_traj_task1 = np.mean(data_task1, axis=0)
    mean_traj_task2 = np.mean(data_task2, axis=0)

    # 3. 두 과제의 평균 궤적 간의 차이를 계산
    task_difference_trajectory = mean_traj_task1 - mean_traj_task2

    # 4. 이 '차이' 데이터에 대해 PCA를 학습
    pca = PCA(n_components=Config.PCA_COMPONENTS)
    # pca = PCA(n_components=10)
    pca.fit(task_difference_trajectory)
    
    print(f"\n과제 차이 기반 PCA 분석 완료. 설명된 분산: {pca.explained_variance_ratio_}")
    return pca

def plot_pca_trajectories(all_data, pca, p0_pairs):
    """Figure 1.d: Hidden State의 PCA 궤적 시각화"""
    plt.figure(figsize=(8, 8))
    colors = {str(p0_pairs[0]): 'crimson', str(p0_pairs[1]): 'royalblue'}
    
    for data in all_data:
        p0_str = str(data['p0'].tolist())
        if p0_str not in colors: continue
        
        transformed_states = pca.transform(data['hidden_states'])
        plt.plot(transformed_states[:, 0], transformed_states[:, 1], 
                 color=colors[p0_str], alpha=0.3, label=f'p0 = {p0_str}' if p0_str not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title("Hidden State Trajectories in PCA Space (Fig 1.d)")
    plt.xlabel("Task-Coding Axis (PC1)")
    plt.ylabel("Within-Task Progress Axis (PC2)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_performance_over_time(all_data, p0_pairs):
    """Figure 1.e & 1.f: 행동 확률 및 가치 함수 시각화"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = {str(p0_pairs[0]): 'crimson', str(p0_pairs[1]): 'royalblue'}

    for p0_pair in p0_pairs:
        p0_str = str(p0_pair)
        relevant_data = [d for d in all_data if np.allclose(d['p0'], p0_pair)]
        if not relevant_data: continue

        correct_arm = np.argmax(p0_pair)
        
        prob_correct = np.array([d['action_probs'][:, correct_arm] for d in relevant_data])
        mean_prob_correct = prob_correct.mean(axis=0)
        std_err_prob = prob_correct.std(axis=0) / np.sqrt(len(relevant_data))

        values = np.array([d['values'] for d in relevant_data])
        mean_values = values.mean(axis=0)
        std_err_values = values.std(axis=0) / np.sqrt(len(relevant_data))

        time_steps = np.arange(len(mean_prob_correct))

        ax1.plot(time_steps, mean_prob_correct, label=f'p(correct) for p0={p0_str}', color=colors[p0_str])
        ax1.fill_between(time_steps, mean_prob_correct - std_err_prob, mean_prob_correct + std_err_prob, color=colors[p0_str], alpha=0.2)
        
        ax2.plot(time_steps, mean_values, label=f'Value for p0={p0_str}', color=colors[p0_str])
        ax2.fill_between(time_steps, mean_values - std_err_values, mean_values + std_err_values, color=colors[p0_str], alpha=0.2)

    ax1.set_title("Probability of Choosing Correct Arm (Fig 1.e)")
    ax1.set_ylabel("p(correct)")
    ax1.legend()
    ax1.grid(True)
    ax1.axhline(0.5, color='k', linestyle='--', alpha=0.5)

    ax2.set_title("Estimated Value Over Time (Fig 1.f)")
    ax2.set_xlabel("Time Step in Episode")
    ax2.set_ylabel("State Value V(s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


