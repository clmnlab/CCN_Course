import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicTwoArmedBandit(gym.Env):
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
        super(DynamicTwoArmedBandit, self).__init__()
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

# --- 환경 사용 예시 ---
if __name__ == '__main__':
    # 1. 학습 모드로 환경 초기화
    env = DynamicTwoArmedBandit()
    obs, info = env.reset(training=True)
    
    print("--- 학습 모드 (Training Mode) ---")
    print(f"초기화 완료.")
    print(f"이번 에피소드 길이: {env.episode_length} trials")
    print(f"샘플링된 p0 값: {info['p0']}")
    assert not (0.1 < info['p0'][0] < 0.2 or 0.3 < info['p0'][0] < 0.4), "p0 값이 제외 구간에 포함됨!"
    print("p0 값이 학습 허용 구간 내에 있음을 확인했습니다.")

    # 2. 테스트 모드로 환경 초기화
    obs, info = env.reset(training=False)
    print("\n--- 테스트 모드 (Test Mode) ---")
    print(f"초기화 완료.")
    print(f"이번 에피소드 길이: {env.episode_length} trials")
    print(f"샘플링된 p0 값: {info['p0']}")
    
    # 3. 간단한 에피소드 실행
    print("\n--- 10번의 시행(Trial) 실행 예시 ---")
    done = False
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample() # 무작위 액션 선택
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Trial {i+1:2d}: Action={action}, Reward={reward}, "
            f"P(a): {info['current_probabilities'].round(3)}, "
            f"n(a): {info['trials_since_chosen']}"
        )
        if terminated or truncated:
            done = True
    print(f"\n10번의 시행 후 총 보상: {total_reward}")