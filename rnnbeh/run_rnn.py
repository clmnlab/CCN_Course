import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 1. 환경 정의: 두 팔 슬롯머신 (TensorFlow 버전과 동일) ---
class TwoArmedBandit:
    """
    두 개의 팔(선택지)을 가진 슬롯머신 환경입니다.
    각 팔은 정해진 확률로 보상 1을 반환하고, 아니면 0을 반환합니다.
    """
    def __init__(self, prob_arm1=0.3, prob_arm2=0.7):
        self.probabilities = [prob_arm1, prob_arm2]
        print(f"슬롯머신 생성됨: 1번 팔 보상 확률={prob_arm1}, 2번 팔 보상 확률={prob_arm2}")
        print(f"이 환경에서 최적의 선택은 {np.argmax(self.probabilities)+1}번 팔입니다.\n")

    def pull_arm(self, arm_index):
        """선택한 팔을 당겨 보상을 받습니다."""
        if arm_index not in [0, 1]:
            raise ValueError("arm_index는 0 또는 1이어야 합니다.")
        
        if np.random.rand() < self.probabilities[arm_index]:
            return 1
        else:
            return 0

# --- 2. 행동 데이터 생성 함수 (TensorFlow 버전과 동일) ---
def generate_behavioral_data_with_q_learning(environment, episodes=200, learning_rate=0.1, exploration_rate=0.1):
    """
    간단한 Q-러닝 에이전트를 실행하여 행동-보상 기록을 생성합니다.
    """
    q_values_all = []
    q_values = [0.0, 0.0]# 두 팔에 대한 Q-값 초기화
    q_values_all.append(q_values)
    history = []
    print("--- Q-러닝 에이전트로 행동 데이터 생성 시작 ---")
    for episode in range(episodes):
        if np.random.rand() < exploration_rate:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(q_values)

        reward = environment.pull_arm(action)
        old_q_value = q_values[action]
        q_values[action] = old_q_value + learning_rate * (reward - old_q_value)
        q_values_all.append(q_values)
        history.append({'action': action, 'reward': reward})

        if (episode + 1) % 5 == 0:
            print(f"에피소드 {episode+1:3d}: Q-가치 = [{q_values[0]:.2f}, {q_values[1]:.2f}]")

    print(f"\nQ-러닝 학습 후 예측하는 최적 행동: {np.argmax(q_values)}번 팔")
    print(f"총 {len(history)}개의 행동-보상 데이터 생성 완료.\n")
    return history

# --- 3. PyTorch RNN 학습을 위한 데이터 준비 ---
def create_sequence_data(history, environment, sequence_length=5):
    """
    행동-보상 기록을 시퀀스 데이터로 변환합니다. (NumPy 배열 반환)
    """
    X, y = [], []
    optimal_action = np.argmax(environment.probabilities)
    for i in range(len(history) - sequence_length):
        sequence = history[i : i + sequence_length]
        input_vector = []
        for step in sequence:
            input_vector.extend([step['action'], step['reward']])
            # why using extend not append?
            # extend는 리스트를 펼쳐서 추가, append는 리스트 전체를 하나의 요소로 추가

        X.append(input_vector)
        y.append(optimal_action)
    
    # NumPy 배열로 변환 후 PyTorch Tensor로 변환 준비
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int64)
    return X_np, y_np

# --- 4. PyTorch RNN 모델 정의 ---
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True 옵션으로 입력 텐서의 차원을 (배치 크기, 시퀀스 길이, 특징 수)로 설정
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, output_size)

    def forward(self, x):
        # 초기 은닉 상태를 0으로 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN 순전파
        out, _ = self.rnn(x, h0)
        
        # 마지막 타임스텝의 출력만 사용
        out = out[:, -1, :]
        
        # 완전 연결 계층 통과
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 5. 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 환경 인스턴스 생성
    env = TwoArmedBandit(prob_arm1=0.2, prob_arm2=0.8)

    # 2. Q-러닝으로 행동 데이터 생성
    behavioral_history = generate_behavioral_data_with_q_learning(env, episodes=200)

    # 3. RNN 학습용 데이터셋 생성 및 Tensor 변환
    SEQUENCE_LENGTH = 5
    X_data, y_data = create_sequence_data(behavioral_history, env, sequence_length=SEQUENCE_LENGTH)
    
    # 입력 데이터 형태 변경: (샘플 수, 시퀀스 길이, 특징 수)
    X_reshaped = X_data.reshape((-1, SEQUENCE_LENGTH, 2))

    # NumPy 배열을 PyTorch Tensor로 변환
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    
    print(f"--- PyTorch 학습 데이터 준비 완료 ---")
    print(f"입력 텐서(X) 형태: {X_tensor.shape}")
    print(f"정답 텐서(y) 형태: {y_tensor.shape}\n")

    # DataLoader 생성
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 4. 모델, 손실 함수, 옵티마이저 정의
    INPUT_SIZE = 2
    HIDDEN_SIZE = 16
    NUM_LAYERS = 1
    OUTPUT_SIZE = 2
    
    model = RNNModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- PyTorch RNN 모델 정의 완료 ---")
    print(model)
    print("\n--- 모델 학습 시작 ---")

    # 5. 모델 학습 루프
    num_epochs = 15
    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(dataloader):
            # 순전파
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 에포크마다 손실 출력
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("모델 학습 완료!\n")

    # 6. 학습된 모델로 예측 수행
    print("--- 학습된 RNN으로 예측 테스트 ---")
    model.eval() # 모델을 평가 모드로 전환
    
    test_sequence_np = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
    test_sequence_reshaped = test_sequence_np.reshape((-1, SEQUENCE_LENGTH, 2))
    test_tensor = torch.tensor(test_sequence_reshaped)

    with torch.no_grad(): # 그래디언트 계산 비활성화
        prediction_logits = model(test_tensor)
        # Softmax를 적용하여 확률로 변환
        prediction_probs = torch.softmax(prediction_logits, dim=1)
        predicted_action = torch.argmax(prediction_probs, dim=1).item()

    print(f"테스트 시퀀스: {test_sequence_np[0]}")
    print(f"RNN의 예측 확률 (0번, 1번 팔): [{prediction_probs[0][0]:.3f}, {prediction_probs[0][1]:.3f}]")
    print(f"RNN이 예측한 다음 최적 행동: {predicted_action}번 팔")
