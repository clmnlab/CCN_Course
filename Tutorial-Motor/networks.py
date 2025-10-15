import torch
import torch.nn as nn
import numpy as np
from utils import rectified_tanh

class RegularizedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, g, h, tau_over_dt=5):
        super(RegularizedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.tau_over_dt = tau_over_dt  # Time constant
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Weight initialization
        self.J = nn.Parameter(torch.randn(hidden_size, hidden_size) * (g / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))))
        self.B = nn.Parameter(torch.randn(hidden_size, input_size) * (h / torch.sqrt(torch.tensor(input_size, dtype=torch.float))))
        self.bx = nn.Parameter(torch.zeros(hidden_size))

        # Nonlinearity
        self.nonlinearity = rectified_tanh

    def forward(self, input, hidden):
        # Calculate the visible firing rate from the hidden state.
        firing_rate_before = self.nonlinearity(hidden)

        # Update hidden state
        recurrent_drive = torch.matmul(self.J, firing_rate_before.transpose(0, 1))
        input_drive = torch.matmul(self.B, input.transpose(0, 1))
        total_drive = recurrent_drive + input_drive + self.bx.unsqueeze(1)
        total_drive = total_drive.transpose(0, 1)

        # Euler integration for continuous-time update
        hidden = hidden + (1 / self.tau_over_dt) * (-hidden + total_drive)

        # Calculate the new firing rate given the update.
        firing_rate = self.nonlinearity(hidden)

        # Project the firing rate linearly to form the output
        output = self.output_linear(firing_rate)

        # Regularization terms (used for R1 calculation)
        firing_rate_reg = firing_rate.pow(2).sum()

        return output, hidden, firing_rate_reg

    def init_hidden(self, batch_size):
        # Initialize hidden state with batch dimension
        return torch.zeros(batch_size, self.hidden_size)
    


class RegularizedRNN_Ver2(nn.Module):
    def __init__(self, n_inputs, n_neurons, n_outputs, tau_ms=50.0, dt_ms=5.0, g=1.2):
        """
        Sussillo et al. (2015)의 연속 시간 RNN 모델 초기화.

        Args:
            n_inputs (int): 입력 차원 (I)
            n_neurons (int): 순환 유닛(뉴런)의 수 (N)
            n_outputs (int): 출력 차원 (M)
            tau_ms (float): 뉴런의 시간 상수 (τ)
            dt_ms (float): 시뮬레이션 시간 스텝 (Δt)
            g (float): 순환 가중치 행렬의 스케일 (g)
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        
        # 시간 관련 파라미터
        self.tau = tau_ms
        self.dt = dt_ms
        self.alpha_dt = self.dt / self.tau  # 오일러 적분을 위한 상수

        # 입력 가중치 행렬 B 초기화 (I -> N)
        self.B = nn.Linear(n_inputs, n_neurons, bias=False)
        nn.init.xavier_normal_(self.B.weight)

        # 순환 가중치 행렬 J 초기화 (N -> N)
        self.J = nn.Linear(n_neurons, n_neurons)
        # 초기 가중치를 g/sqrt(N) 스케일의 정규분포에서 샘플링
        initial_J = torch.randn(n_neurons, n_neurons) * (g / (n_neurons**0.5))
        self.J.weight.data = initial_J
        
        # 출력 가중치 행렬 W 초기화 (N -> M)
        self.W = nn.Linear(n_neurons, n_outputs)
        nn.init.xavier_normal_(self.W.weight)

        # 편향 벡터 b^x, b^z
        self.b_x = self.J.bias # J의 bias를 b_x로 사용
        nn.init.zeros_(self.b_x)
        self.b_z = self.W.bias # W의 bias를 b_z로 사용
        nn.init.zeros_(self.b_z)

    def forward(self, u, initial_x=None):
        """
        RNN의 순방향 패스를 시뮬레이션.

        Args:
            u (Tensor): 입력 시계열 데이터 (batch_size, n_timesteps, n_inputs)
            initial_x (Tensor, optional): 초기 은닉 상태. Defaults to None (zeros).

        Returns:
            Tuple: 출력 z와 은닉 상태 x의 시계열
        """
        batch_size, n_timesteps, _ = u.shape
        
        # 초기 은닉 상태 x 설정
        if initial_x is None:
            x = torch.zeros(batch_size, self.n_neurons, device=u.device)
        else:
            x = initial_x

        # 결과를 저장할 리스트
        x_history = []
        z_history = []

        # 오일러 적분을 사용하여 시간 스텝별로 시뮬레이션
        for t in range(n_timesteps):
            # 발화율 r 계산 (활성화 함수: tanh)
            r = torch.tanh(x)
            
            # 상태 업데이트 방정식: τ * dx/dt = -x + J*r + B*u + b_x
            # 오일러 적분: x_new = x + (dt/τ) * (-x + J*r + B*u + b_x)
            dx = -x + self.J(r) + self.B(u[:, t, :])
            x = x + self.alpha_dt * dx
            
            # 출력 z 계산
            z = self.W(r)
            
            # 히스토리 저장
            x_history.append(x)
            z_history.append(z)
            
        # 텐서로 변환
        x_out = torch.stack(x_history, dim=1)
        z_out = torch.stack(z_history, dim=1)
        
        return z_out, x_out