import torch
from networks import RegularizedRNN
from utils import train_val_split, TimeseriesDataset, prepare_dataset
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, data_loader, n_epochs, lr, alpha, beta, gamma):
    """
    정규화된 손실 함수를 사용하여 RNN 모델을 훈련합니다.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("--- 훈련 시작 ---")
    for epoch in range(n_epochs):
        epoch_total_loss = 0.0
        
        for u_batch, target_z_batch in data_loader:
            # 순방향 패스
            pred_z, _, r_states = model(u_batch)
            
            # 손실 계산
            total_loss, loss_e, _, _, _ = calculate_regularized_loss(
                model, pred_z, target_z_batch, r_states, alpha, beta, gamma
            )
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
        
        avg_epoch_loss = epoch_total_loss / len(data_loader)
        
        if (epoch + 1) % 50 == 0:
            # EMG 오차만 따로 계산하여 출력 (성능 지표)
            with torch.no_grad():
                pred_z_val, _, _ = model(u_batch)
                emg_error = nn.MSELoss()(pred_z_val, target_z_batch)

            print(f'Epoch [{epoch+1}/{n_epochs}], Total Loss: {avg_epoch_loss:.6f}, EMG MSE: {emg_error.item():.6f}')
            
    print("--- 훈련 종료 ---")
    return model