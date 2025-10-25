import os
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
import utils
import agents
import tasks
from pathlib import Path
from tqdm import tqdm
import argparse
import json    

from importlib import reload
reload(agents)
reload(utils)
reload(tasks)
from utils import run_rollout
from tasks import CentreOutFF
from agents import SLAgent
# from sb3_contrib import RecurrentPPO
# ========================================================================================
# 2. 메인 훈련 및 적응 함수
# ========================================================================================
def run_experiment(name='exp_train', device='cpu', load_path=None, config = 'params.json', condition = 'train'):
    """
    초기 훈련과 적응 학습을 순차적으로 진행하는 메인 함수.
    """
    with open(config, 'r') as f:
        config = json.load(f)
    
    # --- parameters setting ---
    env_params = config['env_params']
    train_params = config['training_params']

    # --- 디렉토리 설정 ---
    save_dir = Path.cwd() / 'results' / name
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving in the following directory: {save_dir}")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")




    # Define task and the effector
    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

    env = CentreOutFF(effector=effector, **env_params)
    n_input = env.observation_space.shape[0]
    n_output = env.n_muscles
    loss_weight = np.array([3e+3,1e+5,1e-1,3e-4,0,0,0]) 
    agent = SLAgent(obs_dim=n_input, action_dim=n_output, 
                    batch_size = train_params['batch_size'], 
                    loss_weight=loss_weight,
                    device=device, lr=1e-4)
    # --- 저장된 모델이 있으면 로드 ---
    if load_path is not None:
        load_path = Path(load_path)
        if load_path.exists():
            
            print(f"loading a saved model: {load_path}")
            agent.load(load_path)
        else:
            print(f"Model does not exist: {load_path}")
    agent.save(save_dir / f'agent_0.pth')

    losses = []
    for batch in tqdm(range(train_params['n_batch']), desc="progress of training"):
        data = run_rollout(env, agent, batch_size=train_params['batch_size'], ff_coefficient=0., condition=condition)
        loss = agent.update(data)
        losses.append(loss)

        if (batch + 1) % train_params['log_interval'] == 0:
            avg_loss = np.mean(losses[-train_params['log_interval']:])
            tqdm.write(f"Batch {batch+1}/{train_params['n_batch']}, recent {train_params['log_interval']} batch averaged Loss: {avg_loss:.6f}")
            agent.save(save_dir / f'agent_{batch+1}.pth')
    with open(save_dir / 'losses.json', 'w') as f:
        json.dump({'losses': losses}, f, indent=4)
    
# ========================================================================================
# 4. 스크립트 실행
# ========================================================================================
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, default='exp_train', help='실험 이름')
    # parser.add_argument('--config', type=str, default='parameters.json', help='실험 이름')
    # parser.add_argument('--device', type=str, default='cuda', help='사용할 디바이스: cpu 또는 cuda')
    # parser.add_argument('--ff_coeff', type=float, default=0., help='force field, 0 or 8')
    # parser.add_argument('--load', type=str, default=None, help='불러올 모델의 경로 (예: results/exp_train/agent_1000.pth)')
    # args = parser.parse_args()
    # run_experiment(name=args.name, device=args.device, load_path=args.load)
    # run_experiment(name='baseline_', device='cuda',load_path=None, config='params.json', condition='train',ff_coeff=0.0)
    # print("baseline training done")
    # run_experiment(name='adapt1_', device='cuda', load_path='results/baseline_/agent_30000.pth', config='params.json', condition='test',ff_coeff=8.0)
    # print("adapt1 training done")
    run_experiment(name='baseline', device='cuda', load_path=None, config='params.json', condition='test')
    print("baseline  done")
    # run_experiment(name='adapt2_', device='cuda', load_path='results/washout_/agent_10000.pth', config='params.json', condition='test',ff_coeff=8.0)
    print("relearning done")


    
    
    