import os
import sys 
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from tqdm import tqdm
from torch.distributions import Normal
import torch.nn.functional as F
import networks
from importlib import reload
reload(networks)
from networks import GRUPolicy
 
class SLAgent:
    """Supervised learning agent with a GRU-based policy network."""
    def __init__(self, obs_dim, action_dim, batch_size = 32, hidden_dims=128, device='cpu', lr=3e-4, 
                 loss_weight=None, gamma=0.99, alpha=0.2,
                 freeze_output_layer=False, freeze_input_layer=False):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha  # entropy temperature
        self.network = GRUPolicy(obs_dim, hidden_dims, action_dim, device=device, 
                                       freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer).to(device)
        self.policy_optimizer = optim.Adam(self.network.parameters(), lr=lr)
        # self.replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=1000, device=device)        
        self.loss_weight = loss_weight if loss_weight is not None else np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0])
    def select_action(self, obs, h0=None): ## we can consider "deterministic" option
        h = self.network.init_hidden(batch_size=self.batch_size) if h0 is None else h0
        action, h = self.network(obs, h)
        return action, h
    
    def update(self, data):
        loss, _ = self.calc_loss(data, self.loss_weight)
        loss.to(self.device)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()
    
    def save(self, save_dir):
        th.save(self.network.state_dict(), f"{save_dir}")
        # print("done.")

    def load(self, load_dir):
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.network.load_state_dict(th.load(f"{load_dir}", map_location=device))

        
    def calc_loss(self, data, loss_weight):
        loss = {
            'position': None,
            'jerk': None,
            'muscle': None,
            'muscle_derivative': None,
            'hidden': None,
            'hidden_derivative': None,
            'hidden_jerk': None,}

        loss['position'] = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1),axis=1) # average over time not targets
        loss['jerk'] = th.mean(th.sum(th.square(th.diff(data['vel'], n=2, dim=1)), dim=-1))
        loss['muscle'] = th.mean(th.sum(data['all_force'], dim=-1))
        loss['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_force'], n=1, dim=1)), dim=-1))
        loss['hidden'] = th.mean(th.square(data['all_hidden']))
        loss['hidden_derivative'] = th.mean(th.square(th.diff(data['all_hidden'], n=1, dim=1)))
        loss['hidden_jerk'] = th.mean(th.square(th.diff(data['all_hidden'], n=3, dim=1)))
        
        # if loss_weight is None:
        #     # currently in use
        #     loss_weight = np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0]) # 3.16227766e-04
            
        loss_weighted = {
            'position': loss_weight[0]*loss['position'],
            'jerk': loss_weight[1]*loss['jerk'],
            'muscle': loss_weight[2]*loss['muscle'],
            'muscle_derivative': loss_weight[3]*loss['muscle_derivative'],
            'hidden': loss_weight[4]*loss['hidden'],
            'hidden_derivative': loss_weight[5]*loss['hidden_derivative'],
            'hidden_jerk': loss_weight[6]*loss['hidden_jerk']
        }

        overall_loss = 0
        for key in loss_weighted:
            if key=='position':
                overall_loss += th.mean(loss_weighted[key].to(self.device))
            else:
                overall_loss += loss_weighted[key]

        return overall_loss, loss_weighted


