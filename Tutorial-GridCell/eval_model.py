"""
Create ratemaps for a trained network
"""
import torch
from torch import nn
import glob
from model_lstm import GridTorch
from model_utils import get_latest_model_file
import numpy as np

import utils

from dataloading import Dataset
from torch.utils import data
import scores

N_EPOCHS = 1000
STEPS_PER_EPOCH = 20
ENV_SIZE = 2.2
BATCH_SIZE = 4000
GRAD_CLIPPING = 1e-5
SEED = 8341
N_PC = [256]
N_HDC = [12]
BOTTLENECK_DROPOUT = 0.5
WEIGHT_DECAY = 1e-5
LR = 1e-5
MOMENTUM = 0.9

#loss ops:
#loss ops:
logsoftmax = nn.LogSoftmax(dim=-1)
def cross_entropy(pred, soft_targets):
    return torch.sum(- soft_targets * logsoftmax(pred), -1)

data_params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 8, # num cpus,
          }

dataset = Dataset(batch_size=data_params['batch_size'])
data_generator = data.DataLoader(dataset, **data_params)

# Create the ensembles that provide targets during training
place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=ENV_SIZE,
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_pc=N_PC,
        pc_scale=[0.01])

head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_hdc=N_HDC,
        hdc_concentration=[20.])

ensembles = place_cell_ensembles + head_direction_ensembles

place_cell_ensembles = [ensembles[0]]
head_direction_ensembles = [ensembles[1]]

place_cell_ensembles[0].means = torch.Tensor(np.load('weights/pc_means.npy'))
place_cell_ensembles[0].variances = torch.Tensor(np.load('weights/pc_vars.npy'))

head_direction_ensembles[0].means = torch.Tensor(np.load('weights/hd_means.npy'))
head_direction_ensembles[0].kappa = torch.Tensor(np.load('weights/hd_kappa.npy'))

target_ensembles = place_cell_ensembles + head_direction_ensembles
model = GridTorch(target_ensembles, (BATCH_SIZE, 100, 3), tf_weights_loc='weights/')

for X, y in data_generator:
    break

init_pos , init_hd, ego_vel = X
target_pos, target_hd = y

initial_conds = utils.encode_initial_conditions(init_pos ,
                                                init_hd,
                                                place_cell_ensembles,
                                                head_direction_ensembles)
ensembles_targets = utils.encode_targets(target_pos,
                                        target_hd,
                                        place_cell_ensembles,
                                        head_direction_ensembles)


model.eval()

outs = model.forward(ego_vel.transpose(1, 0), initial_conds)
logits_hd, logits_pc, bottleneck_acts, lstm_states, _ = outs


acts = bottleneck_acts.transpose(1,0).detach().numpy()
pos_xy = target_pos.detach().numpy()


# Create scorer objects
starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
masks_parameters = zip(starts, ends.tolist())
scorer = scores.GridScorer(20, ((-1.1, 1.1), (-1.1, 1.1)),
                                    masks_parameters)

scoress = utils.get_scores_and_plot(scorer, pos_xy, acts, '.', 'test.pdf')
