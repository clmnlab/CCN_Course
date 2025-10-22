import numpy as np
import torch as th
import motornet as mn

## Utility functions for simulating rollouts and collecting data
def run_rollout(env,agent,batch_size=1, catch_trial_perc=50,condition='train', 
                ff_coefficient=0., is_channel=False,detach=False,calc_endpoint_force=False, go_cue_random=None,
                disturb_hidden=False, t_disturb_hidden=0.15, d_hidden=None, seed=None):
  
    device = agent.device
    h = agent.network.init_hidden(batch_size = batch_size).to(device)
    obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, 
                          is_channel=is_channel,calc_endpoint_force=calc_endpoint_force, go_cue_random=go_cue_random, seed = seed)

    if type(obs) is np.ndarray:
        obs = th.tensor(obs, dtype=th.float32)
    obs = obs.to(device)
    # terminated = False
    terminated = np.zeros(batch_size)
    # Initialize a dictionary to store lists
    data = {
        'xy': [],
        'tg': [],
        'vel': [],
        'all_action': [],
        'all_hidden': [],
        'all_muscle': [],
        'all_force': [],
        'endpoint_load': [],
        'endpoint_force': []
    }

    while not terminated.any():
        # add disturn hidden activity
        if disturb_hidden:
            if np.abs(env.elapsed-t_disturb_hidden)<1e-3:
                #print('DONE!!!!')
                dh = d_hidden.repeat(1,batch_size,1)
                h += dh


        # action, h = agent.network(obs,h)
        output = agent.network(obs,h)
        if len(output)==3:
            dist, _, h = output
            action = th.sigmoid(dist.sample()).cpu().numpy()
        elif len(output)==2:
            action, h = output
        
        # obs, terminated, info = env.step(action=action.to('cpu'))
        output = env.step(action=action)
        if len(output) == 3:
            obs, terminated, info = output
        elif len(output) == 5:
            obs, _, terminated, _, info = output
            obs = th.tensor(obs, dtype=th.float32)
        obs = obs.to(device)
        data['all_hidden'].append(h[0, :, None, :])
        data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])
        data['all_force'].append(info['states']['muscle'][:, -1, None, :])
        data['xy'].append(info["states"]["fingertip"][:, None, :])
        data['tg'].append(info["goal"][:, None, :])
        data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
        data['all_action'].append(action[:, None, :])
        data['endpoint_load'].append(info['endpoint_load'][:, None, :])
        data['endpoint_force'].append(info['endpoint_force'][:, None, :])
    # # # Concatenate the lists
    if type(data['all_action'][0]) is np.ndarray:
        for key in data:
            # data[key] = th.tensor(np.concatenate(data[key], axis=1), dtype=th.float32).to(device)
            tensors_on_device = [th.as_tensor(arr, dtype=th.float32).to(device) for arr in data[key]]
            data[key] = th.cat(tensors_on_device, axis=1)
    else:
        for key in data:
            data[key] = th.cat(data[key], axis=1).to(device)
    # for key in data:
    #     data[key] = th.cat(data[key], axis=1)

    if detach:
        # Detach tensors if needed
        for key in data:
            data[key] = th.detach(data[key])

    return data


plotor = mn.plotor.plot_pos_over_time

def plot_simulations2(data):
    xy = data['xy'].detach().cpu().numpy()
    tg = data['tg'].detach().cpu().numpy()
    target_x = tg[:, -1, 0]
    target_y = tg[:, -1, 1]

    plt.figure(figsize=(10,3))

    # plt.subplot(1,2,1)
    # plt.ylim([-.4, 0.6])
    # plt.xlim([-0.6, 0.6])
    plotor(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y)

    # plt.subplot(1,2,2)
    # # plt.ylim([-2, 2])
    # # plt.xlim([-2, 2])
    # plotor(axis=plt.gca(), cart_results=xy - target_xy)
    # plt.axhline(0, c="grey")
    # plt.axvline(0, c="grey")
    # plt.xlabel("X distance to target")
    # plt.ylabel("Y distance to target")
    plt.show()
