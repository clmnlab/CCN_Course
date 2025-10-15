import motornet as mn
import torch as th
import numpy as np
from typing import Any
from typing import Union
import gym
from gym import spaces

go_time = 0.20
# go_time = 0.30

class CentreOutFF(mn.environment.Environment):
  """
     Environment for centre-out reaching task with force field.
     Training mode: A reach to a random target from a random starting position.
     Test mode: Centre-out reaches to 8 targets from the center.
  """

  def __init__(self, *args, **kwargs):
    # pass everything as-is to the parent Environment class
    super().__init__(*args, **kwargs)
    self.__name__ = "CentreOutFF"
    # check if we have K and B in kwargs
    self.K = kwargs.get('K', 150)
    self.B = kwargs.get('B', 0.5)
    

  def reset(self, *,
            seed: int | None = None,
            ff_coefficient: float = 0.,
            condition: str = 'train',
            catch_trial_perc: float = 50,
            go_cue_random = None,
            is_channel: bool = False,
            calc_endpoint_force: bool = False,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.1, 0.3),
            options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

    self._set_generator(seed)

    

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
    deterministic: bool = options.get('deterministic', False)

    self.calc_endpoint_force = calc_endpoint_force
    self.batch_size = batch_size 
    self.catch_trial_perc = catch_trial_perc
    self.ff_coefficient = ff_coefficient
    self.go_cue_range = go_cue_range # in seconds
    self.is_channel = is_channel
    self.condition = condition

    if (condition=='train'): # train net to reach to random targets

      joint_state = None

      goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      if go_cue_random is None:
        go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
      else:
        if go_cue_random:
          go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
        else:
          go_cue_time = np.tile(go_time,batch_size)

      self.go_cue_time = go_cue_time

    elif (condition=='test'): # centre-out reaches to each target

      angle_set = np.deg2rad(np.arange(0,360,45)) # 8 directions
      reps        = int(np.ceil(batch_size / len(angle_set)))
      angle       = np.tile(angle_set, reps=reps)
      batch_size  = reps * len(angle_set)

      reaching_distance = 0.10
      lb = np.array(self.effector.pos_lower_bound)
      ub = np.array(self.effector.pos_upper_bound)
      start_position = lb + (ub - lb) / 2
      
      start_position = np.array([1.047, 1.570])
      start_position = start_position.reshape(1,-1)
      start_jpv = th.from_numpy(np.concatenate([start_position, np.zeros_like(start_position)], axis=1)) # joint position and velocity
      start_cpv = self.joint2cartesian(start_jpv).numpy()
      end_cp = reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)

      goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
      goal_states = goal_states[:,:2]
      goal_states = goal_states.astype(np.float32)

      joint_state = th.from_numpy(np.tile(start_jpv,(batch_size,1)))
      goal = th.from_numpy(goal_states)
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      if go_cue_random is None:
        go_cue_time = np.tile(go_time,batch_size)
      else:
        if go_cue_random:
          go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
        else:
          go_cue_time = np.tile(go_time,batch_size)
      self.go_cue_time = go_cue_time
      
    self.effector.reset(options={"batch_size": batch_size,"joint_state": joint_state})

    self.elapsed = 0.
    action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)
  
    self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
    self.obs_buffer["action"] = [action] * self.action_frame_stacking
    
    # specify catch trials
    catch_trial = np.zeros(batch_size, dtype='float32')
    p = int(np.floor(batch_size * self.catch_trial_perc / 100))
    catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
    self.catch_trial = catch_trial

    # specify go cue time
    self.go_cue_time[self.catch_trial==1] = self.max_ep_duration
    self.go_cue = th.zeros((batch_size,1)).to(self.device)
    self.init = self.states['fingertip']

    obs = self.get_obs(deterministic=deterministic).to(self.device)

    
    
    self.endpoint_load = th.zeros((batch_size,2)).to(self.device)
    self.endpoint_force = th.zeros((batch_size,2)).to(self.device)
    
    info = {
      "states": self.states,
      "endpoint_load": self.endpoint_load,
      "endpoint_force": self.endpoint_force,
      "action": action,
      "noisy action": action,  # no noise here so it is the same
      "goal": self.goal * self.go_cue + self.init * (1-self.go_cue), # target
      }
    return obs, info


  def step(self, action, deterministic: bool = False):
    self.elapsed += self.dt

    if deterministic is False:
      noisy_action = self.apply_noise(action, noise=self.action_noise)
    else:
      noisy_action = action
    
    self.effector.step(noisy_action,endpoint_load=self.endpoint_load)


    # calculate endpoint force (External force)
    self.endpoint_load = get_endpoint_load(self)
    mask = self.elapsed < (self.go_cue_time + (self.vision_delay) * self.dt)
    self.endpoint_load[mask] = 0

    # calculate endpoint force (Internal force)
    self.endpoint_force = get_endpoint_force(self)

    # specify go cue time
    #mask = self.elapsed >= (self.go_cue_time + (self.vision_delay-1) * self.dt)
    mask = self.elapsed > (self.go_cue_time + (self.vision_delay) * self.dt)
    self.go_cue[mask] = 1

    
    obs = self.get_obs(action=noisy_action)
    # terminated = bool(self.elapsed >= self.max_ep_duration)
    is_done_scalar = self.elapsed >= self.max_ep_duration
    terminated = np.full((self.batch_size,), is_done_scalar, dtype=np.float32)
    info = {
      "states": self.states,
      "endpoint_load": self.endpoint_load,
      "endpoint_force": self.endpoint_force,
      "action": action,
      "noisy action": noisy_action,
      "goal": self.goal * self.go_cue + self.init * (1-self.go_cue),
      }
    
    return obs, terminated, info

  def get_proprioception(self):
    mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
    mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
    prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
    return self.apply_noise(prop, self.proprioception_noise)

  def get_vision(self):
    vis = self.states["fingertip"]
    return self.apply_noise(vis, self.vision_noise)

  def get_obs(self, action=None, deterministic: bool = False):
    self.update_obs_buffer(action=action)

    obs_as_list = [
      self.obs_buffer["vision"][0],  # oldest element
      self.obs_buffer["proprioception"][0],   # oldest element
      self.goal, # goal 
      # self.init, # initial position
      self.go_cue, # sepcify go cue as an input to the network
      ]
    obs = th.cat(obs_as_list, dim=-1)

    if deterministic is False:
      obs = self.apply_noise(obs, noise=self.obs_noise)
    return obs  

def get_endpoint_force(self):
  """Internal force
  """
  endpoint_force = th.zeros((self.batch_size, 2)).to(self.device)
  if self.calc_endpoint_force:
      L1 = self.skeleton.L1
      L2 = self.skeleton.L2

      pos0, pos1 = self.states['joint'][:,0], self.states['joint'][:,1]
      pos_sum = pos0 + pos1
      c1 = th.cos(pos0)
      c12 = th.cos(pos_sum)
      s1 = th.sin(pos0)
      s12 = th.sin(pos_sum)

      jacobian_11 = -L1*s1 - L2*s12
      jacobian_12 = -L2*s12
      jacobian_21 = L1*c1 + L2*c12
      jacobian_22 = L2*c12


      forces = self.states['muscle'][:, self.muscle.state_name.index('force'):self.muscle.state_name.index('force')+1, :]
      moments = self.states["geometry"][:, 2:, :]

      torque = -th.sum(forces * moments, dim=-1)

      for i in range(self.batch_size):
          jacobian_i = th.tensor([[jacobian_11[i], jacobian_12[i]], [jacobian_21[i], jacobian_22[i]]])

          endpoint_force[i] = torque[i] @ th.inverse(jacobian_i)

      return endpoint_force
  else:
      return endpoint_force

def get_endpoint_load(self):
  """External force
  """
  # Calculate endpoiont_load
  vel = self.states["cartesian"][:,2:]

  # TODO
  self.goal = self.goal.clone()
  self.init = self.init.clone()

  endpoint_load = th.zeros((self.batch_size,2)).to(self.device)

  if self.is_channel:

    X2 = self.goal
    X1 = self.init

    # vector that connect initial position to the target
    line_vector = X2 - X1

    xy = self.states["cartesian"][:,2:]
    xy = xy - X1

    projection = th.sum(line_vector * xy, axis=-1)/th.sum(line_vector * line_vector, axis=-1)
    projection = line_vector * projection[:,None]

    err = xy - projection

    projection = th.sum(line_vector * vel, axis=-1)/th.sum(line_vector * line_vector, axis=-1)
    projection = line_vector * projection[:,None]
    err_d = vel - projection
    
    F = -1*(self.B*err+self.K*err_d)
    endpoint_load = F
  else:
    FF_matvel = th.tensor([[0, 1], [-1, 0]], dtype=th.float32)
    endpoint_load = self.ff_coefficient * (vel@FF_matvel.T)
  return endpoint_load


class CentreOutFFGym(CentreOutFF):
  """
  CentreOutFFGym 환경을 초기화합니다.

  Args:
      loss_weights (dict, optional): 보상 함수 각 요소의 가중치.
      **kwargs: 부모 클래스인 CentreOutFF의 __init__에 전달될 모든 인자들.
  """
  
  def __init__(self, 
              #  loss_weights: dict = None, 
              # #  reward_scale: float = 100.0,
                reward_scale: float = 1.0,
              #  goal_bonus: float = 50.0,
                **kwargs,
                ):

      self.reward_scale = reward_scale
      # self.loss_weights = loss_weights if loss_weights is not None else np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0])
      
      self.target_radius = 0.005 # target size: 1 cm
      # self.gamma_pos_in =  0# 
      self.gamma_pos_out = 1.0
      self.gamma_ctrl_in = 1.0
      self.gamma_ctrl_out = 0.03


      # if loss_weights is None:
      #     self.loss_weights = {
      #         # 'position': 3e+3, 'jerk': 1e+2,
      #         # 'muscle': 1e-1, 'muscle_derivative': 3e-4
      #         'position': 3, 'jerk': 0,
      #         'muscle': 0.1, 'muscle_derivative': 0
      #     }
      # else:
      #     self.loss_weights = loss_weights
      
      kwargs['differentiable'] = False
      super().__init__(**kwargs)
      self.f_vec = th.tensor(self.effector.tobuild__muscle['max_isometric_force'], device=self.device) ## 1000 N 
      self.f_norm_sq = th.sum(th.square(self.f_vec))
      
  # def _reset_history(self):
  #     """보상 계산에 필요한 history 변수들을 초기화하는 헬퍼 함수입니다."""
  #     batch_size = self.batch_size if hasattr(self, 'batch_size') else 1
  #     zeros_vel = th.zeros((batch_size, self.skeleton.space_dim), device=self.device)
  #     self.last_vel, self.prev_last_vel = zeros_vel, zeros_vel
  #     self.last_force = th.zeros((batch_size, 1, self.muscle.n_muscles), device=self.device)
  #     self.last_total_cost = th.zeros(batch_size, device=self.device)
      
  # def _calculate_total_cost(self) -> th.Tensor:
  #     """
  #     🔔 [추가] 현재 상태(self.states)를 기반으로 총 비용을 계산하는 헬퍼 함수.
  #     """
  #     current_pos = self.states['fingertip'][:, :2]
  #     distance = th.linalg.norm(current_pos - self.goal, ord=1, dim=1) # L1 Norm
  #     # 목표 반경 내/외부에 따라 가중치(gamma)를 다르게 설정
  #     is_inside = distance < self.target_radius
  #     gamma_pos = th.where(is_inside, 0, self.gamma_pos_out) # inside the target, pos_cost = 0
  #     gamma_ctrl = th.where(is_inside, self.gamma_ctrl_in, self.gamma_ctrl_out)

  #     # 1. 위치 비용 계산
  #     cost_pos = gamma_pos * distance        
      
  #     cost_c = gamma_ctrl * self.states['muscle'][:,0,:]
  #     current_vel = self.states['cartesian'][:, 2:]
  #     jerk = current_vel - 2 * self.last_vel + self.prev_last_vel
  #     cost_jerk = th.mean(th.square(jerk), dim=1)
  #     muscle_force = self.states['muscle'][:, -1:, :]  ## last element is all_force!!
  #     cost_muscle = th.mean(th.square(muscle_force), dim=2).squeeze()
  #     muscle_force_derivative = muscle_force - self.last_force
  #     cost_muscle_derivative = th.mean(th.square(muscle_force_derivative), dim=2).squeeze()
  #     # total_cost = cost_pos + cost_ctrl
      
  #     # total_cost = (self.loss_weights['position'] * cost_pos +
  #     #               self.loss_weights['jerk'] * cost_jerk +
  #     #               self.loss_weights['muscle'] * cost_muscle +
  #     #               self.loss_weights['muscle_derivative'] * cost_muscle_derivative)
  #     return total_cost
    
  def get_obs(self, action=None, deterministic: bool = False) -> th.Tensor:
      """
      [TypeError 해결을 위한 오버라이드]
      부모 클래스의 get_obs를 호출하기 전, `self.goal`이 Numpy 배열이면
      다시 PyTorch 텐서로 변환하여 `th.cat` 오류를 방지합니다.
      
      [AttributeError 해결을 위한 수정]
      이 메소드는 이제 항상 torch.Tensor를 반환합니다. Numpy 변환은 reset/step에서 처리합니다.
      """
      if hasattr(self, 'goal') and self.goal is not None and isinstance(self.goal, np.ndarray):
          self.goal = th.from_numpy(self.goal).to(self.device).float()
      
      # 부모의 get_obs는 torch.Tensor를 반환
      return super().get_obs(action=action, deterministic=deterministic)
    
  def reset(self, *,
            seed: int | None = None,
            ff_coefficient: float = 0.,
            condition: str = 'train',
            catch_trial_perc: float = 50,
            go_cue_random = None,
            is_channel: bool = False,
            calc_endpoint_force: bool = False,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.1, 0.3),
            options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
            """ Gymnasium API 표준에 맞는 reset 메소드입니다. """
            
            obs, info = super().reset(seed=seed, condition = condition, 
                                      ff_coefficient = ff_coefficient,
                                      catch_trial_perc = catch_trial_perc,
                                      go_cue_random = go_cue_random,
                                      is_channel = is_channel,
                                      calc_endpoint_force = calc_endpoint_force,
                                      go_cue_range = go_cue_range,
                                      options=options)
            # self._reset_history()

            # 🔔 [추가] 에피소드 시작 시의 초기 비용을 계산하고 저장합니다.
            # initial_cost = self._calculate_total_cost()
            # self.last_total_cost = initial_cost

            obs_np = obs.cpu().numpy()
            obs_squeezed = np.squeeze(obs_np)
    
            return obs_squeezed.astype(np.float32), info

  def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
      """ Gymnasium API 표준에 맞는 step 메소드입니다. """
      # 부모 클래스(CentreOutFF)는 (obs, terminated, info) 3-튜플을 반환. obs는 Tensor.
      # 부모 클래스의 step은 (obs_batch, terminated, info)를 반환. obs_batch는 Tensor.
      action_tensor = th.from_numpy(action).float().to(self.device)
      obs_batch, terminated, info = super().step(action_tensor)
      # --- 보상 계산 로직 변경 ---
      current_pos = self.states['fingertip'][:, :2]
      effective_goal = info['goal']
      # effective_goal = self.goal
      # Use info['goal'] instead of self.goal to account for go_cue
      # L1norm_error = th.linalg.norm(current_pos - effective_goal, ord=1, dim=1) 
      distance = th.linalg.norm(current_pos - effective_goal, ord=2, dim=1) 
      is_inside = distance < self.target_radius
      is_early_start = (distance >= self.target_radius) & (self.elapsed < (self.go_cue_time + (self.vision_delay) * self.dt))
      # gamma_pos = th.where(
      #     is_early_start.bool(),      # 첫 번째 조건
      #     5.0,                # is_early_start가 참일 때의 값
      #     th.where(is_inside.bool(), 0.0, self.gamma_pos_out) # is_early_start가 거짓일 때, 다시 if/else 검사
      # )
      gamma_pos = 1.0
      
      gamma_ctrl = th.where(is_inside, self.gamma_ctrl_in, self.gamma_ctrl_out)
      cost_pos = gamma_pos * th.sum(th.abs(current_pos - effective_goal), dim=1)
      cost_control = gamma_ctrl * th.sum(action_tensor * self.f_vec, dim=1)**2 / self.f_norm_sq
      # cost_control = gamma_ctrl * th.sum(th.square(action_tensor), dim=1)
      current_cost = cost_pos + cost_control  
      # print(cost_control)
      # current_cost = cost_pos
      reward = -current_cost * self.reward_scale
      # print(cost_pos)
      # action_tensor가 근육 활성화(u_t)에 해당합니다.
      # control_term = th.sum(action_tensor * self.f_vec), dim=1) / self.f_norm_sq
              
      # 1. 잠재력 기반 보상: 이전 스텝 대비 비용 감소량을 보상으로 설정
      # reward = (self.last_total_cost - current_cost) * self.reward_scale
      # 2. negative loss 보상: 현재 비용을 보상으로 설정
      # # 다음 스텝을 위해 History 변수 업데이트
      # self.last_total_cost = current_cost
      # self.prev_last_vel = self.last_vel.clone()
      # self.last_vel = self.states['cartesian'][:, 2:].clone()
      # self.last_force = self.states['muscle'][:, 4:5, :].clone()
      
      obs_np = obs_batch.cpu().numpy()
      obs_squeezed = np.squeeze(obs_np)
      
      return obs_squeezed.astype(np.float32), reward.cpu().numpy(), terminated, False, info

  # def render(self, mode='human'):
  #     if mode == 'human': self.plot()


  def close(self): pass
