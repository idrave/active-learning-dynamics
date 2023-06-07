from typing import Tuple
import gym
from gym import spaces
import logging
import time
from alrd.environment.env import AbsEnv
from alrd.utils import rotate_2d_vector
from alrd.subscriber import ChassisSub
from gym.core import Env
from alrd.environment.names import *
logger = logging.getLogger(__name__)

class CosSinObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = spaces.Box(np.array([MIN_X, MIN_Y, -1.,-1., MIN_X_VEL, MIN_Y_VEL, MIN_A_VEL]), 
                                            np.array([MAX_X, MAX_Y, 1.,1., MAX_X_VEL, MAX_Y_VEL, MAX_A_VEL]))
                                        
    def observation(self, observation):
        x, y, angle, x_vel, y_vel, a_vel = observation
        return np.array([x, y, np.cos(angle), np.sin(angle), x_vel, y_vel, a_vel])

class GlobalFrameActionWrapper(gym.ActionWrapper):
    def __init__(self, env: AbsEnv):
        super().__init__(env)
    
    def action(self, action):
        action[:2] = rotate_2d_vector(action[:2], -self.env._last_obs[2])
        return action

class RemoveAngleActionWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Box(env.action_space.low[:2], env.action_space.high[:2])
    
    def action(self, action):
        return np.array([action[0], action[1], 0.])

class KeepObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env, keep):
        super().__init__(env)
        self.keep = keep
        self.observation_space = spaces.Box(env.observation_space.low[keep], env.observation_space.high[keep])
    
    def observation(self, observation):
        return observation[self.keep]

class RemoveAngleWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]), np.array([MAX_X_VEL, MAX_Y_VEL]))
        self.observation_space = spaces.Box(np.array([MIN_X, MIN_Y, MIN_X_VEL, MIN_Y_VEL]), 
                                            np.array([MAX_X, MAX_Y, MAX_X_VEL, MAX_Y_VEL]))
    
    def observation(self, observation):
        x, y, angle, x_vel, y_vel, a_vel = observation
        return np.array([x, y, x_vel, y_vel])
    
    def step(self, action):
        x_vel, y_vel = action
        return super().step(np.array([x_vel, y_vel, 0.]))

class RepeatActionWrapper(gym.Wrapper):
    def __init__(self, env: Env, repeat):
        super().__init__(env)
        self.observation_space = spaces.Box(np.tile(env.observation_space.low, repeat), np.tile(env.observation_space.high, repeat))
        self.repeat = repeat

    def step(self, action):
        obs_list = []
        reward_list = []
        infos = {}
        for i in range(self.repeat):
            obs, reward, terminated, truncated, info = super().step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            infos[i] = info
            done = terminated or truncated
            if done:
                break
        return np.concatenate(obs_list), np.mean(reward_list), terminated, truncated, infos
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return np.tile(obs, self.repeat), [info] * self.repeat

class UnderSampleWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, sample_rate: int):
        super().__init__(env)
        self.sample_rate = sample_rate

    def step(self, action):
        obs_list = []
        reward_list = []
        info_list = []
        for i in range(self.sample_rate):
            obs, reward, terminated, truncated, info = super().step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            info_list.append(info)
            done = terminated or truncated
            if done:
                break
        return np.median(obs_list, axis=0), np.median(reward_list), terminated, truncated, info_list