import gym
from gym import spaces
import numpy as np
from alrd.environment.spot.spot import SpotEnvironmentConfig
from alrd.environment.spot.spot2d import Spot2DReward, MIN_X, MIN_Y, MAX_X, MAX_Y
from alrd.utils.utils import Frame2D
import math
from alrd.environment.spot.utils import MAX_SPEED, MAX_ANGULAR_SPEED, get_hitbox
from abc import ABC, abstractmethod

MARGIN = 0.2

class Spot2DBaseSim(gym.Env, ABC):
    obs_shape = (7,)
    action_shape = (3,)

    def __init__(self, config: SpotEnvironmentConfig, action_cost=0.0, velocity_cost=0.0):
        self.config = config
        self.action_cost = action_cost
        self.velocity_cost = velocity_cost
        self.goal_frame = None
        self.state = None
        self.reward = Spot2DReward.create(action_coeff=action_cost, velocity_coeff=velocity_cost)
        self.body_start_frame = Frame2D(0, 0, 0)
        self.observation_space = spaces.Box(low=np.array([MIN_X, MIN_Y, -1, -1,-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                            high=np.array([MAX_X, MAX_Y, 1, 1, MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.action_space = spaces.Box(low=np.array([-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                        high=np.array([MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.reset()
    
    @property
    def default_reset(self):
        return (self.config.start_x, self.config.start_y, self.config.start_angle)

    def start(self):
        pass

    def _get_obs(self):
        return np.array([*self.state[:2], np.cos(self.state[2]), np.sin(self.state[2]), *self.state[3:]])

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        super().reset(seed=seed, options=options)
        pose = options.get('pose', (self.config.start_x, self.config.start_y, self.config.start_angle))
        goal = options.get('goal', (self.config.start_x, self.config.start_y, self.config.start_angle))
        self.goal_frame = Frame2D(*goal)
        pose = self.goal_frame.transform_pose(*pose)
        self.state = np.array([*pose, 0, 0, 0])
        obs = self._get_obs()
        return obs, {}

    def is_in_bounds(self, state) -> bool:
        x, y, theta = state[:3]
        xg, yg, thetag = self.goal_frame.inverse_pose(x, y, theta)
        box = get_hitbox(xg, yg, thetag)
        min_x, min_y = np.min(box, axis=0)
        max_x, max_y = np.max(box, axis=0)
        return min_x > self.config.min_x + MARGIN and max_x < self.config.max_x - MARGIN and \
                min_y > self.config.min_y + MARGIN and max_y < self.config.max_y - MARGIN
    
    @abstractmethod
    def _update_state(self, action):
        pass

    def step(self, action):
        self._update_state(action)
        obs = self._get_obs()
        reward = self.reward.predict(obs, action)
        truncate = not self.is_in_bounds(self.state)
        info = {}
        info['dist'] = np.linalg.norm(obs[:2]).item()
        info['angle'] = np.abs(np.arctan2(obs[3], obs[2])).item()
        return obs, reward, False, truncate, info
    
    def stop_robot(self):
        pass


class Spot2DEnvSim(Spot2DBaseSim):
    def __init__(self, config: SpotEnvironmentConfig, freq, **kwargs):
        super().__init__(config, **kwargs)
        self.freq = freq
        self.dt = 1 / freq

    def _update_state(self, action):
        action_global = Frame2D(*self.state[:3]).transform_direction(action[:2])
        x = self.state[0] + self.state[3] * self.dt + (self.state[3] - action_global[0]) * self.dt / 2
        y = self.state[1] + self.state[4] * self.dt + (self.state[4] - action_global[1]) * self.dt / 2
        theta = self.state[2] + self.state[5] * self.dt + (self.state[5] - action[2]) * self.dt / 2
        theta = math.remainder(theta, math.tau)
        vx = action_global[0]
        vy = action_global[1]
        vtheta = action[2]
        self.state = np.array([x, y, theta, vx, vy, vtheta])