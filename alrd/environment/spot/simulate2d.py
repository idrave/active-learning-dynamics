import gym
from gym import spaces
import numpy as np
from alrd.environment.spot.spot import SpotEnvironmentConfig
from alrd.environment.spot.spot2d import Spot2DReward, MIN_X, MIN_Y, MAX_X, MAX_Y
from alrd.utils.utils import Frame2D
import math
from alrd.environment.spot.utils import MAX_SPEED, MAX_ANGULAR_SPEED

class Spot2DEnvSim(gym.Env):
    obs_shape = (7,)
    action_shape = (3,)

    def __init__(self, config: SpotEnvironmentConfig, freq, action_cost=0.0, velocity_cost=0.0):
        self.config = config
        self.freq = freq
        self.dt = 1 / freq
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
    
    def start(self):
        pass

    def _get_obs(self):
        return np.array([*self.state[:2], np.cos(self.state[2]), np.sin(self.state[2]), *self.state[3:]])

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        pose = options.get('pose', (self.config.start_x, self.config.start_y, self.config.start_angle))
        goal = options.get('goal', (self.config.start_x, self.config.start_y, self.config.start_angle))
        self.goal_frame = Frame2D(*goal)
        pose = self.goal_frame.transform_pose(*pose)
        self.state = np.array([*pose, 0, 0, 0])
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        action_global = Frame2D(*self.state[:3]).transform_direction(action[:2])
        x = self.state[0] + self.state[3] * self.dt + (self.state[3] - action_global[0]) * self.dt / 2
        y = self.state[1] + self.state[4] * self.dt + (self.state[4] - action_global[1]) * self.dt / 2
        theta = self.state[2] + self.state[5] * self.dt + (self.state[5] - action[2]) * self.dt / 2
        theta = math.remainder(theta, math.tau)
        vx = action_global[0]
        vy = action_global[1]
        vtheta = action[2]
        self.state = np.array([x, y, theta, vx, vy, vtheta])
        obs = self._get_obs()
        reward = self.reward.predict(obs, action)
        position = self.goal_frame.inverse([x, y])
        truncate = position[0] < self.config.min_x or position[0] > self.config.max_x \
                    or position[1] < self.config.min_y or position[1] > self.config.max_y
        info = {}
        info['dist'] = np.linalg.norm(obs[:2])
        info['angle'] = np.abs(np.arctan2(obs[3], obs[2]))
        return obs, reward, False, truncate, info
    
    def stop_robot(self):
        pass
