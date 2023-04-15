import gymnasium as gym
from gymnasium import spaces
from robomaster import robot
from alrd.subscriber import RobotSubBox
from alrd.environment.env import VelocityControlEnv, init_robot
import numpy as np
import time
import logging
from alrd.environment.names import *

logger = logging.getLogger(__name__)

def create_robomaster_env(x_low, x_high, y_low, y_high, freq=50):
    _robot = init_robot()
    return RobomasterEnv(_robot, x_low, x_high, y_low, y_high, freq=freq)

class RobomasterEnv(VelocityControlEnv):
    OOB = 'oob'
    metadata = {}
    def __init__(self, robot, x_low, x_high, y_low, y_high, freq=20) -> None:
        super().__init__(robot, RobotSubBox(robot, x_low, x_high, y_low, y_high, freq=freq))
        self.data_log = []
        self.last_reset_time = None
        
    def _init_obs_space(self):
        super()._init_obs_space()
        self.observation_space[self.OOB] = spaces.Discrete(2)

    def _subscriber_state_to_obs(self, subscriber_state):
        obs = super()._subscriber_state_to_obs(subscriber_state) 
        obs[self.OOB] = subscriber_state['chassis']['position']['oob']
        return obs
    
    def get_subscriber_log(self):
        return self.subscriber.to_dict()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
        self.data_log.append(self.subscriber.to_dict())
        # go to origin
        logger.info('getting state for reset')
        obs = self._get_obs()
        angle = obs[self.ANGLE].item()
        logger.info('moving %s degrees' % (-angle))
        self.robot.chassis.move(z=-angle, z_speed=60.).wait_for_completed(timeout=5.)
        obs = self._get_obs()
        position = obs[self.POSITION]
        while np.abs(position).max() > 5e-2:
            xy_move = np.clip(-position, -MAX_MOVE_POS, MAX_MOVE_POS)
            logger.info('moving %s' % (xy_move))
            self.robot.chassis.move(xy_move[0].item(), xy_move[1].item(), xy_speed=1.).wait_for_completed(timeout=5.)
            obs = self._get_obs()
            position = obs[self.POSITION]
        logger.info('reset to %s'%position)
        self.robot.robotic_arm.recenter().wait_for_completed(timeout=2.)
        obs = self._get_obs()
        self.last_reset_time = time.time()
        self.subscriber.reset()
        return obs, None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        out_of_bounds = obs[self.OOB]
        if out_of_bounds:
            truncated = True
        info['out_of_bounds'] = out_of_bounds
        return obs, reward, terminated, truncated, info