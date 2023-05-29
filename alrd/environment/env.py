import gym
from gym import spaces
from robomaster import robot
from alrd.subscriber import RobotSubAbs, ChassisSub
import robomaster.robot
import numpy as np
import time
import logging
from alrd.environment.names import *
from alrd.utils import rotate_2d_vector, sleep_ms
from abc import ABC, abstractmethod
import traceback


logger = logging.getLogger(__name__)

def init_robot(conn_type='ap'):
    _robot = robot.Robot()
    _robot.initialize(conn_type=conn_type)
    _robot.reset()
    return _robot

class AbsEnv(gym.Env, ABC):
    POSITION = 'position'
    ANGLE = 'angle'
    VELOCITY = 'velocity'
    ANGULAR_V = 'angular_velocity'
    ACCELERATION = 'acceleration'
    MOTOR_SP = 'motor_speed'
    ARM_POSITION = 'arm_position'
    metadata = {}
    def __init__(self, robot: robomaster.robot.Robot, subscriber: ChassisSub, transforms=None) -> None:
        self.robot = robot
        self.subscriber = subscriber
        self.data_log = []
        self.transforms = transforms if transforms is not None else [] # TODO transforms could be wrappers
        self._init_obs_space()
        self._init_action_space()
        self.period = 1./subscriber.freq
        self.last_action_time = None
    
    def _init_obs_space(self):
        self.observation_space = spaces.Box(np.array([MIN_X, MIN_Y, -180, MIN_X_VEL, MIN_Y_VEL, MIN_A_VEL]), 
                                            np.array([MAX_X, MAX_Y, 180, MAX_X_VEL, MAX_Y_VEL, MAX_A_VEL]))

    @abstractmethod
    def _init_action_space(self):
        pass

    def _subscriber_state_to_obs(self, chassis_state):
        return np.array([
            chassis_state['position']['x'], chassis_state['position']['y'],     # position
            chassis_state['attitude']['yaw'],                                   # angle
            chassis_state['velocity']['vgx'], chassis_state['velocity']['vgy'], # velocity
            chassis_state['imu']['gyro_z']                                      # angular velocity
        ])

    def _get_obs(self, blocking=False):
        subscriber_state = self.subscriber.get_state(blocking=blocking)
        obs = self._subscriber_state_to_obs(subscriber_state)
        for transform in self.transforms:
            obs = transform(obs)
        return obs
    
    @abstractmethod
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action_time = None
    
    def get_reward(self, obs):
        return 0.
    
    def is_terminating(self, obs):
        return False
    
    @abstractmethod
    def is_action_valid(self, action):
        pass

    @abstractmethod
    def _apply_action(self, action):
        pass
    
    def step(self, action):
        """
        The method will wait until the period has passed after applying the action before returning the new observation.
        """
        start = time.time()
        truncated = False
        action_valid = self.is_action_valid(action)
        if self.last_action_time is not None:
            elapsed = time.time() - self.last_action_time
        else:
            elapsed = 0
        self.last_action_time = time.time()
        if not action_valid:
            self.robot.chassis.drive_wheels(timeout=0.01)
            truncated = True
        else:
                
            self._apply_action(action)
        time.sleep(self.period)
        # get new observation after period has passed
        obs = self._get_obs()
        reward = self.get_reward(obs)
        terminated = self.is_terminating(obs)
        info = {'action_valid': action_valid, 'step_time': time.time()-start, 'elapsed_last_step': elapsed}
        return obs, reward, terminated, truncated, info
    
    def stop_robot(self):
        self.robot.chassis.drive_wheels(0,0,0,0)
    
class VelocityControlEnv(AbsEnv):
    def __init__(self, robot, subscriber, transforms=None) -> None:
        super().__init__(robot, subscriber, transforms=transforms)
        
    def _init_action_space(self):
        self.action_space = spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL, MIN_A_VEL]), np.array([MAX_X_VEL, MAX_Y_VEL, MAX_A_VEL]))

    def is_action_valid(self, action):
        try:
            return action.shape == self.action_space.shape
        except:
            logger.error(traceback.format_exc())
            return False
    
    def _apply_action(self, action):
        if np.allclose(action, 0, atol=1e-4):
            self.robot.chassis.drive_wheels()
            return
        vel = action[:2]
        ang_vel = action[2]
        self.robot.chassis.drive_speed(vel[0].item(), vel[1].item(), ang_vel.item())