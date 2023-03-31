import gymnasium as gym
from gymnasium import spaces
from robomaster import robot
from alrd.subscriber import RobotSubAbs
import robomaster.robot
import numpy as np
import time
import logging
from alrd.environment.names import *
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
    def __init__(self, robot: robomaster.robot.Robot, subscriber: RobotSubAbs) -> None:
        self.robot = robot
        self.subscriber = subscriber
        self.data_log = []
        self.last_reset_time = None
        self._init_obs_space()
        self._init_action_space()
    
    def _init_obs_space(self):
        self.observation_space = spaces.Dict({
            self.POSITION: spaces.Box(-np.inf, np.inf, shape=(2,)),
            self.ANGLE: spaces.Box(-180.,180.),
            self.VELOCITY: spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]),np.array([MAX_X_VEL, MAX_Y_VEL])),
            self.ANGULAR_V: spaces.Box(MIN_A_VEL, MAX_A_VEL),
            self.ACCELERATION: spaces.Box(-np.inf, np.inf, shape=(2,)),
            self.MOTOR_SP: spaces.Box(MIN_MOTOR_VEL, MAX_MOTOR_VEL, shape=(4,)),
            self.ARM_POSITION: spaces.Box(np.array([MIN_ARM_X, MIN_ARM_Y]), np.array([MAX_ARM_X, MAX_ARM_Y]))
        })

    @abstractmethod
    def _init_action_space(self):
        pass

    def _subscriber_state_to_obs(self, subscriber_state):
        chassis_state = subscriber_state['chassis']
        arm_state = subscriber_state['arm']
        return {
            self.POSITION: np.array([chassis_state['position']['x'], chassis_state['position']['y']]),
            self.ANGLE: np.array([-chassis_state['attitude']['yaw']]),
            self.VELOCITY: np.array([chassis_state['velocity']['vgx'], chassis_state['velocity']['vgy']]),
            self.ANGULAR_V: np.array([chassis_state['imu']['gyro_z']]), # TODO: check gyro_z readings
            self.ACCELERATION: np.array([chassis_state['imu']['acc_x'], chassis_state['imu']['acc_y']]),
            self.MOTOR_SP: np.array([chassis_state['esc']['speed']]),
            self.ARM_POSITION: np.array([arm_state['x'], arm_state['y']])
        }

    def _get_obs(self):
        subscriber_state = self.subscriber.get_state()
        return self._subscriber_state_to_obs(subscriber_state)
    
    @abstractmethod
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
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
        truncated = False
        action_valid = self.is_action_valid(action)
        if not action_valid:
            self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
            truncated = True
        else:
            # apply action
            self._apply_action(action)
        # get new observation. The method waits if step is called before the subscriber has received new information
        obs = self._get_obs()
        reward = self.get_reward(obs)
        terminated = self.is_terminating(obs)
        info = {'action_valid': action_valid}
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
        self.subscriber.unsubscribe()
        self.robot.close()

class VelocityControlEnv(AbsEnv):
    def __init__(self, robot, subscriber) -> None:
        super().__init__(robot, subscriber)
        
    def _init_action_space(self):
        self.action_space = spaces.Dict({
            self.VELOCITY: spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]),np.array([MAX_X_VEL, MAX_Y_VEL])),
            self.ANGULAR_V: spaces.Box(MIN_A_VEL, MAX_A_VEL)}) 

    def is_action_valid(self, action):
        try:
            return (self.VELOCITY in action and self.ANGULAR_V in action and
                    action[self.VELOCITY].shape == self.action_space[self.VELOCITY].shape and 
                    action[self.ANGULAR_V].shape == self.action_space[self.ANGULAR_V].shape)
        except:
            logger.error(traceback.format_exc())
            return False
    
    def _apply_action(self, action):
        vel = action[self.VELOCITY]
        ang_vel = action[self.ANGULAR_V]
        self.robot.chassis.drive_speed(vel[0].item(), vel[1].item(), ang_vel.item())