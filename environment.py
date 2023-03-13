import gymnasium as gym
from gymnasium import spaces
from subscriber import RobotSub
import numpy as np
from robomaster import robot
import time
import logging

MAX_MOVE_POS = 5.
MIN_X_VEL = -3.5
MAX_X_VEL = -3.5
MIN_Y_VEL = 3.5
MAX_Y_VEL = 3.5
MIN_A_VEL = -600.
MAX_A_VEL = 600.
MIN_MOTOR_VEL = -8192.
MAX_MOTOR_VEL = 8191.
MIN_ARM_X = -np.inf
MIN_ARM_Y = -np.inf
MAX_ARM_X = np.inf
MAX_ARM_Y = np.inf # TODO: check arm coordinate range
POSITION = 'position'
ANGLE = 'angle'
VELOCITY = 'velocity'
ANGULAR_V = 'angular_velocity'
ACCELERATION = 'acceleration'
MOTOR_SP = 'motor_speed'
ARM_POSITION = 'arm_position'
OOB = 'oob'

logger = logging.getLogger(__name__)

class RobomasterEnv(gym.Env):
    metadata = {}
    def __init__(self, robot: robot.Robot, x_low, x_high, y_low, y_high, freq=20) -> None:
        self.robot = robot
        self.subscriber = RobotSub(robot, freq=freq)
        self.pos_low = (x_low, y_low)
        self.pos_high = (x_high, y_high)
        self.observation_space = spaces.Dict({
            POSITION: spaces.Box(np.array([x_low,y_low], np.array([x_high,y_high]))),
            ANGLE: spaces.Box(-180.,-180.),
            VELOCITY: spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]),np.array(MAX_X_VEL, MAX_Y_VEL)),
            ANGULAR_V: spaces.Box(MIN_A_VEL, MAX_A_VEL),
            ACCELERATION: spaces.Box(-np.inf, np.inf, shape=(2,)),
            MOTOR_SP: spaces.Box(MIN_MOTOR_VEL, MAX_MOTOR_VEL, shape=(4,)),
            ARM_POSITION: spaces.Box(np.array([MIN_ARM_X, MIN_ARM_Y]), np.array([MAX_ARM_X, MAX_ARM_Y]))})
    
        self.action_space = spaces.Dict({
            VELOCITY: spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]),np.array(MAX_X_VEL, MAX_Y_VEL)),
            ANGULAR_V: spaces.Box(MIN_A_VEL, MAX_A_VEL)}) # TODO: add arm movement
        self.data_log = []
        self.finished = False
        
    def _get_obs(self):
        subscriber_state = self.subscriber.get_state()
        chassis_state = subscriber_state['chassis']
        arm_state = subscriber_state['arm']
        return {
            POSITION: np.array([chassis_state['position']['x'], chassis_state['position']['y']]),
            ANGLE: np.array([chassis_state['attitude']['yaw']]),
            VELOCITY: np.array([chassis_state['velocity']['vgx']], chassis_state['velocity']['vgy']),
            ANGULAR_V: np.array([chassis_state['imu']['gyro_z']]), # TODO: check gyro_z readings
            ACCELERATION: np.array([chassis_state['imu']['acc_x'], chassis_state['imu']['acc_y']]),
            MOTOR_SP: np.array([chassis_state['esc']['speed']]),
            ARM_POSITION: np.array([arm_state['x'], arm_state['y']])
        }

    def reset(self):
        self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
        self.data_log.append(self.subscriber.to_dict())
        self.subscriber.reset()
        # go to origin
        obs = self._get_obs()
        position = obs[POSITION]
        while position.max() > MAX_MOVE_POS:
            self.robot.chassis.move(*np.clip(-position, -MAX_MOVE_POS, MAX_MOVE_POS), xy_speed=1.).wait_for_completed()
            obs = self._get_obs()
            position = obs[POSITION]
        z = obs[ANGLE]
        self.robot.chassis.move(*(-position), z, xy_speed=1., z_speed=180).wait_for_completed()
        obs = self._get_obs()
        self.finished = False
        return obs, None
    
    def get_reward(self, obs):
        return 0.
    
    def is_terminating(self, obs):
        return False
    
    def is_action_valid(self, action):
        try:
            return (VELOCITY in action and ANGULAR_V in action and
                    action[VELOCITY].shape == self.action_space[VELOCITY].shape and 
                    action[ANGULAR_V].shape == self.action_space[ANGULAR_V].shape)
        except:
            return False
    
    def clip(self, action):
        new_action = {}
        new_action[VELOCITY] = np.clip(action[VELOCITY], self.action_space[VELOCITY].low, self.action_space[VELOCITY].high)
        new_action[ANGULAR_V] = np.clip(action[ANGULAR_V], self.action_space[ANGULAR_V].low, self.action_space[ANGULAR_V].high)
        return new_action

    def step(self, action):
        if self.finished:
            raise Exception('Episode has finished. Call reset() to start a new episode.')
        truncated = False
        action_valid = self.is_action_valid(action)
        if action_valid:
            self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
            truncated = True
        else:
            # apply action
            action = self.clip(action)
            vel = action[VELOCITY]
            ang_vel = action[ANGULAR_V]
            self.robot.chassis.drive_speed(*vel, ang_vel)
        # get new observation. The method waits for new information if step is called before the sampling period passed
        obs = self._get_obs()[POSITION]
        reward = self.get_reward(obs)
        terminated = self.is_terminating(obs)
        out_of_bounds = obs[OOB]
        if out_of_bounds:
            truncated = True
        info = {'out_of_bounds': out_of_bounds, 'action_valid': action_valid}
        self.finished = terminated or truncated
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        self.subscriber.unsubscribe()