import gymnasium as gym
from gymnasium import spaces
from robomaster import robot
from subscriber import RobotSub
import numpy as np
from robomaster import robot
import time
import logging

MAX_MOVE_POS = 5.
MIN_X_VEL = -3.5
MAX_X_VEL = 3.5
MIN_Y_VEL = -3.5
MAX_Y_VEL = 3.5
MIN_A_VEL = -600.
MAX_A_VEL = 600.
MIN_MOTOR_VEL = -8192.
MAX_MOTOR_VEL = 8191.
MIN_ARM_X = -np.inf
MIN_ARM_Y = -np.inf
MAX_ARM_X = -30.
MAX_ARM_Y = 30. # TODO: check arm coordinate range

logger = logging.getLogger(__name__)

class RobomasterEnv(gym.Env):
    POSITION = 'position'
    ANGLE = 'angle'
    VELOCITY = 'velocity'
    ANGULAR_V = 'angular_velocity'
    ACCELERATION = 'acceleration'
    MOTOR_SP = 'motor_speed'
    ARM_POSITION = 'arm_position'
    OOB = 'oob'
    metadata = {}
    def __init__(self, x_low, x_high, y_low, y_high, freq=20) -> None:
        self.robot = robot.Robot()
        self.robot.initialize(conn_type='ap')
        self.robot.reset()
        self.subscriber = RobotSub(self.robot, x_low, x_high, y_low, y_high, freq=freq)
        self.pos_low = (x_low, y_low)
        self.pos_high = (x_high, y_high)
        self.observation_space = spaces.Dict({
            self.POSITION: spaces.Box(np.array([x_low,y_low]), np.array([x_high,y_high])),
            self.OOB: spaces.Discrete(2),
            self.ANGLE: spaces.Box(-180.,180.),
            self.VELOCITY: spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]),np.array([MAX_X_VEL, MAX_Y_VEL])),
            self.ANGULAR_V: spaces.Box(MIN_A_VEL, MAX_A_VEL),
            self.ACCELERATION: spaces.Box(-np.inf, np.inf, shape=(2,)),
            self.MOTOR_SP: spaces.Box(MIN_MOTOR_VEL, MAX_MOTOR_VEL, shape=(4,)),
            self.ARM_POSITION: spaces.Box(np.array([MIN_ARM_X, MIN_ARM_Y]), np.array([MAX_ARM_X, MAX_ARM_Y]))
        })
    
        self.action_space = spaces.Dict({
            self.VELOCITY: spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]),np.array([MAX_X_VEL, MAX_Y_VEL])),
            self.ANGULAR_V: spaces.Box(MIN_A_VEL, MAX_A_VEL),
            self.ARM_POSITION: spaces.Box(np.array([MIN_ARM_X, MIN_ARM_Y]), np.array([MAX_ARM_X, MAX_ARM_Y]))}) 
        self.data_log = []
        self.finished = False
        self.last_reset_time = None
        
    def _get_obs(self):
        subscriber_state = self.subscriber.get_state()
        chassis_state = subscriber_state['chassis']
        arm_state = subscriber_state['arm']
        return {
            self.POSITION: np.array([chassis_state['position']['x'], chassis_state['position']['y']]),
            self.OOB: chassis_state['position']['oob'],
            self.ANGLE: np.array([-chassis_state['attitude']['yaw']]),
            self.VELOCITY: np.array([chassis_state['velocity']['vgx'], chassis_state['velocity']['vgy']]),
            self.ANGULAR_V: np.array([chassis_state['imu']['gyro_z']]), # TODO: check gyro_z readings
            self.ACCELERATION: np.array([chassis_state['imu']['acc_x'], chassis_state['imu']['acc_y']]),
            self.MOTOR_SP: np.array([chassis_state['esc']['speed']]),
            self.ARM_POSITION: np.array([arm_state['x'], arm_state['y']])
        }
    
    def get_subscriber_log(self):
        return self.subscriber.to_dict()

    def reset(self, seed=None, options=None):
        logger.debug('reset')
        super().reset(seed=seed)
        self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
        self.data_log.append(self.subscriber.to_dict())
        self.subscriber.reset()
        # go to origin
        obs = self._get_obs()
        angle = obs[self.ANGLE].item()
        logger.debug('moving %s degrees' % (-angle))
        self.robot.chassis.move(z=-angle, z_speed=60.).wait_for_completed(timeout=5.)
        obs = self._get_obs()
        position = obs[self.POSITION]
        while np.abs(position).max() > 5e-2:
            xy_move = np.clip(-position, -MAX_MOVE_POS, MAX_MOVE_POS)
            logger.debug('moving %s' % (xy_move))
            self.robot.chassis.move(xy_move[0].item(), xy_move[1].item(), xy_speed=1.).wait_for_completed(timeout=5.)
            obs = self._get_obs()
            position = obs[self.POSITION]
        logger.debug('reset to %s'%position)
        self.robot.robotic_arm.recenter().wait_for_completed(timeout=2.)
        obs = self._get_obs()
        self.finished = False
        self.last_reset_time = time.time()
        return obs, None
    
    def get_reward(self, obs):
        return 0.
    
    def is_terminating(self, obs):
        return False
    
    def is_action_valid(self, action):
        try:
            return (self.VELOCITY in action and self.ANGULAR_V in action and self.ARM_POSITION in action and
                    action[self.VELOCITY].shape == self.action_space[self.VELOCITY].shape and 
                    action[self.ANGULAR_V].shape == self.action_space[self.ANGULAR_V].shape and
                    action[self.ARM_POSITION].shape == self.action_space[self.ARM_POSITION].shape)
        except:
            return False
    
    def clip(self, action):
        new_action = {}
        new_action[self.VELOCITY] = np.clip(action[self.VELOCITY], self.action_space[self.VELOCITY].low, self.action_space[self.VELOCITY].high)
        new_action[self.ANGULAR_V] = np.clip(action[self.ANGULAR_V], self.action_space[self.ANGULAR_V].low, self.action_space[self.ANGULAR_V].high)
        new_action[self.ARM_POSITION] = np.clip(action[self.ARM_POSITION], self.action_space[self.ARM_POSITION].low, self.action_space[self.ARM_POSITION].high)
        return new_action

    def step(self, action):
        if self.finished:
            raise Exception('Episode has finished. Call reset() to start a new episode.')
        truncated = False
        action_valid = self.is_action_valid(action)
        if not action_valid:
            self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
            truncated = True
        else:
            # apply action
            action = self.clip(action)
            vel = action[self.VELOCITY]
            ang_vel = action[self.ANGULAR_V]
            arm_pos = action[self.ARM_POSITION]
            self.robot.chassis.drive_speed(vel[0].item(), vel[1].item(), ang_vel.item())
            #self.robot.robotic_arm.move(arm_pos[0].item(), arm_pos[1].item())
        # get new observation. The method waits for new information if step is called before the sampling period passed
        obs = self._get_obs()
        reward = self.get_reward(obs)
        terminated = self.is_terminating(obs)
        out_of_bounds = obs[self.OOB]
        if out_of_bounds:
            truncated = True
        info = {'out_of_bounds': out_of_bounds, 'action_valid': action_valid, 'elapsed': time.time() - self.last_reset_time}
        self.finished = terminated or truncated
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        self.robot.chassis.drive_speed(0.,0.,0.,timeout=0.01)
        self.subscriber.unsubscribe()
        self.robot.close()