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

class BaseRobomasterEnv(gym.Env, ABC):
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
        self.transforms = transforms if transforms is not None else [] 
        self._init_obs_space()
        self._init_action_space()
        self.period = 1./subscriber.freq
        self.last_action_time = None
        self._origin = None
        self._state_log = []
        obs = self._get_obs(blocking=True)
        obs = self._get_obs() # for some reason the first observation is bugged sometimes
        self._origin = obs[:3]
        self._last_obs = obs
    
    def _init_obs_space(self):
        self.observation_space = spaces.Box(np.array([MIN_X, MIN_Y, -180, MIN_X_VEL, MIN_Y_VEL, MIN_A_VEL]), 
                                            np.array([MAX_X, MAX_Y, 180, MAX_X_VEL, MAX_Y_VEL, MAX_A_VEL]))

    @abstractmethod
    def _init_action_space(self):
        pass

    def record_state(self, state):
        self._state_log.append(state)

    def _subscriber_state_to_obs(self, chassis_state):
        return np.array([
            chassis_state['position']['x'], chassis_state['position']['y'],     # position
            chassis_state['attitude']['yaw'],                                   # angle
            chassis_state['velocity']['vgx'], chassis_state['velocity']['vgy'], # velocity
            chassis_state['imu']['gyro_z']                                      # angular velocity
        ])

    def _get_obs(self, blocking=False):
        subscriber_state = self.subscriber.get_state(blocking=blocking)
        subscriber_state['time'] = time.time()
        self.record_state(subscriber_state)
        for transform in self.transforms:
            subscriber_state = transform(subscriber_state)
        obs = self._subscriber_state_to_obs(subscriber_state)
        if self._origin is not None:
            obs[:3] -= self._origin
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_action_time = None
        logger.info('called reset')
        self.stop_robot()
        time.sleep(0.5)
        obs = self._get_obs()
        logger.info(f'reseting from {obs}')
        self.robot.chassis.drive_speed(z=-np.sign(obs[2])*60)
        time.sleep(np.abs(obs[2])/60)
        norm = np.linalg.norm(obs[:2])
        if norm > 1e-5:
            self.robot.chassis.drive_speed(x=-0.5 * obs[0]/norm,y=-0.5 * obs[1]/norm)
            time.sleep(2 * norm)
        self.stop_robot()
        done = False
        while not done:
            s = input('Please reset the robot to the desired position and enter "yes" to continue...')
            if s == 'yes':
                done = True
            else:
                print('Invalid input. Please enter "yes" to continue...')
        time.sleep(2*self.period)
        obs = self._get_obs()
        self._origin += obs[:3]
        obs = self._get_obs()
        logger.info(f'Reset position: {obs[:3]}')
        self.subscriber.reset()
        self._last_obs = obs
        return obs, {'state_idx': len(self._state_log)-1}
    
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
            logger.warning(f'action {action} is invalid. action space is {self.action_space}')
            self.robot.chassis.drive_wheels(timeout=0.01)
            truncated = True
        else:
            self._apply_action(action)
        # get new observation after period has passed
        obs = self._get_obs()
        reward = self.get_reward(obs)
        terminated = self.is_terminating(obs)
        info = {'action_valid': action_valid, 'step_time': time.time()-start, 'elapsed_last_step': elapsed,
                'state_idx': len(self._state_log)-1}
        self._last_obs = obs
        return obs, reward, terminated, truncated, info
    
    def stop_robot(self):
        self.robot.chassis.drive_wheels(0,0,0,0)

    def get_state_log(self):
        return self._state_log

class BaseEnvAccel(BaseRobomasterEnv):
    def __init__(self, robot: robomaster.robot.Robot, subscriber: ChassisSub, transforms=None) -> None:
        self._last_pos = np.zeros((3,))
        self._last_vel = np.zeros((2,))
        self._time_last_read = None
        super().__init__(robot, subscriber, transforms)

    def _subscriber_state_to_obs(self, chassis_state):
        if self._time_last_read is None:
            self._time_last_read = chassis_state['time']
            return np.zeros((6,))
        acc_x = chassis_state['imu']['acc_x']
        acc_y = chassis_state['imu']['acc_y']
        ang_vel = chassis_state['imu']['gyro_z']
        time_delta = chassis_state['time'] - self._time_last_read
        vel_x = self._last_vel[0] + time_delta * acc_x
        vel_y = self._last_vel[1] + time_delta * acc_y
        pos_x = self._last_pos[0] + time_delta * vel_x + 0.5 * time_delta ** 2 * acc_x 
        pos_y = self._last_pos[0] + time_delta * vel_y + 0.5 * time_delta ** 2 * acc_y 
        ang = self._last_pos[2] + time_delta * ang_vel
        self._last_pos = np.array([pos_x, pos_y, ang])
        self._last_vel = np.array([vel_x, vel_y])
        self._time_last_read = chassis_state['time']
        print('deltat %.3f x %.2f y %.2f vx %.2f vy %.2f ax %.2f ay %.2f'%(time_delta, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y))
        return np.array([*self._last_pos, *self._last_vel, ang_vel])
    
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed, options)
        obs = np.zeros((6,))
        self._origin = obs[:3]
        self._last_obs = obs
        self._last_pos = obs[:3]
        self._last_vel = obs[3:5]
        return obs, info

class FrameMixin(BaseRobomasterEnv):
    def __init__(self, global_act=False, *args, **kwargs):
        """
        Action must be an array. Rotates the first two entries according to last angle if the actions must be in global frame 
        """
        super().__init__(*args, **kwargs)
        self.global_frame = global_act
    
    def step(self, action: np.ndarray):
        if self.global_frame:
            action[:2] = rotate_2d_vector(action[:2], -self._last_obs[2])
        result = super().step(action)
        return result
    
class VelocityControlEnv(BaseRobomasterEnv):
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
        time.sleep(self.period)

class PositionControlEnv(BaseRobomasterEnv):
    X_DELTA = 0.5
    Y_DELTA = 0.5
    A_DELTA = 60
    def __init__(self, robot, subscriber, transforms=None, xy_speed=2., a_speed=120) -> None:
        super().__init__(robot, subscriber, transforms=transforms)
        self.xy_speed = xy_speed
        self.a_speed = a_speed
    
    def _init_action_space(self):
        self.action_space = spaces.Box(np.array([-self.X_DELTA, -self.Y_DELTA, -self.A_DELTA]), np.array([self.X_DELTA, self.Y_DELTA, self.A_DELTA]))

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
        start = time.time()
        cmd = self.robot.chassis.move(x=action[0].item(), y=action[1].item(), z=action[2].item(), xy_speed=self.xy_speed, z_speed=self.a_speed)
        cmd.wait_for_completed(timeout=self.period)
        time_passed = time.time() - start
        if self.period - time_passed > 0:
            time.sleep(self.period - time_passed)