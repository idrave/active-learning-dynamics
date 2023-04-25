from alrd.maze import Maze
from alrd.subscriber import RobotSub, MazeMarginChecker, VelocityActionSub, ChassisSub
from alrd.environment.env import init_robot, VelocityControlEnv
from robomaster.robot import Robot
from gym import spaces
import numpy as np
import time
import logging

logger = logging.getLogger(__file__)

def create_maze_env(coordinates, margin=0.22, freq=50, transforms=None):
    _robot = init_robot()
    subscriber = ChassisSub(_robot.chassis, freq=freq)
    maze = Maze(coordinates, margin=margin)
    return MazeEnv(_robot, subscriber, maze, transforms=transforms)

class MazeEnv(VelocityControlEnv):
    def __init__(self, robot: Robot, subscriber: ChassisSub, maze: Maze, transforms=None, slide_wall=True) -> None:
        super().__init__(robot, subscriber, transforms=transforms)
        self.__velocity_cmd_sub = VelocityActionSub()
        self.__margin_checker = MazeMarginChecker(self.robot.chassis,
                                                  self.subscriber.position_sub,
                                                  self.subscriber.attribute_sub,
                                                  self.__velocity_cmd_sub, maze,
                                                  slide_wall=slide_wall, freq=subscriber.freq)
        self.__origin = None
        obs = self._get_obs(blocking=True)
        self.__last_pos_angle = obs[:3]
        self.__origin = obs[:3]

    def _get_obs(self, blocking=False):
        obs = super()._get_obs(blocking)
        if self.__origin is not None:
            obs[:3] -= self.__origin
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.__last_pos_angle = obs[:3]
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        vx, vy = action[0].item(), action[1].item()
        vz = action[2].item()
        action = self.__margin_checker.check_action(*self.__last_pos_angle, vx, vy, vz) # TODO: may want to make the checker monitor speed asynchronously
        if np.allclose(action, 0, atol=1e-4):
            self.robot.chassis.drive_wheels()
            return
        self.robot.chassis.drive_speed(action[0].item(), action[1].item(), action[2].item())

    def reset(self, seed=None, options=None):
        done = False
        while not done:
            s = input('Please reset the robot to the desired position and enter "yes" to continue...')
            if s == 'yes':
                done = True
            else:
                print('Invalid input. Please enter "yes" to continue...')
        time.sleep(2*self.period)
        super().reset(seed, options)
        obs = self._get_obs()
        self.__origin += obs[:3]
        logger.info(f'Reset position: {obs[:3]}')
        self.subscriber.reset()
        self.__last_pos_angle = self._get_obs()[:3]
        return obs, {}

    def get_subscriber_log(self):
        return self.subscriber.to_dict()