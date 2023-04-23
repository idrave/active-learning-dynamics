from alrd.maze import Maze
from alrd.subscriber import RobotSub, MazeMarginChecker, VelocityActionSub, ChassisSub
from alrd.environment.env import init_robot, VelocityControlEnv
from robomaster.robot import Robot
from gym import spaces
import logging

logger = logging.getLogger(__file__)

def create_maze_env(coordinates, margin=0.22, freq=50, transforms=None):
    _robot = init_robot()
    subscriber = ChassisSub(_robot.chassis, freq=freq)
    maze = Maze(coordinates, margin=margin)
    return MazeEnv(_robot, subscriber, maze, transforms=transforms)

class MazeEnv(VelocityControlEnv):
    def __init__(self, robot: Robot, subscriber: ChassisSub, maze: Maze, transforms=None) -> None:
        super().__init__(robot, subscriber, transforms=transforms)
        self.__velocity_cmd_sub = VelocityActionSub()
        self.__margin_checker = MazeMarginChecker(self.robot.chassis,
                                                  self.subscriber.position_sub,
                                                  self.subscriber.attribute_sub,
                                                  self.__velocity_cmd_sub, maze, freq=subscriber.freq)
        self.__last_pos_angle = None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.__last_pos_angle = obs[:3]
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        vx, vy = action[0].item(), action[1].item()
        vz = action[2].item()
        self.__margin_checker.drive_speed(*self.__last_pos_angle, vx, vy, vz) # TODO: may want to make the checker monitor speed asynchronously

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.subscriber.reset() # TODO: go back to initial position
        init_obs = self._get_obs()
        self.__last_pos_angle = init_obs[:3]
        return init_obs, {}

    def get_subscriber_log(self):
        return self.subscriber.to_dict()