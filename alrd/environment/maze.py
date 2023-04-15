from alrd.maze import Maze
from alrd.subscriber import RobotSub, MazeMarginChecker, VelocityActionSub
from alrd.environment.env import init_robot, VelocityControlEnv
from gymnasium import spaces
import logging

logger = logging.getLogger(__file__)

def create_maze_env(coordinates, margin=0.22, freq=50, transforms=None):
    _robot = init_robot()
    subscriber = RobotSub(_robot, freq=freq)
    maze = Maze(coordinates, margin=margin)
    return MazeEnv(_robot, subscriber, maze, freq=freq, transforms=transforms)

class MazeEnv(VelocityControlEnv):
    def __init__(self, robot, subscriber, maze: Maze, freq=50, transforms=None) -> None:
        super().__init__(robot, subscriber, transforms=transforms)
        self.__velocity_cmd_sub = VelocityActionSub()
        self.__margin_checker = MazeMarginChecker(self.robot.chassis,
                                                  self.subscriber.chassis_sub.position_sub,
                                                  self.subscriber.chassis_sub.attribute_sub,
                                                  self.__velocity_cmd_sub, maze, freq=freq)
        self.__last_pos_angle = None
        #self.__margin_checker.start()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.__last_pos_angle = (*obs[self.POSITION], obs[self.ANGLE].item())
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        vx, vy = action[self.VELOCITY]
        vz = action[self.ANGULAR_V].item()
        #self.__velocity_cmd_sub.callback((vx, vy, vz))
        self.__margin_checker.drive_speed(*self.__last_pos_angle, vx, vy, vz) # TODO: may want to make the checker monitor speed asynchronously


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.subscriber.reset() # TODO: go back to initial position
        init_obs = self._get_obs()
        self.__last_pos_angle = (*init_obs[self.POSITION], init_obs[self.ANGLE].item())
        return init_obs, {}

    def get_subscriber_log(self):
        return self.subscriber.to_dict()
    
    def close(self):
        #self.__margin_checker.join()
        super().close()