from alrd.maze import Maze
from alrd.subscriber import RobotSub, MazeMarginChecker, VelocityActionSub
from alrd.environment.env import init_robot, VelocityControlEnv
from gymnasium import spaces

def create_maze_env(coordinates, margin=0.22, freq=50):
    _robot = init_robot()
    subscriber = RobotSub(_robot, freq=freq)
    maze = Maze(coordinates, margin=margin)
    return MazeEnv(_robot, subscriber, maze, freq=freq)

class MazeEnv(VelocityControlEnv):
    def __init__(self, robot, subscriber, maze: Maze, freq=50) -> None:
        super().__init__(robot, subscriber)
        self.__velocity_cmd_sub = VelocityActionSub()
        self.__margin_checker = MazeMarginChecker(self.robot.chassis, self.__velocity_cmd_sub, maze, freq=freq)
    
    def _apply_action(self, action):
        vel = action[self.VELOCITY]
        ang_vel = action[self.ANGULAR_V]
        self.__margin_checker.drive_speed(vel[0], vel[1], ang_vel)

    def step(self, action):
        self.__velocity_cmd_sub.callback(action)
        super().step(action)

    def reset(self, seed=None, options=None):
        raise NotImplementedError

    def get_subscriber_log(self):
        return self.subscriber.to_dict()
    
    def close(self):
        self.__margin_checker.unsubscribe()
        super().close()