from alrd.maze import Maze
from alrd.subscriber import RobotSub, MazeMarginChecker, VelocityActionSub, ChassisSub
from alrd.environment.env import init_robot, AbsEnv, VelocityControlEnv, PositionControlEnv
from mbse.utils.replay_buffer import Transition
from robomaster.robot import Robot
from gym import spaces
import numpy as np
import time
import logging
from mbse.models.reward_model import RewardModel
import jax
import jax.numpy as jnp
from alrd.maze import create_maze
from alrd.utils import rotate_2d_vector

logger = logging.getLogger(__name__)

def create_maze_env(coordinates, margin=0.22, freq=50, transforms=None, global_act=False):
    _robot = init_robot()
    subscriber = ChassisSub(_robot.chassis, freq=freq)
    maze = Maze(coordinates, margin=margin)
    return MazeEnv(_robot, subscriber, maze, transforms=transforms, global_act=global_act)

def create_maze_goal_env(goal, coordinates=None, margin=0.22, freq=50, slide_wall=False, transforms=None, global_act=False):
    _robot = init_robot()
    subscriber = ChassisSub(_robot.chassis, freq=freq)
    if coordinates is None:
        maze = create_maze(margin=margin)
    else:
        maze = Maze(coordinates, margin=margin)
    return MazeGoalEnv(_robot, subscriber, maze, goal, slide_wall=slide_wall, transforms=transforms, global_act=global_act)

class MazeEnv(AbsEnv):
    def __init__(self, robot: Robot, subscriber: ChassisSub, maze: Maze, slide_wall=True, **kwargs) -> None:
        super().__init__(robot, subscriber, **kwargs)
        self.__velocity_cmd_sub = VelocityActionSub()
        self.maze = maze
        self._margin_checker = MazeMarginChecker(self.robot.chassis,
                                                  self.subscriber.position_sub,
                                                  self.subscriber.attribute_sub,
                                                  self.__velocity_cmd_sub, maze,
                                                  slide_wall=slide_wall, freq=subscriber.freq)

    def _apply_action(self, action):
        action[:2] = rotate_2d_vector(action[:2], self._last_obs[2])
        action = self._margin_checker.check_action(*self._last_obs[:3], *action) # TODO: may want to make the checker monitor speed asynchronously
        action[:2] = rotate_2d_vector(action[:2], -self._last_obs[2])
        # must be in local frame
        super()._apply_action(action)

class MazeReward(RewardModel):
    def __init__(self, goal_pos, maze: Maze):
        self.goal_pos = jnp.array(goal_pos)
        self.maze = maze
        self._init_fn()
    
    def _init_fn(self):
        def _predict(obs, action, goal_pos, next_obs=None, rng=None):
            pos = obs[...,:2]
            reward = -jnp.linalg.norm(pos - goal_pos) ** 2
            return reward
        def predict(obs, action, next_obs=None, rng=None):
            return _predict(obs, action, self.goal_pos, next_obs, rng)
        self.predict = jax.jit(predict)
    
    def train_step(self, tran: Transition):
        raise NotImplementedError
    
    def set_bounds(self, max_action, min_action=None):
        raise NotImplementedError

class MazeGoalEnv(MazeEnv):
    def __init__(self, robot: Robot, subscriber: ChassisSub, maze: Maze, goal: np.ndarray, **kwargs) -> None:
        super().__init__(robot, subscriber, maze, **kwargs)
        self.reward = MazeReward(goal, maze)
    
    def get_reward(self, obs):
        return self.reward.predict(obs, None)

    @classmethod
    def _create_maze_goal_env(cls, goal, freq=50, coordinates=None, margin=0.22, slide_wall=False, transforms=None, **kwargs):
        _robot = init_robot()
        subscriber = ChassisSub(_robot.chassis, freq=freq)
        if coordinates is None:
            maze = create_maze(margin=margin)
        else:
            maze = Maze(coordinates, margin=margin)
        return cls(_robot, subscriber, maze, goal, slide_wall=slide_wall, transforms=transforms, **kwargs)
    
    #def step(self, action):
    #    obs, reward, _, truncated, info = super().step(action)
    #    terminated = jnp.linalg.norm(obs[:2] - self.reward.goal_pos) < self.goaldist
    #    return obs, reward, terminated, truncated, info

class MazeGoalVelocityEnv(MazeGoalEnv, VelocityControlEnv):
    # mro: MazeGoalEnv, MazeEnv, VelocityControlEnv, AbsEnv
    @staticmethod
    def create_env(goal, freq=50, coordinates=None, margin=0.22, slide_wall=False, transforms=None):
        return MazeGoalVelocityEnv._create_maze_goal_env(goal, freq, coordinates, margin, slide_wall, transforms)

class MazeGoalPositionEnv(MazeGoalEnv, PositionControlEnv):
    # mro: MazeGoalEnv, MazeEnv, PositionControlEnv, AbsEnv
    @staticmethod
    def create_env(goal, freq=50, coordinates=None, margin=0.22, slide_wall=False, transforms=None, xy_speed=0.5, a_speed=120.):
        return MazeGoalPositionEnv._create_maze_goal_env(goal, freq, coordinates, margin, slide_wall, transforms, xy_speed=xy_speed, a_speed=a_speed)