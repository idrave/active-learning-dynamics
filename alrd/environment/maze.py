from alrd.maze import Maze
from alrd.subscriber import RobotSub, MazeMarginChecker, VelocityActionSub, ChassisSub
from alrd.environment.env import init_robot, VelocityControlEnv
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

class MazeEnv(VelocityControlEnv):
    def __init__(self, robot: Robot, subscriber: ChassisSub, maze: Maze, slide_wall=True, global_act=True, **kwargs) -> None:
        super().__init__(robot, subscriber, **kwargs)
        self.__velocity_cmd_sub = VelocityActionSub()
        self.maze = maze
        self.__margin_checker = MazeMarginChecker(self.robot.chassis,
                                                  self.subscriber.position_sub,
                                                  self.subscriber.attribute_sub,
                                                  self.__velocity_cmd_sub, maze,
                                                  slide_wall=slide_wall, freq=subscriber.freq)
        self.__origin = None
        obs = self._get_obs(blocking=True)
        obs = self._get_obs() # for some reason the first observation is bugged sometimes
        self.__last_pos_angle = obs[:3]
        self.__origin = obs[:3]
        self.global_act = global_act
        logger.info(f'Created env, frame {"global" if global_act else "local"}')

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
        if not self.global_act:
            action[:2] = rotate_2d_vector(action[:2], self.__last_pos_angle[2])
        action = self.__margin_checker.check_action(*self.__last_pos_angle, *action) # TODO: may want to make the checker monitor speed asynchronously
        action[:2] = rotate_2d_vector(action[:2], -self.__last_pos_angle[2])
        # must be in local frame
        super()._apply_action(action)

    def reset(self, seed=None, options=None):
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
        super().reset(seed, options)
        obs = self._get_obs()
        self.__origin += obs[:3]
        obs = self._get_obs()
        logger.info(f'Reset position: {obs[:3]}')
        self.subscriber.reset()
        self.__last_pos_angle = obs[:3]
        return obs, {}

    def get_subscriber_log(self):
        return self.subscriber.to_dict()

class MazeGoalEnv(MazeEnv):
    def __init__(self, robot: Robot, subscriber: ChassisSub, maze: Maze, goal: np.ndarray, **kwargs) -> None:
        super().__init__(robot, subscriber, maze, **kwargs)
        self.reward = MazeReward(goal, maze)
    
    def get_reward(self, obs):
        return self.reward.predict(obs, None)
    
    #def step(self, action):
    #    obs, reward, _, truncated, info = super().step(action)
    #    terminated = jnp.linalg.norm(obs[:2] - self.reward.goal_pos) < self.goaldist
    #    return obs, reward, terminated, truncated, info

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