import gym
import gym.spaces
from alrd.environment.robomaster.maze import Maze, create_maze
import numpy as np
from alrd.environment.robomaster.names import *
from alrd.environment.robomaster.maze_env import MazeReward

class VirtualMaze(gym.Env):
    def __init__(self, maze: Maze, goal, freq=10, accel=5.) -> None:
        self.freq = freq
        self.accel = accel
        self.period = 1 / freq
        self.maze = maze
        self.last_pos = None
        self.last_vel = None
        self.observation_space = gym.spaces.Box(np.array([MIN_X, MIN_Y, MIN_X_VEL, MIN_Y_VEL]), np.array([MAX_X, MAX_Y, MAX_X_VEL, MAX_Y_VEL]))
        self.action_space = gym.spaces.Box(np.array([MIN_X_VEL, MIN_Y_VEL]), np.array([MAX_X_VEL, MAX_Y_VEL]))
        self.reward = MazeReward(goal, maze)
    
    @staticmethod
    def create_env(goal, freq=10, coordinates=None, margin=0.2, accel=5.):
        if coordinates is None:
            maze = create_maze(margin=margin)
        else:
            maze = Maze(coordinates, margin=margin)
        return VirtualMaze(maze, goal, freq, accel)

    def step(self, action):
        # check if action is in box range
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be in range of action space. Action: {action}, action space: {self.action_space}")
        vx, vy = self.maze.clamp_direction(self.last_pos, 0, action)
        v_delta = np.array([vx, vy]) - self.last_vel
        if np.allclose(v_delta, 0):
            acceleration = np.zeros(2)
        else:
            acceleration = v_delta / max(1., np.linalg.norm(v_delta) / (self.accel * self.period))
        velocity = self.last_vel + acceleration * self.period
        position = self.last_pos + self.last_vel * self.period + 0.5 * acceleration * self.period**2
        prev_obs = np.concatenate([self.last_pos, self.last_vel])
        self.last_pos = position
        self.last_vel = velocity
        obs = np.concatenate([position, velocity])
        return obs, self.reward.predict(prev_obs, action, obs), False, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_pos = np.array([0., 0.])
        self.last_vel = np.array([0., 0.])
        return np.concatenate([self.last_pos, self.last_vel]), {}