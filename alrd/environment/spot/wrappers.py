from __future__ import annotations
from typing import Tuple
from alrd.environment.spot.spot2d import Spot2DEnv
from alrd.utils.utils import Frame2D, rotate_2d_vector
from alrd.environment.spot.utils import get_hitbox
from gym.core import Env, Wrapper
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class RandomGoalWrapper(Wrapper):
    def __init__(self, env: Spot2DEnv):
        super().__init__(env)
    
    def reset(self, **kwargs):
        kwargs['goal'] = self.env.observation_space.sample()[:2]
        return self.env.reset(**kwargs)
    
class QueryStartWrapper(Wrapper):
    def __init__(self, env: Spot2DEnv):
        super().__init__(env)
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        x = input('Enter start x: ')
        y = input('Enter start y: ')
        angle = input('Enter start angle: ')
        if options is None:
            options = {}
        options['pose'] = np.array([float(x), float(y), float(angle)])
        return self.env.reset(seed=seed, options=options)

class QueryGoalWrapper(Wrapper):
    def __init__(self, env: Spot2DEnv):
        super().__init__(env)
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        x = input('Enter goal x: ')
        y = input('Enter goal y: ')
        angle = input('Enter goal angle: ')
        if options is None:
            options = {}
        options['goal'] = np.array([float(x), float(y), float(angle)])
        return self.env.reset(seed=seed, options=options)

class FixedGoal2DWrapper(Wrapper):
    def __init__(self, env: Spot2DEnv, goal: np.ndarray):
        assert goal.shape == (3,)
        super().__init__(env)
        self.goal = goal
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {}
        options['goal'] = self.goal
        return self.env.reset(seed=seed, options=options)

class OptionWrapper(Wrapper):
    def __init__(self, env: Env, options: dict):
        super().__init__(env)
        self.__options = options
    
    def reset(self, options: dict | None = None, **kwargs):
        if options is None:
            options = {}
        options.update(self.__options)
        return super().reset(options=options, **kwargs)

class Trajectory2DWrapper(Wrapper):
    def __init__(self, env: Spot2DEnv, output_dir: str):
        super().__init__(env)
        self.__trajectory = []
        self.__output_dir = Path(output_dir)
        self.__counter = 0
        self.__tag = None

    def reset(self, options: dict | None = None, **kwargs):
        self.__end_episode()
        if options is not None:
            self.__tag = options.get('tag', None)
        obs, info = super().reset(options=options, **kwargs)    
        self.__trajectory.append(obs[:4])
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.__trajectory.append(obs[:4])
        if terminated or truncated:
            self.__end_episode()
        return obs, reward, terminated, truncated, info
    
    def close(self):
        super().close()
        self.__end_episode()

    def __end_episode(self):
        if len(self.__trajectory) > 1:
            self.__save_trajectory()
            self.__counter += 1
        self.__trajectory = []
        self.__tag = None

    def __save_trajectory(self):
        trajectory = np.array(self.__trajectory)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Position')
        goal_frame = self.env.goal_frame
        goal_in_env = self.env.body_start_frame.transform_pose(
            goal_frame.x, goal_frame.y, goal_frame.angle
        )
        frame = Frame2D(goal_in_env[0], goal_in_env[1], 0.0) # only translate
        corners = [
            [self.env.config.min_x, self.env.config.min_y],
            [self.env.config.min_x, self.env.config.max_y],
            [self.env.config.max_x, self.env.config.max_y],
            [self.env.config.max_x, self.env.config.min_y],
        ]
        corners = np.array(corners)
        corners = frame.transform(corners)
        position = rotate_2d_vector(trajectory[:, :2], goal_in_env[2], degrees=False) # align to environment
        last_x, last_y = position[-1, :2]
        last_angle = np.arctan2(trajectory[-1, 3], trajectory[-1, 2])
        last_angle += goal_in_env[2]
        plt.xlim(corners[:,0].min() - 0.5, corners[:,0].max() + 0.5)
        plt.ylim(corners[:,1].min() - 0.5, corners[:,1].max() + 0.5)
        plt.gca().set_aspect('equal')
        plt.scatter(position[:, 0], position[:, 1], c=np.linspace(0.2,1,position.shape[0]), cmap="Reds")
        plt.scatter([0],[0], color='green', marker='x')
        plt.arrow(0., 0., 0.5 * np.cos(goal_in_env[2]), 0.5 * np.sin(goal_in_env[2]), color='green', width=0.05)
        for i in range(-1, len(corners)-1):
            plt.plot([corners[i,0], corners[i+1,0]], [corners[i,1], corners[i+1,1]], marker = 'o', color='black')
        box = get_hitbox(last_x, last_y, last_angle)
        for i in range(-1, len(box)-1):
            plt.plot([box[i,0], box[i+1,0]], [box[i,1], box[i+1,1]], marker = 'o', color='yellow')
        plt.subplot(1, 2, 2)
        plt.title('Angle to goal')
        plt.ylim(-np.pi, np.pi)
        plt.plot(np.arange(trajectory.shape[0]), np.arctan2(trajectory[:, 3], trajectory[:, 2]))
        name = f'{self.__counter:03d}.png'
        if self.__tag is not None:
            name = f'{self.__tag}-{name}'
        plt.savefig(self.__output_dir / name)
        plt.close()