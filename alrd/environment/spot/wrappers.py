from __future__ import annotations
from typing import Tuple
from alrd.environment.spot.spot2d import Spot2DEnv
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
        self.__end_episode()
        super().close()

    def __end_episode(self):
        if len(self.__trajectory) > 1:
            self.__save_trajectory()
            self.__counter += 1
        self.__trajectory = []
        self.__tag = None

    def __save_trajectory(self):
        trajectory = np.array(self.__trajectory)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Position')
        plt.xlim(self.env.observation_space.low[0], self.env.observation_space.high[0])
        plt.ylim(self.env.observation_space.low[1], self.env.observation_space.high[1])
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.linspace(0.2,1,trajectory.shape[0]), cmap="Reds")
        plt.subplot(1, 2, 2)
        plt.title('Angle')
        plt.ylim(-np.pi, np.pi)
        plt.plot(np.arange(trajectory.shape[0]), np.arctan2(trajectory[:, 3], trajectory[:, 2]))
        name = f'{self.__counter:03d}.png'
        if self.__tag is not None:
            name = f'{self.__tag}-{name}'
        plt.savefig(self.__output_dir / name)
        plt.close()