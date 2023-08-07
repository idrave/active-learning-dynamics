from __future__ import annotations
from alrd.environment.spot.spot2d import Spot2DEnv
from gym.core import Env, Wrapper
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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

class Trajectory2DWrapper(Wrapper):
    def __init__(self, env: Spot2DEnv, output_dir: str):
        super().__init__(env)
        self.__trajectory = []
        self.__output_dir = Path(output_dir)
        self.__counter = 0

    def reset(self, **kwargs):
        if len(self.__trajectory) > 0:
            self.__save_trajectory()
            self.__trajectory = []
        obs, info = self.env.reset(**kwargs)    
        self.__trajectory.append(obs[:2])
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.__trajectory.append(obs[:2])
        if terminated or truncated:
            self.__save_trajectory()
            self.__trajectory = []
        return obs, reward, terminated, truncated, info

    def __save_trajectory(self):
        trajectory = np.array(self.__trajectory)
        plt.figure()
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.linspace(0.2,1,trajectory.shape[0]), cmap="Reds")
        plt.savefig(self.__output_dir / f'{self.__counter:03d}.png')
        plt.close()
        self.__counter += 1