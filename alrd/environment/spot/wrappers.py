from __future__ import annotations
from alrd.environment.spot.spot2d import Spot2DEnv
from gym.core import Wrapper
import numpy as np

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
        if options is None:
            options = {}
        options['goal'] = np.array([float(x), float(y)])
        return self.env.reset(seed=seed, options=options)