from gym.core import Env, Wrapper
from alrd.environment.spot.spot2d import Spot2DEnv
import random
import numpy as np

class RandomPosInit(Wrapper):
    def __init__(self, env: Spot2DEnv, seed: int, x_range, y_range):
        super().__init__(env)
        self.rng = random.Random(seed)
        self.x_range = np.array(x_range)
        self.y_range = np.array(y_range)
    
    def reset(self, seed=None, options=None, **kwargs):
        if options is None:
            options = {}
        if 'pose' not in options:
            x0, y0, _ = self.env.default_reset
            x = self.rng.uniform(*(self.x_range + x0))
            y = self.rng.uniform(*(self.y_range + y0))
            theta = self.rng.uniform(-np.pi, np.pi)
            options['pose'] = np.array([x, y, theta])
        return super().reset(seed=seed, options=options, **kwargs)