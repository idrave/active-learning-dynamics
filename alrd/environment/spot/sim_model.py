from mbse.models.dynamics_model import DynamicsModel
from alrd.environment.spot.simulate2d import Spot2DBaseSim
import numpy as np

class Spot2DModelSim(Spot2DBaseSim):
    def __init__(self, model: DynamicsModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.model.reward_model = self.reward
        self.model._init_fn()
    
    def _update_state(self, action):
        obs = self._get_obs()
        next_obs, _ = self.model.evaluate(
            self.model.model_params,
            obs,
            action,
            rng=None,
            sampling_idx=None,
            model_props=self.model.model_props
        )
        angle = np.arctan2(next_obs[3,None], next_obs[2,None])
        self.state = np.concatenate([next_obs[:2], angle, next_obs[4:]])