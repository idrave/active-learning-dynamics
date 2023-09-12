from __future__ import annotations
from alrd.agent.absagent import Agent
from mbse.optimizers.sac_based_optimizer import SACOptimizer
import numpy as np

class SACAgent(Agent):
    def __init__(self, optimizer: SACOptimizer, smoothing: float | None, agent_idx = 0) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.smoothing = smoothing
        self.model_state = None
        self.agent_idx = agent_idx
        self._act_dim = optimizer.dynamics_model.act_dim
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.optimizer.get_action_for_eval(
            obs, self.model_state, None, self.agent_idx
        )
        if self.model_state is not None:
            finalaction = self.smoothing * action[...,:self._act_dim] + \
                                                (1 - self.smoothing) * self.model_state
        else:
            finalaction = action[...,:self._act_dim]
        self.model_state = self.optimizer.dynamics_model.next_model_state(self.model_state, action)
        return finalaction
    
    def reset(self):
        self.model_state = self.optimizer.dynamics_model.init_model_state()