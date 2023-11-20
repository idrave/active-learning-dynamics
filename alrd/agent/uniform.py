from alrd.agent.absagent import Agent
import numpy as np

class UniformAgent(Agent):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=3)
    
    def reset(self):
        pass