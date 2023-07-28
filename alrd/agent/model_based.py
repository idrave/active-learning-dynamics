from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from alrd.agent.absagent import AgentReset
import numpy as np
from alrd.ui import KeyboardListener

class ModelBasedAgentAdapter(AgentReset):
    def __init__(self, agent: ModelBasedAgent, rng) -> None:
        self.listener = KeyboardListener()
        self.agent = agent
        self.rng = rng

    def act(self, obs: np.ndarray) -> np.ndarray:
        pressed = list(self.listener.which_pressed(['k']))
        if len(pressed) > 0:
            return None
        return self.agent.act(obs, self.rng, eval=True)
    
    def reset(self):
        self.agent.prepare_agent_for_rollout()