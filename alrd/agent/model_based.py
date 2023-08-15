from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from alrd.agent.absagent import AgentReset
import numpy as np
from alrd.utils.keyboard import KeyboardListener

class ModelBasedAgentAdapter(AgentReset):
    def __init__(self, agent: ModelBasedAgent, rng, eval: bool) -> None:
        self.listener = KeyboardListener()
        self.agent = agent
        self.rng = rng
        self.eval = eval

    def act(self, obs: np.ndarray) -> np.ndarray:
        pressed = list(self.listener.which_pressed(['k']))
        if len(pressed) > 0:
            return None
        return self.agent.act(obs, self.rng, eval=self.eval)
    
    def reset(self):
        self.agent.prepare_agent_for_rollout()