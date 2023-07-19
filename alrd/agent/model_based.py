from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from alrd.agent.absagent import Agent
import numpy as np

class ModelBasedAgentAdapter(Agent):
    def __init__(self, agent: ModelBasedAgent, rng) -> None:
        self.agent = agent
        self.rng = rng

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.act(obs, self.rng, eval=True)
    
    def reset(self):
        self.agent.prepare_agent_for_rollout()