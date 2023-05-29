from mbse.agents.actor_critic.sac import SACAgent
from alrd.agent.absagent import Agent
import jax

class AgentAdapter(Agent):
    def __init__(self, agent, rng=None) -> None:
        self.agent = agent
        self.rng = rng

    def act(self, obs):
        self.rng, rng = jax.random.split(self.rng)
        return self.agent.act(obs, rng=rng)