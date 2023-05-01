from alrd.agent.absagent import Agent
from alrd.agent.gp import RandomGPAgent, PiecewiseRandomGPAgent, create_async_rbf_gp_agent
from alrd.agent.keyboard import KeyboardAgent, KeyboardGPAgent
from enum import Enum

class AgentType(Enum):
    KEYBOARD = 'keyboard'
    KEYBOARD_GP = 'keyboard_gp'
    
    def __call__(self, args):
        if self == AgentType.KEYBOARD:
            return KeyboardAgent(xy_speed=1, a_speed=1)
        elif self == AgentType.KEYBOARD_GP:
            gp_agent = create_async_rbf_gp_agent(
                length_scale=args.length_scale,
                noise=args.noise,
                scale=(1, 1, 1),
                max_steps=args.episode_len if args.episode_len is not None else args.freq * 60,
                freq=args.freq, sample=args.gp_undersample, seed=args.seed)
            return KeyboardGPAgent(gp_agent=gp_agent, xy_speed=1, a_speed=1)
        else:
            raise NotImplementedError(f'Agent type {self} not implemented')

__all__ = ['Agent', 'RandomGPAgent', 'PiecewiseRandomGPAgent', 'KeyboardAgent']