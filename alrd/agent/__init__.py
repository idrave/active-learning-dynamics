from alrd.agent.absagent import Agent
from alrd.agent.gp import RandomGPAgent, PiecewiseRandomGPAgent
from alrd.agent.keyboard import KeyboardAgent
from enum import Enum

class AgentType(Enum):
    RANDOM_GP = 'gp'
    PIECEWISE_RANDOM_GP = 'piecewise_gp'
    KEYBOARD = 'keyboard'
    
    def __call__(self, *args, **kwargs):
        if self == AgentType.RANDOM_GP:
            raise NotImplementedError('RandomGPAgent not implemented')
            return RandomGPAgent(*args, **kwargs)
        elif self == AgentType.PIECEWISE_RANDOM_GP:
            raise NotImplementedError('PiecewiseRandomGPAgent not implemented')
            return PiecewiseRandomGPAgent(*args, **kwargs)
        elif self == AgentType.KEYBOARD:
            return KeyboardAgent(*args, **kwargs)
        else:
            raise NotImplementedError(f'Agent type {self} not implemented')

__all__ = ['Agent', 'RandomGPAgent', 'PiecewiseRandomGPAgent', 'KeyboardAgent']