from alrd.agent.absagent import Agent
from alrd.agent.gp import RandomGPAgent, PiecewiseRandomGPAgent, create_async_rbf_gp_agent
from alrd.agent.keyboard import KeyboardAgent, KeyboardGPAgent
from alrd.agent.trajaxopt import TraJaxOptAgent
from alrd.agent.adapter import AgentAdapter
from mbse.utils.agent_checkpoint import SACCheckpoint
from enum import Enum
import pickle

class AgentType(Enum):
    KEYBOARD = 'keyboard'
    KEYBOARD_GP = 'keyboard_gp'
    SAC = 'sac'
    ILQROPEN = 'ilqropen'
    
    def __call__(self, args):
        if self == AgentType.KEYBOARD:
            return KeyboardAgent(xy_speed=args.xy_speed, a_speed=args.a_speed, noangle=args.noangle)
        elif self == AgentType.KEYBOARD_GP:
            gp_agent = create_async_rbf_gp_agent(
                length_scale=args.length_scale,
                noise=args.noise,
                scale=(1, 1, 0.5),
                max_steps=args.episode_len if args.episode_len is not None else args.freq * 60,
                freq=args.freq, sample=args.gp_undersample, seed=args.seed)
            return KeyboardGPAgent(gp_agent=gp_agent, xy_speed=1, a_speed=1)
        elif self == AgentType.SAC:
            with open(args.agent_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            agent, opt_state = checkpoint.create_agent()
            return AgentAdapter(agent, rng=args.agent_rng)
        elif self == AgentType.ILQROPEN:
            with open(args.model_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            checkpoint.args['reward_model'] = args.reward_model
            model = checkpoint.create_model()
            print('args rng',args.agent_rng)
            return TraJaxOptAgent.create(
                model=model,
                action_dim=(3,),
                horizon=args.horizon,
                rng=args.agent_rng
            )
        else:
            raise NotImplementedError(f'Agent type {self} not implemented')

__all__ = ['Agent', 'RandomGPAgent', 'PiecewiseRandomGPAgent', 'KeyboardAgent']