from alrd.agent.absagent import Agent
from alrd.agent.gp import RandomGPAgent, PiecewiseRandomGPAgent, create_async_rbf_gp_agent
from alrd.agent.keyboard import KeyboardAgent, KeyboardGPAgent, KeyboardResetAgent
from alrd.agent.trajaxopt import TraJaxOptAgent
from alrd.agent.adapter import AgentAdapter
from mbse.utils.agent_checkpoint import SACCheckpoint
from enum import Enum
import pickle

class AgentType(Enum):
    KEYBOARD = 'keyboard'
    KEYBOARD_GP = 'keyboard_gp'
    KEYBOARD_GP_XY = 'keyboard_gp_xy'
    SAC = 'sac'
    ILQROPEN = 'ilqropen'
    
    def __call__(self, args):
        if self == AgentType.KEYBOARD:
            return KeyboardAgent(xy_speed=args.xy_speed, a_speed=args.a_speed, noangle=args.noangle)
        elif self == AgentType.KEYBOARD_GP or self == AgentType.KEYBOARD_GP_XY:
            if self == AgentType.KEYBOARD_GP_XY:
                scale = (args.xy_speed, args.xy_speed, 0.0)
            else:
                scale = (args.xy_speed, args.xy_speed, 0.5*args.a_speed/120.)
            gp_agent = create_async_rbf_gp_agent(
                length_scale=args.length_scale,
                noise=args.noise,
                scale=scale if not args.noangle else scale[:2],
                max_steps=args.episode_len if args.episode_len is not None else args.freq * 60,
                freq=args.freq//(args.repeat_action if args.repeat_action is not None else 1), sample=args.gp_undersample, seed=args.seed)
            return KeyboardGPAgent(gp_agent=gp_agent, xy_speed=args.xy_speed, a_speed=args.a_speed)
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

class SpotAgentEnum(Enum):
    KEYBOARD='keyboard'

def create_spot_agent(agent_type: SpotAgentEnum):
    if agent_type == SpotAgentEnum.KEYBOARD:
        base_agent = KeyboardAgent(xy_speed=1, a_speed=1)
        return KeyboardResetAgent(base_agent)
    else:
        raise NotImplementedError(f'Agent type {agent_type} not implemented')

__all__ = ['Agent', 'RandomGPAgent', 'PiecewiseRandomGPAgent', 'KeyboardAgent']