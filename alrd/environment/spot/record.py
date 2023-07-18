from alrd.environment.spot.mobility_command import MobilityCommand
from alrd.environment.spot.orientation_command import OrientationCommand
from alrd.environment.spot.arm_command import ArmCylindricalVelocityCommand
from alrd.environment.spot.robot_vel_command import RobotVelCommand
from alrd.environment.spot.robot_state import SpotState, JOINT_NAMES, KinematicState
from alrd.environment.spot.command import Command, CommandEnum
from typing import List, Tuple
from dataclasses import dataclass, field, asdict
from mbse.utils.replay_buffer import Transition, BaseBuffer
import numpy as np
import textwrap

def get_command_class(cmd_enum: CommandEnum) -> type:
    if cmd_enum == CommandEnum.MOBILITY:
        return MobilityCommand
    elif cmd_enum == CommandEnum.ORIENTATION:
        return OrientationCommand
    elif cmd_enum == CommandEnum.ARM_CYLINDRICAL:
        return ArmCylindricalVelocityCommand
    elif cmd_enum == CommandEnum.ROBOT_VEL:
        return RobotVelCommand
    else:
        raise NotImplementedError

class Episode:
    def __init__(self, init_obs) -> None:
        self.obs = [init_obs]
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, action: Command, next_obs: SpotState, reward: float, done: bool):
        self.actions.append(action)
        self.obs.append(next_obs)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.actions)

def states_actions_to_str(states: List[KinematicState], actions: List[MobilityCommand]):
    ss = []
    for state, action in zip(states, actions):
        ss.append(textwrap.dedent(f'''\
                time {{
                \tvalue: {state.time }
                }}'''))
        ss.append(action.to_str())
        ss.append(state.to_str())
    return '\n'.join(ss)

class Session(BaseBuffer):
    def __init__(self, only_kinematic: bool, cmd_type: CommandEnum = CommandEnum.MOBILITY,
                 max_size: int=1000000) -> None:
        """
        Parameters
            joint_names: list of joint names
            only_kinematic: if True, only store KinematicState data
            cmd_type: type of commands stored
            max_size: maximum size of the buffer
        """
        if not only_kinematic:
            raise NotImplementedError
        else:
            obs_shape = (KinematicState.size(),)
        super().__init__(obs_shape=obs_shape, action_shape=(get_command_class(cmd_type).size(),),
                         max_size=max_size)
        self.only_kinematic = only_kinematic
        self.cmd_type = cmd_type
        self.episode_lengths = []
    
    def add_episode(self, episode: Episode):
        if self.only_kinematic:
            obs = np.array([np.array(obs.kinematic_state, dtype=np.float64) for obs in episode.obs])
        else:
            obs = np.array([np.array(obs, dtype=np.float64) for obs in episode.obs])
        tran = Transition(
            obs = obs[:-1],
            action= np.array([np.array(action, dtype=np.float64) for action in episode.actions]),
            next_obs= obs[1:],
            reward=np.array([episode.rewards]),
            done=np.array([episode.dones])
        ) 
        self.add(tran)
        self.episode_lengths.append(len(episode))
    
    def asdict(self):
        return {
            'obs': self.obs[:self.size],
            'action': self.action[:self.size],
            'next_obs': self.next_obs[:self.size],
            'reward': self.reward[:self.size],
            'done': self.done[:self.size],
            'joint_names': JOINT_NAMES,
            'episode_lengths': self.episode_lengths,
            'only_kinematic': self.only_kinematic,
            'cmd_type': self.cmd_type.value,
            'max_size': self.max_size
        }
    
    @staticmethod
    def fromdict(d: dict):
        session = Session(
            only_kinematic=d['only_kinematic'],
            cmd_type=CommandEnum(d['cmd_type']),
            max_size=d['max_size'])
        tran = Transition(
            obs = d['obs'],
            action= d['action'],
            next_obs= d['next_obs'],
            reward=d['reward'],
            done=d['done']
        )
        session.add(tran)
        session.episode_lengths = d['episode_lengths']
        return session

    def get_episode(self, i: int) -> Episode:
        assert 0 <= i < len(self.episode_lengths), f'Episode index {i} out of range'
        start = sum(self.episode_lengths[:i])
        end = start + self.episode_lengths[i]
        if self.only_kinematic:
            states = [SpotState(kinematic_state=KinematicState.fromarray(obs)) for obs in self.obs[start:end]]
            states.append(SpotState(kinematic_state=KinematicState.fromarray(self.next_obs[end-1])))
        else:
            states = [SpotState.fromarray(obs) for obs in self.obs[start:end]]
            states.append(SpotState.fromarray(self.next_obs[end-1]))
        actions = [get_command_class(self.cmd_type).fromarray(a) for a in self.action[start:end]]
        rewards = self.reward[start:end]
        dones = self.done[start:end]
        episode = Episode(states[0])
        for state, action, reward, done in zip(states[1:], actions, rewards, dones):
            episode.add(action, state, reward, done)
        return episode
    
    def get_episode_str(self, i: int) -> str:
        assert self.cmd_type == CommandEnum.MOBILITY
        episode = self.get_episode(i)
        states = [state.kinematic_state for state in episode.obs[:-1]]
        actions = episode.actions
        return states_actions_to_str(states, actions)
    
    def __len__(self):
        return len(self.episode_lengths)

