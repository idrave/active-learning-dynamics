from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from alrd.environment.env import (BaseRobomasterEnv, PositionControlEnv,
                                  init_robot)
from alrd.environment.maze import (MazeEnv, MazeGoalEnv, MazeGoalKinemEnv,
                                   MazeGoalPositionEnv, MazeGoalVelocityEnv,
                                   create_maze_env)
from alrd.environment.spot.record import Episode, Session
from alrd.environment.spot.robot_state import KinematicState
from alrd.environment.spot.spot2d import (Spot2DEnv, Spot2DReward,
                                          change_spot2d_obs_frame)
from alrd.environment.spot.spotgym import SpotGym
from alrd.environment.wrappers import (CosSinObsWrapper,
                                       GlobalFrameActionWrapper,
                                       KeepObsWrapper,
                                       RemoveAngleActionWrapper,
                                       RepeatActionWrapper)
from jax import vmap

from mbse.utils.replay_buffer import EpisodicReplayBuffer

__all__ = ['BaseRobomasterEnv', 'RobomasterEnv', 'create_maze_env', 'MazeEnv', 'init_robot', 'MazeGoalEnv', 'PositionControlEnv', 'SpotGym']
GOAL = (2.5, 1.8)
def create_robomaster_env(
        poscontrol=False,
        estimate_from_acc=True,
        margin=0.3,
        freq=50,
        slide_wall=True,
        global_frame=False,
        cossin=False,
        noangle=False,
        novelocity=False,
        repeat_action=None,
        square=False,
        xy_speed=0.5,
        a_speed=120.
):
    transforms = []
    coordinates = None
    if square:
        coordinates = np.array([
            [-4, -4],
            [-4, 4],
            [4, 4],
            [4, -4],
    ])
    env_kwargs = dict(
        goal=GOAL,
        coordinates=coordinates,
        margin=margin,
        freq=freq,
        slide_wall=slide_wall,
        transforms=transforms
    )
    if not poscontrol:
        if estimate_from_acc:
            env = MazeGoalKinemEnv.create_env(**env_kwargs)
        else:
            env = MazeGoalVelocityEnv.create_env(**env_kwargs)
    else:
        env = MazeGoalPositionEnv.create_env(
            **env_kwargs,
            xy_speed=xy_speed,
            a_speed=a_speed
        )
    time.sleep(1)
    if global_frame:
        env = GlobalFrameActionWrapper(env)
    assert not noangle or not cossin
    all_idx = set(range(6))
    vel_idx = [3, 4, 5]
    if cossin:
        all_idx.add(6)
        vel_idx = [4, 5, 6]
        env = CosSinObsWrapper(env)
    keep_idx = set(all_idx)
    if noangle:
        keep_idx.remove(2) # remove angle
        keep_idx.remove(5) # remove angular vel
        env = RemoveAngleActionWrapper(env)
    if novelocity:
        keep_idx.difference_update(vel_idx)
    if len(all_idx) != len(keep_idx):
        env = KeepObsWrapper(env, list(keep_idx))
    if repeat_action is not None:
        env = RepeatActionWrapper(env, repeat_action)
    return env

def create_spot_env(
        hostname: str,
        cmd_freq: float,
        monitor_freq: float = 30,
        log_dir: str | Path | None = None):
    """
    Creates and initializes spot environment.
    """
    env = Spot2DEnv(cmd_freq, monitor_freq, log_dir=log_dir)
    env.initialize_robot(hostname)
    return env

def load_episodic_dataset(
        buffer_path: str,
        usepast: Optional[int] = None,
        usepastact: bool = False,
        goal = None
        ):
    """
    Parameters
        buffer_path: path to input buffer
        usepast: number of past observations to include in sampled observation
        usepastact: whether to include past actions in sampled observation
        goal: goal position (x, y, theta)
    """
    data = pickle.load(open(buffer_path, 'rb'))
    obs_shape = (7,)
    action_shape = (3,)
    #hide_in_obs = [0,1,2,3]
    hide_in_obs = None
    assert isinstance(data, EpisodicReplayBuffer)
    buffer = EpisodicReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        normalize=True,
        action_normalize=True,
        learn_deltas=True,
        use_history=usepast,
        use_action_history=usepastact,
        hide_in_obs=hide_in_obs
    )
    if goal is not None:
        reward_model = Spot2DReward()
        reward_fn = vmap(reward_model.predict)
    for i in range(data.num_episodes):
        tran = data.get_episode(i)
        if goal is not None:
            tran.obs[:] = change_spot2d_obs_frame(tran.obs, goal[:2], goal[2])
            tran.next_obs[:] = change_spot2d_obs_frame(tran.next_obs, goal[:2], goal[2])
            tran.reward[:,0] = reward_fn(tran.next_obs, tran.action)[:]
        buffer.add(tran)

    return buffer