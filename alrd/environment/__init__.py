from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from alrd.environment.robomaster.env import (
    BaseRobomasterEnv,
    PositionControlEnv,
    init_robot,
)
from alrd.environment.robomaster.maze_env import (
    MazeEnv,
    MazeGoalEnv,
    MazeGoalKinemEnv,
    MazeGoalPositionEnv,
    MazeGoalVelocityEnv,
    create_maze_env,
)
from alrd.environment.spot.spot import SpotEnvironmentConfig
from alrd.environment.spot.spot2d import (
    Spot2DEnv,
    Spot2DEnvDone,
    Spot2DReward,
    change_spot2d_obs_frame,
)
from alrd.environment.spot.simulate2d import Spot2DEnvSim
from alrd.environment.spot.sim_model import Spot2DModelSim
from alrd.environment.spot.spotgym import SpotGym
from alrd.environment.spot.wrappers import QueryGoalWrapper, QueryStartWrapper
from alrd.environment.wrappers.transforms import (
    CosSinObsWrapper,
    GlobalFrameActionWrapper,
    KeepObsWrapper,
    RemoveAngleActionWrapper,
    RepeatActionWrapper,
)
from alrd.environment.spot.random_pos import RandomPosInit
from jax import vmap

from mbse.utils.replay_buffer import EpisodicReplayBuffer, Transition, ReplayBuffer, BaseBuffer
from mbse.models.dynamics_model import DynamicsModel

__all__ = [
    "BaseRobomasterEnv",
    "RobomasterEnv",
    "create_maze_env",
    "MazeEnv",
    "init_robot",
    "MazeGoalEnv",
    "PositionControlEnv",
    "SpotGym",
]
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
    a_speed=120.0,
):
    transforms = []
    coordinates = None
    if square:
        coordinates = np.array(
            [
                [-4, -4],
                [-4, 4],
                [4, 4],
                [4, -4],
            ]
        )
    env_kwargs = dict(
        goal=GOAL,
        coordinates=coordinates,
        margin=margin,
        freq=freq,
        slide_wall=slide_wall,
        transforms=transforms,
    )
    if not poscontrol:
        if estimate_from_acc:
            env = MazeGoalKinemEnv.create_env(**env_kwargs)
        else:
            env = MazeGoalVelocityEnv.create_env(**env_kwargs)
    else:
        env = MazeGoalPositionEnv.create_env(
            **env_kwargs, xy_speed=xy_speed, a_speed=a_speed
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
        keep_idx.remove(2)  # remove angle
        keep_idx.remove(5)  # remove angular vel
        env = RemoveAngleActionWrapper(env)
    if novelocity:
        keep_idx.difference_update(vel_idx)
    if len(all_idx) != len(keep_idx):
        env = KeepObsWrapper(env, list(keep_idx))
    if repeat_action is not None:
        env = RepeatActionWrapper(env, repeat_action)
    return env


def create_spot_env(
    config: SpotEnvironmentConfig,
    cmd_freq: float,
    monitor_freq: float = 30,
    log_dir: str | Path | None = None,
    query_goal: bool = False,
    action_cost: float = 0.0,
    velocity_cost: float = 0.0,
    simulated: bool = False,
    dynamics_model: DynamicsModel | None = None,
    seed: int | None = None,
    random_init_pose: Tuple[float, float, float, float] | None = None,
    done_on_goal_tol: Tuple[float, float, float] | None = None
):
    """
    Creates and initializes spot environment.
    """
    assert not random_init_pose or not query_goal
    assert not random_init_pose or seed
    if not simulated:
        if done_on_goal_tol is not None:
            # the episode is ended when the robot is at certain distance and angle
            # from the goal with velocity smaller than the one specified
            env = Spot2DEnvDone(
                done_on_goal_tol[0],
                done_on_goal_tol[1],
                done_on_goal_tol[2],
                config,
                cmd_freq,
                monitor_freq,
                log_dir=log_dir,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
        else:
            env = Spot2DEnv(
                config,
                cmd_freq,
                monitor_freq,
                log_dir=log_dir,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
    else:
        if dynamics_model is None:
            env = Spot2DEnvSim(
                config,
                cmd_freq,
                action_cost=action_cost,
                velocity_cost=velocity_cost,
            )
        else:
            env = Spot2DModelSim(
                dynamics_model,
                config,
                action_cost=action_cost,
                velocity_cost=velocity_cost
            )
    if query_goal:
        env = QueryStartWrapper(env)
        env = QueryGoalWrapper(env)
    if random_init_pose:
        env = RandomPosInit(env, seed, random_init_pose[:2], random_init_pose[2:])
    return env


def load_dataset(
    buffer_path: str,
    goal=None,
    action_cost: float = 0.0,
    velocity_cost: float = 0.0,
    normalize: bool = True,
    action_normalize: bool = False,
    learn_deltas: bool = True
):
    """
    Parameters
        buffer_path: path to input buffer
        goal: goal position (x, y)
        action_cost: action cost used to compute reward when goal is specified
        velocity_cost: velocity cost used to compute reward when goal is specified
        action_normalize: whether to normalize actions
    """
    data = pickle.load(open(buffer_path, "rb"))
    obs_shape = (7,)
    action_shape = (3,)
    assert isinstance(data, BaseBuffer)
    buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=learn_deltas,
    )
    if goal is not None:
        reward_model = Spot2DReward.create(
            action_coeff=action_cost, velocity_coeff=velocity_cost
        )
        reward_fn = reward_model.predict
    tran = data.get_full_raw_data()
    if goal is not None:
        tran.obs[:] = change_spot2d_obs_frame(tran.obs, goal[:2], goal[2])
        tran.next_obs[:] = change_spot2d_obs_frame(tran.next_obs, goal[:2], goal[2])
        tran.reward[:, 0] = reward_fn(tran.next_obs, tran.action)[:]
    buffer.add(tran)
    return buffer

def load_episodic_dataset(
    buffer_path: str,
    usepast: Optional[int] = None,
    usepastact: bool = False,
    goal=None,
    action_cost: float = 0.0,
    velocity_cost: float = 0.0,
    normalize: bool = True,
    action_normalize: bool = False,
    learn_deltas: bool = True,
    episode_len: Optional[int] = None
):
    """
    Parameters
        buffer_path: path to input buffer
        usepast: number of past observations to include in sampled observation
        usepastact: whether to include past actions in sampled observation
        goal: goal position (x, y)
        action_cost: action cost used to compute reward when goal is specified
        velocity_cost: velocity cost used to compute reward when goal is specified
        action_normalize: whether to normalize actions
    """
    data = pickle.load(open(buffer_path, "rb"))
    obs_shape = (7,)
    action_shape = (3,)
    # hide_in_obs = [0,1,2,3]
    hide_in_obs = None
    assert isinstance(data, BaseBuffer)
    if episode_len is None:
        assert isinstance(data, EpisodicReplayBuffer)
        num_episodes = data.num_episodes
    else:
        num_episodes = data.size // episode_len
    buffer = EpisodicReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=learn_deltas,
        use_history=usepast,
        use_action_history=usepastact,
        hide_in_obs=hide_in_obs,
    )
    if goal is not None:
        reward_model = Spot2DReward.create(
            action_coeff=action_cost, velocity_coeff=velocity_cost
        )
        reward_fn = reward_model.predict
    for i in range(num_episodes):
        if episode_len is None:
            tran = data.get_episode(i)
        else:
            tran = Transition(
                obs         = data.obs[i * episode_len : (i + 1) * episode_len],
                action      = data.action[i * episode_len : (i + 1) * episode_len],
                next_obs    = data.next_obs[i * episode_len : (i + 1) * episode_len],
                reward      = data.reward[i * episode_len : (i + 1) * episode_len],
                done        = data.done[i * episode_len : (i + 1) * episode_len],
            )
        if goal is not None:
            tran.obs[:] = change_spot2d_obs_frame(tran.obs, goal[:2], goal[2])
            tran.next_obs[:] = change_spot2d_obs_frame(tran.next_obs, goal[:2], goal[2])
            tran.reward[:, 0] = reward_fn(tran.next_obs, tran.action)[:]
        buffer.add(tran)

    return buffer


def add_2d_zero_samples(buffer: EpisodicReplayBuffer, num_samples: int, reward_model=None):
    """
    Adds zero samples to buffer.
    """
    action_shape = (3,)
    pose = np.random.uniform(
        low=[buffer.obs[:, 0].min(), buffer.obs[:, 1].min(), -np.pi],
        high=[buffer.obs[:, 0].max(), buffer.obs[:, 1].max(), np.pi],
        size=(num_samples, 3),
    )
    obs = np.hstack(
        [
            pose[:, :2],
            np.cos(pose[:, [2]]),
            np.sin(pose[:, [2]]),
            np.zeros((num_samples, 3)),
        ]
    )
    if reward_model is None:
        reward = np.zeros((num_samples, 1))
    else:
        reward = vmap(reward_model.predict)(obs, np.zeros((num_samples, *action_shape)))
    buffer.add(
        Transition(
            obs=obs,
            action=np.zeros((num_samples, *action_shape)),
            next_obs=obs,
            reward=reward,
            done=np.zeros((num_samples, 1)),
        )
    )

def get_first_n(buffer: BaseBuffer, n: int):
    """ Get a buffer with same args as buffer but with only first n transitions """
    tran = buffer.get_full_raw_data()
    new_tran = Transition(
        obs=tran.obs[:n],
        action=tran.action[:n],
        next_obs=tran.next_obs[:n],
        reward=tran.reward[:n],
        done=tran.done[:n],
    )
    augment = None
    if isinstance(buffer, EpisodicReplayBuffer):
        augment = buffer.augment
    new_buffer = EpisodicReplayBuffer(
        obs_shape=buffer.obs_shape,
        action_shape=buffer.action_shape,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas,
        augment=augment
    )
    new_buffer.add(new_tran)
    return new_buffer