from alrd.environment.env import BaseRobomasterEnv, init_robot, PositionControlEnv
from alrd.environment.maze import MazeEnv, create_maze_env, MazeGoalEnv, MazeGoalPositionEnv, MazeGoalVelocityEnv, MazeGoalKinemEnv
from alrd.environment.wrappers import GlobalFrameActionWrapper, CosSinObsWrapper, RemoveAngleActionWrapper, KeepObsWrapper, RepeatActionWrapper
from alrd.environment.spot.spot import SpotGym, Spot2DEnv
from alrd.subscriber import ChassisSub
import numpy as np
import time

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
        hostname,
        cmd_freq,
        monitor_freq):
    """
    Creates and initializes spot environment.
    """
    env = Spot2DEnv(cmd_freq, monitor_freq)
    env.initialize_robot(hostname)
    return env