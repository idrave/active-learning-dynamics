from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from alrd.environment.spot.command import Command, CommandEnum
from alrd.environment.spot.mobility_command import MobilityCommand
from alrd.environment.spot.record import Episode, Session
from alrd.environment.spot.robot_state import SpotState
from alrd.environment.spot.spot import (MAX_ANGULAR_SPEED, MAX_SPEED, MAXX,
                                        MAXY, MINX, MINY)
from alrd.environment.spot.spotgym import SpotGym
from alrd.environment.spot.utils import get_front_coord
from alrd.utils import change_frame_2d, rotate_2d_vector
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from gym import spaces
from scipy.spatial.transform import Rotation as R

from mbse.models.reward_model import RewardModel
from jdm_control.rewards import get_tolerance_fn


class Spot2DReward(RewardModel):
    def __init__(self, goal_pos: np.ndarray | jnp.ndarray | None = None): # TODO check input is 2D
        self._goal_pos = jnp.array(goal_pos) if goal_pos is not None else jnp.zeros((2))
        self._tolerance = get_tolerance_fn(bounds=(0,0.1), margin=1., sigmoid='gaussian')

    @jax.partial(jax.jit, static_argnums=0) # assumes object is static
    def predict(self, obs, action, next_obs=None, rng=None):
        x, y, cos, sin = obs[:4]
        front_x, front_y = get_front_coord(x, y, cos, sin)
        reward = self._tolerance(jnp.linalg.norm(jnp.array([front_x, front_y]) - self._goal_pos))
        return reward


class Spot2DEnv(SpotGym):
    def __init__(self, cmd_freq: float, monitor_freq: float = 30, log_dir: str | Path | None = None):
        session = Session(only_kinematic=True, cmd_type=CommandEnum.MOBILITY)
        super().__init__(cmd_freq, monitor_freq, log_dir=log_dir, session=session, log_str=True)
        self.observation_space = spaces.Box(low=np.array([MINX, MINY, -1, -1,-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                            high=np.array([MAXX, MAXY, 1, 1, MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.action_space = spaces.Box(low=np.array([-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                        high=np.array([MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.goal_pos = None # goal position in vision frame
        self._vision_origin = change_frame_2d(np.array([0, 0]), self._startpos[:2], self._startpos[2], degrees=False) # vision frame origin relative to environment frame
        self.reward = Spot2DReward()

    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        """
        Returns
            [x, y, cos, sin, vx, vy, w] with the origin at the goal position and axis aligned to environment frame
        """
        origin = self.goal_pos
        _, _, theta0 = self._startpos
        x, y, _, qx, qy, qz, qw = state.pose_of_body_in_vision
        angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        x, y = change_frame_2d(np.array([x, y]), origin, theta0, degrees=False)
        angle -= theta0
        vx, vy, _, _, _, w = state.velocity_of_body_in_vision
        vx, vy = rotate_2d_vector(np.array([vx, vy]), -theta0)
        return np.array([x, y, np.cos(angle), np.sin(angle), vx, vy, w])

    def get_cmd_from_action(self, action: np.ndarray) -> Command:
        return MobilityCommand(action[0], action[1], action[2], height=0.0, pitch=0.0,
                              locomotion_hint=spot_command_pb2.HINT_AUTO, stair_hint=0)

    @staticmethod
    def get_action_from_command(cmd: MobilityCommand) -> np.ndarray:
        return np.array([cmd.vx, cmd.vy, cmd.w])

    def get_reward(self, action, next_obs):
        return self.reward.predict(next_obs, action)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        goal = options.get("goal", None) # optional goal expressed relative to environment frame
        if goal is None:
            self.goal_pos = self._startpos[:2]
        else:
            origin = self._vision_origin
            goal_pos = change_frame_2d(np.array(goal[:2]), origin, -self._startpos[2], degrees=False) # convert to vision frame
            self.goal_pos = np.array([goal_pos[0], goal_pos[1]])
        return super().reset(seed=seed, options=options)


def change_spot2d_obs_frame(obs: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    """
    Change the frame of the observation to the given origin and with the x axis tilted by angle theta.
    Parameters:
        obs: [..., 7] array of observations (x, y, cos, sin, vx, vy, w)
        origin: (x,y) origin of the new frame
        theta: angle in radians
    """
    new_obs = np.array(obs)
    new_obs[..., :2] = change_frame_2d(obs[..., :2], origin, theta, degrees=False)
    new_obs[..., 2:4] = rotate_2d_vector(obs[..., 2:4], -theta, degrees=False)
    new_obs[..., 4:6] = rotate_2d_vector(obs[..., 4:6], -theta, degrees=False)
    return new_obs