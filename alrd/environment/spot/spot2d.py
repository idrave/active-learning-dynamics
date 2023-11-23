from __future__ import annotations

import functools
import textwrap
from pathlib import Path
from typing import Any, Optional, Tuple, Callable

from jax import jit 
import jax
import jax.numpy as jnp
import numpy as np
from alrd.environment.spot.command import Command, CommandEnum
from alrd.environment.spot.mobility_command import MobilityCommand
from alrd.environment.spot.record import Episode, Session
from alrd.environment.spot.robot_state import SpotState
from alrd.environment.spot.spotgym import SpotGym 
from alrd.environment.spot.spot import SpotEnvironmentConfig
from alrd.environment.spot.utils import MAX_ANGULAR_SPEED, MAX_SPEED, get_front_coord
from alrd.utils.utils import change_frame_2d, rotate_2d_vector, Frame2D
from alrd.agent.keyboard import KeyboardResetAgent, KeyboardAgent
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from gym import spaces
from scipy.spatial.transform import Rotation as R
from alrd.environment.spot.utils import MAX_ANGULAR_SPEED, MAX_SPEED

from mbse.models.reward_model import RewardModel
from jdm_control.rewards import get_tolerance_fn
from dataclasses import InitVar
from flax import struct

def norm(x: jax.Array, axis: int):
    norm = jnp.sum(x * x, axis=axis)
    return jnp.sqrt(norm + 1e-12)

_DEFAULT_VALUE_AT_MARGIN = 0.1

@struct.dataclass
class DistReward(RewardModel):
    tolerance_fn: Callable = struct.field(pytree_node=False)
    goal_pos: jax.Array
    margin: float

    @staticmethod
    def create(
        goal_pos: jax.Array,
        margin: float,
        sigmoid: str = 'long_tail',
        bounds = None,
        value_at_margin = _DEFAULT_VALUE_AT_MARGIN
    ):
        if bounds is None:
            bounds = (0., 0.)
        bounds = np.array(bounds)
        assert goal_pos.shape == (2,)
        tolerance_fn = get_tolerance_fn(
            bounds=bounds/margin, margin=1., sigmoid=sigmoid,
            value_at_margin=value_at_margin)
        return DistReward(tolerance_fn, goal_pos, margin)
    
    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        #dist = jnp.linalg.norm(obs[..., 0:2] - self.goal_pos, axis=-1)
        dist = norm(obs[..., 0:2] - self.goal_pos, -1)
        return self.tolerance_fn(dist / self.margin)

@struct.dataclass
class AngleReward(RewardModel):
    tolerance_fn: Callable = struct.field(pytree_node=False)
    goal_angle: float
    margin: float

    @staticmethod
    def create(
        goal_angle: float,
        margin: float,
        sigmoid: str = 'long_tail',
        bounds = None,
        value_at_margin = _DEFAULT_VALUE_AT_MARGIN
    ):
        if bounds is None:
            bounds = (0., 0.)
        bounds = np.array(bounds)
        tolerance_fn = get_tolerance_fn(
            bounds=bounds/margin, margin=1., sigmoid=sigmoid,
            value_at_margin=value_at_margin)
        return AngleReward(tolerance_fn, goal_angle, margin)

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        cos = obs[..., 2]
        sin = obs[..., 3]
        angle_diff = jnp.abs(jnp.arctan2(sin, cos) - self.goal_angle)
        angle_diff = jnp.where(angle_diff < 2. * jnp.pi - angle_diff, angle_diff, 2. * jnp.pi - angle_diff)
        return self.tolerance_fn(angle_diff / self.margin)


@struct.dataclass
class ActionCost(RewardModel):
    tolerance_fn: Callable = struct.field(pytree_node=False)
    
    @staticmethod
    def create(sigmoid: str = 'long_tail'):
        tolerance_fn = get_tolerance_fn(margin=jnp.linalg.norm(jnp.ones([3])).item(), sigmoid=sigmoid)
        return ActionCost(tolerance_fn)

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        act_norm = norm(action[..., :3], axis=-1)
        return 1 - act_norm / jnp.linalg.norm(jnp.ones([3]))


@struct.dataclass
class LinearVelCost(RewardModel):
    tolerance_fn: Callable = struct.field(pytree_node=False)
    
    @staticmethod
    def create(sigmoid: str = 'long_tail'):
        tolerance_fn = get_tolerance_fn(margin=1., sigmoid=sigmoid)
        return LinearVelCost(tolerance_fn)

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        #vel_norm = jnp.linalg.norm(obs[..., 4:6], axis=-1)
        vel_norm = norm(obs[..., 4:6], axis=-1)
        return self.tolerance_fn(vel_norm / MAX_SPEED)


@struct.dataclass
class AngularVelCost(RewardModel):
    tolerance_fn: Callable = struct.field(pytree_node=False)

    @staticmethod
    def create(sigmoid: str = 'long_tail'):
        tolerance_fn = get_tolerance_fn(margin=1., sigmoid=sigmoid)
        return AngularVelCost(tolerance_fn)

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        return self.tolerance_fn(obs[..., 6] / MAX_ANGULAR_SPEED)


@struct.dataclass
class GoalLinearVelCost(RewardModel):
    vel_cost: LinearVelCost
    dist_rew: DistReward

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        return self.dist_rew.predict(obs, action, next_obs, rng) * \
                self.vel_cost.predict(obs, action, next_obs, rng)


@struct.dataclass
class GoalAngularVelCost(RewardModel):
    vel_cost: AngularVelCost
    angl_rew: AngleReward

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        return self.angl_rew.predict(obs, action, next_obs, rng) * \
                self.vel_cost.predict(obs, action, next_obs, rng)

@struct.dataclass
class Spot2DReward(RewardModel):
    dist_rew: DistReward
    angl_rew: AngleReward
    act_cost: ActionCost
    linvel_cost: LinearVelCost | GoalLinearVelCost
    angvel_cost: AngularVelCost| GoalAngularVelCost
    action_coeff: float = struct.field(default=0.)
    velocity_coeff: float = struct.field(default=0.)
    angle_coeff: float = struct.field(default=0.5)

    @staticmethod
    def create(
        goal_pos: np.ndarray | jnp.ndarray = None,
        angle_coeff: float = 0.5,
        action_coeff: float = 0.,
        velocity_coeff: float = 0., 
        dist_margin: float = 6., 
        angle_margin: float = np.pi, 
        sigmoid: str = 'long_tail', 
        vel_cost_on_goal: bool = False,
        vel_lin_margin: float = None,
        vel_ang_margin: float = None,
        dist_bound: float = 0.,
        angle_bound: float = 0.
    ):
        if vel_cost_on_goal:
            if vel_lin_margin is None:
                vel_lin_margin = 0.2
            if vel_ang_margin is None:
                vel_ang_margin = jnp.pi / 18.
        if goal_pos is None:
            goal_pos = jnp.zeros([3])
        dist_rew = DistReward.create(goal_pos[:2], dist_margin, sigmoid=sigmoid, bounds=(0., dist_bound))
        angl_rew = AngleReward.create(goal_pos[2], angle_margin, sigmoid=sigmoid, bounds=(0., angle_bound))
        act_cost = ActionCost.create(sigmoid=sigmoid)
        linvel_cost = LinearVelCost.create(sigmoid=sigmoid)
        angvel_cost = AngularVelCost.create(sigmoid=sigmoid)
        if vel_cost_on_goal:
            value_at_margin = 1e-2
            step_dist = DistReward.create(
                goal_pos[:2],
                vel_lin_margin,
                sigmoid='gaussian',
                value_at_margin=value_at_margin)
            linvel_cost = GoalLinearVelCost(
                linvel_cost, step_dist
            )
            step_ang = AngleReward.create(
                goal_pos[2],
                vel_ang_margin,
                sigmoid='gaussian',
                value_at_margin=value_at_margin
            )
            angvel_cost = GoalAngularVelCost(
                angvel_cost, step_ang
            )
        return Spot2DReward(
            dist_rew,
            angl_rew,
            act_cost,
            linvel_cost,
            angvel_cost,
            action_coeff,
            velocity_coeff,
            angle_coeff
        )

    @jit
    def predict(self, obs, action, next_obs=None, rng=None):
        reward = (1-self.angle_coeff) * self.dist_rew.predict(obs, action, next_obs, rng) + \
                    (self.angle_coeff) * self.angl_rew.predict(obs, action, next_obs, rng)
        action_cost = self.act_cost.predict(obs, action, next_obs, rng)
        velocity_cost = (1-self.angle_coeff) * self.linvel_cost.predict(obs, action, next_obs, rng) + \
                            (self.angle_coeff) * self.angvel_cost.predict(obs, action, next_obs, rng)
        return reward * (1. - self.action_coeff - self.velocity_coeff) + \
                self.action_coeff * action_cost + self.velocity_coeff * velocity_cost

MIN_X = -4
MIN_Y = -3
MAX_X = 4
MAX_Y = 3

class Spot2DEnv(SpotGym):
    obs_shape = (7,)
    action_shape = (3,)

    def __init__(self,
                 config: SpotEnvironmentConfig,
                 cmd_freq: float,
                 monitor_freq: float = 30,
                 log_dir: str | Path | None = None,
                 action_cost=0.0,
                 velocity_cost=0.0,
                 skip_ui: bool = False,
                 log_str=True):
        if log_dir is None:
            session = None
        else:
            session = Session(only_kinematic=True, cmd_type=CommandEnum.MOBILITY)
        super().__init__(config, cmd_freq, monitor_freq, log_dir=log_dir, session=session, log_str=log_str)
        self.observation_space = spaces.Box(low=np.array([MIN_X, MIN_Y, -1, -1,-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                            high=np.array([MAX_X, MAX_Y, 1, 1, MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.action_space = spaces.Box(low=np.array([-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                        high=np.array([MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.__goal_frame = None # goal position in vision frame
        self.reward = Spot2DReward.create(action_coeff=action_cost, velocity_coeff=velocity_cost)
        self.__keyboard = KeyboardResetAgent(KeyboardAgent(0.5, 0.5))
        self.__skip_ui = skip_ui

    def start(self):
        super().start()
    
    @property
    def goal_frame(self) -> Frame2D:
        return self.__goal_frame

    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        """
        Returns
            [x, y, cos, sin, vx, vy, w] with the origin at the goal position and axis aligned to environment frame
        """
        return Spot2DEnv.get_obs_from_state_goal(state, self.__goal_frame)

    @staticmethod
    def get_obs_from_state_goal(state: SpotState, goal_frame: Frame2D) -> np.ndarray:
        x, y, _, qx, qy, qz, qw = state.pose_of_body_in_vision
        angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        x, y, angle = goal_frame.transform_pose(x, y, angle)
        vx, vy, _, _, _, w = state.velocity_of_body_in_vision
        vx, vy = goal_frame.transform_direction(np.array((vx, vy)))
        return np.array([x, y, np.cos(angle), np.sin(angle), vx, vy, w])

    def get_cmd_from_action(self, action: np.ndarray) -> Command:
        return MobilityCommand(action[0], action[1], action[2], height=0.0, pitch=0.0,
                              locomotion_hint=spot_command_pb2.HINT_AUTO, stair_hint=0)

    @staticmethod
    def get_action_from_command(cmd: MobilityCommand) -> np.ndarray:
        return np.array([cmd.vx, cmd.vy, cmd.w])

    def get_reward(self, action, next_obs):
        return self.reward.predict(next_obs, action)
    
    def is_done(self, obs: np.ndarray) -> bool:
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminate, truncate, info = super().step(action)
        info['dist'] = np.linalg.norm(obs[:2])
        info['angle'] = np.abs(np.arctan2(obs[3], obs[2]))
        return obs, reward, terminate, truncate, info
    
    def _show_ui(self):
        MANUAL_CONTROL_FREQ = 10
        input('called reset, press enter to continue...')
        prompt = input('type "yes" to enter options menu or press enter to continue: ')
        if len(prompt) == 0:
            return
        if prompt != 'yes':
            print(f'entered "{prompt}". continuing to reset...')
        else:
            option = None
            while option != 'c':
                option = input(textwrap.dedent("""
                    Options:
                    --------------
                    k: keyboard control
                    r: reset base position to current position
                    b: print hitbox
                    c: continue
                    h: why am I seeing this?
                    answer: """))
                while option not in ['k', 'r', 'c', 'h', 'b']:
                    print(f'entered "{option}". invalid option...')
                    option = input('answer: ')
                if option == 'k':
                    print(self.__keyboard.kb_agent.description() + "\nk: end manual control")
                    action = self.__keyboard.act(None)
                    while action is not None:
                        success, result = self._issue_unmonitored_command(self.get_cmd_from_action(action), 1/MANUAL_CONTROL_FREQ)
                        if not success:
                            print("command failed, exiting manual control. press enter to continue...")
                            break
                        action = self.__keyboard.act(None)
                    input("manual control ended. press enter to go back to options...")
                elif option == 'b':
                    print("hitbox:")
                    for row in self._get_spot_hitbox(self._read_robot_state()):
                        print(row)
                    input("press enter to go back to options...")
                elif option == 'r':
                    print("WARNING: boundary safety checks and reset position will now be computed relative to the current pose")
                    confirm = input("confirm action by entering yes: ")
                    if confirm == 'yes':
                        self.set_start_frame()
                        print("origin reset done")
                    else:
                        print("origin reset cancelled")
                elif option == 'h':
                    print(textwrap.dedent("""
                        The robot continualy updates an estimate of its position using its onboard sensors
                        and camera. After long periods of operation, this estimate can drift from the truth,
                        which affects the safety checks done by the environment.
                        This interface allows you to move the robot to the position where the program was
                        started and reset the base pose that the environment uses to compute the safety
                        checks, which are expressed relative to this base pose."""))
                    input("press enter to go back to options...")
                else:
                    print("continuing...")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        if not self.__skip_ui:
            # TODO this stop should not be necessary, but need this for now to guarantee the unmonitored
            # command works in the state machine
            success = self._issue_stop()
            if not success:
                self.logger.error("Reset stop failed")
                return None, {}
            self._show_ui()
        if options is None:
            options = {}
        goal = options.get("goal", None) # optional goal expressed relative to environment frame
        if goal is None:
            goal = (self.config.start_x, self.config.start_y, self.config.start_angle)
        self.logger.info("Resetting environment with goal {}".format(goal))
        goal_pos = self.body_start_frame.inverse_pose(*goal) # convert to vision frame
        self.__goal_frame = Frame2D(*goal_pos)
        return super().reset(seed=seed, options=options)


class Spot2DEnvDone(Spot2DEnv):
    """Stops the robot when close to goal pose with low velocity"""
    def __init__(self, dist_tol, ang_tol, vel_tol, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dist_tol = dist_tol
        self.ang_tol = ang_tol
        self.vel_tol = vel_tol
    
    def is_done(self, obs: np.ndarray) -> bool:
        return np.linalg.norm(obs[:2]) < self.dist_tol and \
                np.abs(np.arctan2(obs[3], obs[2])) < self.ang_tol and \
                np.linalg.norm(obs[4:]) < self.vel_tol

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