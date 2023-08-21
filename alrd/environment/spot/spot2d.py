from __future__ import annotations

import functools
import textwrap
from pathlib import Path
from typing import Any, Optional, Tuple

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

from mbse.models.reward_model import RewardModel
from jdm_control.rewards import get_tolerance_fn


class Spot2DReward(RewardModel):
    def __init__(self, goal_pos: np.ndarray | jnp.ndarray | None = None, action_cost=0.0, velocity_cost=0.0): # TODO check input is 2D
        self._goal_pos = jnp.array(goal_pos) if goal_pos is not None else jnp.zeros((3))
        self._tolerance = get_tolerance_fn(bounds=(0,0), margin=1., sigmoid='gaussian')
        self._dist_margin = 2.0
        self._angle_margin = np.pi
        self.action_cost = action_cost
        self.velocity_cost = velocity_cost

    @functools.partial(jit, static_argnums=0) # assumes object is static
    def predict(self, obs, action, next_obs=None, rng=None):
        pos = obs[...,0:2]
        cos = obs[...,2]
        sin = obs[...,3]
        #front_x, front_y = get_front_coord(x, y, cos, sin)
        reward = self._tolerance((jnp.linalg.norm(pos - self._goal_pos[:2], axis=-1)) / self._dist_margin)
        angle_diff = jnp.abs(jnp.arctan2(sin, cos) - self._goal_pos[2])
        angle_diff = jnp.where(angle_diff < 2 * jnp.pi - angle_diff, angle_diff, 2 * jnp.pi - angle_diff)
        reward = 0.5 * reward + 0.5 * self._tolerance(angle_diff / self._angle_margin)
        vel_reward = self._tolerance(jnp.linalg.norm(obs[..., 4:7]))
        reward = reward * (1 - self.velocity_cost) + vel_reward * self.velocity_cost 
        cost = jnp.linalg.norm(action) / jnp.linalg.norm(jnp.array([1,1,1]))
        return (reward * (1 - self.action_cost) + (1 - cost) * self.action_cost)

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
                 always_reset_pos: bool = True,
                 skip_ui: bool = False,
                 log_str=True):
        if log_dir is None:
            session = None
        else:
            session = Session(only_kinematic=True, cmd_type=CommandEnum.MOBILITY)
        super().__init__(config, cmd_freq, monitor_freq, log_dir=log_dir, session=session, log_str=log_str, always_reset_pos=always_reset_pos)
        self.observation_space = spaces.Box(low=np.array([MIN_X, MIN_Y, -1, -1,-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                            high=np.array([MAX_X, MAX_Y, 1, 1, MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.action_space = spaces.Box(low=np.array([-MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED]),
                                        high=np.array([MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED]))
        self.goal_pos = None # goal position in vision frame
        self.reward = Spot2DReward(action_cost=action_cost, velocity_cost=velocity_cost)
        self.__keyboard = KeyboardResetAgent(KeyboardAgent(0.5, 0.5))
        self.__skip_ui = skip_ui

    def start(self):
        super().start()

    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        """
        Returns
            [x, y, cos, sin, vx, vy, w] with the origin at the goal position and axis aligned to environment frame
        """
        return Spot2DEnv.get_obs_from_state_goal(state, self.goal_pos)

    @staticmethod
    def get_obs_from_state_goal(state: SpotState, goal: np.ndarray) -> np.ndarray:
        origin = goal[:2]
        theta0 = goal[2]
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
    
    def is_done(self, obs: np.ndarray) -> bool:
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminate, truncate, info = super().step(action)
        info['dist'] = np.linalg.norm(obs[:2])
        info['angle'] = np.arctan2(obs[3], obs[2])
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
                    c: continue
                    h: why am I seeing this?
                    answer: """))
                while option not in ['k', 'r', 'c', 'h']:
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
            #angle = self.config.start_angle
            #goal = get_front_coord(self.config.start_x, self.config.start_y, np.cos(angle), np.sin(angle))
            goal = (self.config.start_x, self.config.start_y, self.config.start_angle)
        self.logger.info("Resetting environment with goal {}".format(goal))
        goal_pos = self.body_start_frame.inverse(np.array(goal[:2])) # convert to vision frame
        goal_angle = goal[2] + self.body_start_frame.angle
        self.goal_pos = (*goal_pos, goal_angle)
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