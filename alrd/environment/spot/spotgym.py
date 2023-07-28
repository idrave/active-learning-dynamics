from __future__ import annotations

import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import gym
import numpy as np
from alrd.environment.spot.command import Command
from alrd.environment.spot.orientation_command import OrientationCommand
from alrd.environment.spot.record import Session, Episode
from alrd.environment.spot.robot_state import SpotState
from alrd.environment.spot.spot import (COMMAND_DURATION, CHECK_TIMEOUT,
                                        SpotGymStateMachine, SpotEnvironmentConfig, State)
from alrd.utils import get_timestamp_str

from enum import Enum

class ResetEnum(Enum):
    RESET_POSE = 0
    STAND = 1
    NONE = 2

class SpotGym(SpotGymStateMachine, gym.Env, ABC):
    def __init__(self, config: SpotEnvironmentConfig, cmd_freq: float, monitor_freq: float = 30,
                 log_dir: str | Path | None = None, session: Session | None = None, log_str: bool = False, always_reset_pos: bool = False):
        """
        Parameters:
            cmd_freq: Environment's maximum action frequency. Commands will take at least 1/cmd_freq seconds to execute.
            monitor_freq: Environment's maximum state monitoring frequency for checking position boundaries.
            truncate_on_timeout: If True, the episode will end when a robot command takes longer than STEP_TIMEOUT seconds.
            log_dir: Directory where to save logs.
            session: Session object to record episode data.
            log_str: If True, command and state info is logged as a string to a file.
        If log_dir is not None and session is not None, session data will be dumped after each episode.
        """
        #assert 1/monitor_freq <= CHECK_TIMEOUT + 1e-5, "Monitor frequency must be higher than 1/{} Hz to ensure safe navigation".format(CHECK_TIMEOUT)
        assert 1/cmd_freq <= COMMAND_DURATION + 1e-5, "Command frequency must be higher than 1/COMMAND_DURATION ({} Hz) ".format(1/COMMAND_DURATION)
        assert session is None or log_dir is not None, "If session is not None, log_dir must be specified"
        assert not log_str or log_dir is not None, "If log_str is True, log_dir must be specified"
        super().__init__(config, monitor_freq=monitor_freq)
        self.__cmd_freq = cmd_freq
        self.__should_reset = True
        self.__last_robot_state = None
        self.__current_episode = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.log_file = None
        self.session = session
        self.log_str = log_str
        self.always_reset_pos = always_reset_pos
        if log_dir is not None:
            self.logger.addHandler(logging.FileHandler(self.log_dir / "spot_gym.log"))

    def start(self):
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(exist_ok=False, parents=True)
        super().start()

    def close(self):
        super().close()
        self._end_episode()

    def print_to_file(self, command: Command, state: SpotState, currentTime):
        if self.log_file is None:
            filepath = self.log_dir / ('session-'+get_timestamp_str() + ".txt")
            self.log_file = open(filepath, "w")
        self.log_file.write("time {{\n \tvalue: {:.5f} \n}}\n".format(currentTime))
        self.log_file.write(command.to_str()+"\n")
        self.log_file.write(state.to_str()+"\n")

    def stop_robot(self) -> bool:
        if not self.isopen:
            raise RuntimeError("Environment is closed but stop was called.")
        result = self._issue_stop()
        self._end_episode()
        return result

    @property
    def should_reset(self):
        return self.__should_reset

    def _end_episode(self):
        self.__should_reset = True
        self.__last_robot_state = None
        if self.log_dir is not None:
            if self.log_str and self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            if self.session is not None and self.__current_episode is not None and len(self.__current_episode) > 0:
                self.session.add_episode(self.__current_episode)
                self.__current_episode = None
                pickle.dump(self.session.asdict(), open(self.log_dir / "record.pkl", "wb"))

    def _step(self, cmd: Command) -> Tuple[SpotState | None, float, float | None, bool | None]:
        """
        Apply the command for as long as the command period specified.
        Returns:
            new_state: The new state of the robot after the command is applied. This is None if the state service timed out reading the state.
            cmd_time: The time it took to issue the command + read the state.
            read_time: The time it took to read the state of the robot.
            oob: Whether the robot was out of bounds in the new state returned
        """
        if not self.isopen or self.should_reset:
            raise RuntimeError("Environment is closed or should be reset but step was called.")
        start_cmd = time.time()
        success, result = self._issue_command(cmd, 1/self.__cmd_freq)
        cmd_time = time.time() - start_cmd
        if not success:
            # command was interrupted
            self._end_episode()
            return None, cmd_time, None, None, False
        next_state, read_time, oob = result
        if next_state is None or oob:
            self._end_episode()
        return next_state, cmd_time, read_time, oob

    @abstractmethod
    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_cmd_from_action(self, action: np.ndarray) -> Command:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, action: np.ndarray, next_obs: np.ndarray) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def is_done(self, obs: np.ndarray) -> bool:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Converts the action to a robot command, applies it and returns the next state as a numpy array
        If the 
        """
        cmd = self.get_cmd_from_action(action)
        next_state, cmd_time, read_time, oob = self._step(cmd)
        info = {"cmd_time": cmd_time, "read_time": read_time, "oob": oob}
        truncate = False
        if next_state is None:
            truncate = True
            next_state = self.__last_robot_state
        if oob:
            truncate = True
        if self.session is not None:
            self.__current_episode.add(cmd, next_state, reward, done)
        if self.log_str:
            self.print_to_file(cmd, self.__last_robot_state, time.time())
        self.__last_robot_state = next_state
        obs = self.get_obs_from_state(next_state)
        reward = self.get_reward(action, obs)
        done = self.is_done(obs)
        if done:
            self.stop_robot()
        return obs, reward, done, done, info

    def _reset(self, action: ResetEnum) -> Tuple[SpotState | None, float] | None:
        """
        Reset the robot to the origin.
        """
        if not self.isopen:
            raise RuntimeError("Environment is closed but reset was called.")
        if not self.should_reset:
            self._end_episode()
        reset_pose = self.state in {State.STOPPED, State.STOPPING} or action == ResetEnum.RESET_POSE
        if reset_pose or action == ResetEnum.STAND:
            self.logger.info("Reset called, stopping robot...")
            success = self._issue_stop()
            if not success:
                self.logger.error("Reset stop failed")
                return None
            self.logger.info("Robot stopped")
            if reset_pose:
                input("Press enter to reset the robot to the origin... ")
                # reset position
                success, _ = self._issue_reset()
                if not success:
                    self.logger.error("Failed to reset robot position")
                    return None
            input("Reset done. Press enter to continue.")
        start = time.time()
        new_state = self._read_robot_state()
        read_time = time.time() - start
        self.__should_reset = False
        self.__last_robot_state = new_state
        self.__current_episode = Episode(new_state)
        return new_state, read_time

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray | None, dict]:
        """
        Args:
            seed: set the seed for the environment's random number generator.
            options: additional options for the reset, among them:
                action: ResetEnum indicating  whether to reset pose, stand, or do nothing.
        """
        super().reset(seed=seed, options=options)
        if options is None:
            options = {}
        action = options.get("action", ResetEnum.RESET_POSE)
        result = self._reset(action)
        if result is None:
            return None, {}
        state, read_time = result
        info = {"read_time": read_time}
        if state is None:
            return None, info
        return self.get_obs_from_state(state), info