from __future__ import annotations

import numpy as np
from gym import spaces

from alrd.environment.spot.command import Command, CommandEnum
from alrd.environment.spot.arm_command import ARM_MAX_LINEAR_VELOCITY
from alrd.environment.spot.record import Session
from alrd.environment.spot.robot_state import SpotState
from alrd.environment.spot.spotgym import SpotGym
from pathlib import Path

MIN_HEIGHT = 0.0
MAX_HEIGHT = 1.8
MIN_AZIMUTHAL = - (150 / 180) * np.pi
MAX_AZIMUTHAL = np.pi
MIN_RADIAL_POS = -1.
MAX_RADIAL_POS = 1.
MAX_RADIAL_VEL = ARM_MAX_LINEAR_VELOCITY
MAX_VERTICAL_VEL = ARM_MAX_LINEAR_VELOCITY
MAX_AZIMUTHAL_VEL = np.pi / 4

class ArmEnv(SpotGym):
    def __init__(self, cmd_freq: float, monitor_freq: float = 30, log_dir: str | Path | None = None):
        session = Session(only_kinematic=False, cmd_type=CommandEnum.ARM_CYLINDRICAL)
        super().__init__(cmd_freq, monitor_freq, log_dir=log_dir, session=session, log_str=False)
        self.observation_space = spaces.Box(
            low=np.array([MIN_RADIAL_POS, MIN_AZIMUTHAL, MIN_HEIGHT, -MAX_RADIAL_VEL, -MAX_AZIMUTHAL_VEL, -MAX_VERTICAL_VEL]),
            high=np.array([MAX_RADIAL_POS, MAX_AZIMUTHAL, MAX_HEIGHT, MAX_RADIAL_VEL, MAX_AZIMUTHAL_VEL, MAX_VERTICAL_VEL]))
        self.action_space = spaces.Box(
            low=np.array([-MAX_RADIAL_VEL, -MAX_AZIMUTHAL_VEL, -MAX_VERTICAL_VEL]),
            high=np.array([MAX_RADIAL_VEL, MAX_AZIMUTHAL_VEL, MAX_VERTICAL_VEL]))

    def get_obs_from_state(self, state: SpotState) -> np.ndarray:
        pose = state.manipulator_state.pose_of_hand
        radial = np.linalg.norm([pose.position.x, pose.position.y])
        azimuthal = np.arctan2(pose.position.y, pose.position.x)
        height = pose.position.z
        velocity = state.manipulator_state.velocity_of_hand