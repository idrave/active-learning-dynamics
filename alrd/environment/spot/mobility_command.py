from __future__ import annotations
import numpy as np
from alrd.environment.spot.command import Command, CommandEnum, LocomotionHint
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY


from dataclasses import asdict, dataclass


@dataclass
class MobilityCommand(Command):
    cmd_type=CommandEnum.MOBILITY
    vx: float
    vy: float
    w: float
    height: float
    pitch: float
    locomotion_hint: LocomotionHint
    stair_hint: bool
    def __post_init__(self) -> None:
        # build command
        orientation = EulerZXY(roll=0.0, pitch=self.pitch, yaw=0.0)
        mobility_params = RobotCommandBuilder.mobility_params(
            body_height=self.height, footprint_R_body=orientation, locomotion_hint=self.locomotion_hint,
            stair_hint=self.stair_hint)
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=self.vx, v_y=self.vy, v_rot=self.w, params=mobility_params)
        super().__init__(cmd)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.vx, self.vy, self.w, self.height, self.pitch, self.locomotion_hint, self.stair_hint], dtype=dtype)

    @staticmethod
    def fromarray(arr: np.ndarray) -> MobilityCommand:
        return MobilityCommand(*arr[:5], int(arr[5]), int(arr[6]))

    def asdict(self) -> dict:
        return {
            **super().asdict(),
            **asdict(self)
        }

    @staticmethod
    def fromdict(d: dict) -> MobilityCommand:
        return MobilityCommand(
            vx=d["vx"], vy=d["vy"], w=d["w"], height=d["height"], pitch=d["pitch"],
            locomotion_hint=d["locomotion_hint"], stair_hint=d["stair_hint"])

    def to_str(self) -> str:
        s = "velocity_commands {\n"
        s += "\tx: {:.5f}\n".format(self.vx)
        s += "\ty: {:.5f}\n".format(self.vy)
        s += "\trot: {:.5f}\n".format(self.w)
        s += "}"
        return s

    @staticmethod
    def size() -> int:
        return 7