from __future__ import annotations
from alrd.environment.spot.command import Command, CommandEnum
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY


class OrientationCommand(Command):
    cmd_type=CommandEnum.ORIENTATION
    def __init__(self, roll: float, pitch: float, yaw: float, height: float) -> None:
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.height = height
        # build command
        orientation = EulerZXY(roll=self.roll, pitch=self.pitch, yaw=self.yaw)
        cmd = RobotCommandBuilder.synchro_stand_command(
            body_height=self.height, footprint_R_body=orientation)
        super().__init__(cmd)

    def asdict(self) -> dict:
        return {
            **super().asdict(),
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "height": self.height,
        }

    @staticmethod
    def fromdict(d: dict) -> OrientationCommand:
        return OrientationCommand(
            roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"], height=d["height"])