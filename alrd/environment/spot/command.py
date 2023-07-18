from __future__ import annotations

from abc import ABC
from enum import Enum

from bosdyn.api.robot_command_pb2 import RobotCommand

LocomotionHint = int

class CommandEnum(Enum):
    MOBILITY = "mobility"
    ORIENTATION = "orientation"
    ARM_CYLINDRICAL = "arm_cylindrical"
    ROBOT_VEL = "robot_vel"
    
class Command(ABC):
    cmd_type=None
    def __init__(self, cmd: RobotCommand) -> None:
        super().__init__()
        self.__cmd = cmd
    
    @property
    def cmd(self) -> RobotCommand:
        return self.__cmd

    def asdict(self) -> dict:
        return {"type": self.cmd_type}
    
    @staticmethod
    def fromdict(self, d: dict) -> Command:
        raise NotImplementedError