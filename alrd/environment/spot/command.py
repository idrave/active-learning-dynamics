from __future__ import annotations
from abc import ABC, abstractmethod
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api.robot_command_pb2 import RobotCommand
from bosdyn.geometry import EulerZXY
from typing import Optional, Any
from enum import Enum

LocomotionHint = int

class CommandEnum(Enum):
    MOBILITY = "mobility"
    ORIENTATION = "orientation"
    def get_class(self) -> type:
        if self == CommandEnum.MOBILITY:
            return MobilityCommand
        elif self == CommandEnum.ORIENTATION:
            return OrientationCommand
        else:
            raise NotImplementedError
    
    def fromdict(self, d: dict) -> Command:
        return self.get_class().fromdict(d)

class Command(ABC):
    cmd_type=None
    def __init__(self, cmd: RobotCommand) -> None:
        super().__init__()
        self.__cmd = cmd
    
    @property
    def cmd(self) -> RobotCommand:
        return self.__cmd

    @abstractmethod
    def asdict(self) -> dict:
        return {"type": self.cmd_type}
    
    @staticmethod
    def fromdict(self, d: dict) -> Command:
        raise NotImplementedError

class MobilityCommand(Command):
    cmd_type=CommandEnum.MOBILITY
    def __init__(self, vx: float, vy: float, w: float, height: float, pitch: float,
                 locomotion_hint: LocomotionHint, stair_hint: bool, build_on_command: Optional[Command]=None) -> None:
        self.vx = vx
        self.vy = vy
        self.w = w
        self.height = height
        self.pitch = pitch
        self.locomotion_hint = locomotion_hint
        self.stair_hint = stair_hint
        self.build_on_command = build_on_command
        # build command
        orientation = EulerZXY(roll=0.0, pitch=self.pitch, yaw=0.0)
        mobility_params = RobotCommandBuilder.mobility_params(
            body_height=self.height, footprint_R_body=orientation, locomotion_hint=self.locomotion_hint,
            stair_hint=self.stair_hint)
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=self.vx, v_y=self.vy, v_rot=self.w, params=mobility_params,
            build_on_command=self.build_on_command.cmd if self.build_on_command is not None else None)
        super().__init__(cmd)
    
    def asdict(self) -> dict:
        return {
            **super().asdict(),
            "vx": self.vx,
            "vy": self.vy,
            "w": self.w,
            "height": self.height,
            "pitch": self.pitch,
            "locomotion_hint": self.locomotion_hint,
            "stair_hint": self.stair_hint,
            "build_on_command": self.build_on_command.asdict() if self.build_on_command is not None else None
        }
    
    @staticmethod
    def fromdict(d: dict) -> MobilityCommand:
        return MobilityCommand(
            vx=d["vx"], vy=d["vy"], w=d["w"], height=d["height"], pitch=d["pitch"],
            locomotion_hint=d["locomotion_hint"], stair_hint=d["stair_hint"],
            build_on_command=CommandEnum.fromdict(d["build_on_command"]) if d["build_on_command"] is not None else None)
    
    def to_str(self) -> str:
        s = "velocity_commands {\n"
        s += "\tx: {:.5f}\n".format(self.vx)
        s += "\ty: {:.5f}\n".format(self.vy)
        s += "\trot: {:.5f}\n".format(self.w)
        s += "}\n"
        return s


class OrientationCommand(Command):
    cmd_type=CommandEnum.ORIENTATION
    def __init__(self, roll: float, pitch: float, yaw: float, height: float,
                 build_on_command: Optional[Command]=None) -> None:
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.height = height
        self.build_on_command = build_on_command
        # build command
        orientation = EulerZXY(roll=self.roll, pitch=self.pitch, yaw=self.yaw)
        cmd = RobotCommandBuilder.synchro_stand_command(
            body_height=self.height, footprint_R_body=orientation,
            build_on_command=self.build_on_command.cmd if self.build_on_command is not None else None)
        super().__init__(cmd)
    
    def asdict(self) -> dict:
        return {
            **super().asdict(),
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "height": self.height,
            "build_on_command": self.build_on_command.asdict() if self.build_on_command is not None else None
        }
    
    @staticmethod
    def fromdict(d: dict) -> OrientationCommand:
        return OrientationCommand(
            roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"], height=d["height"],
            build_on_command=CommandEnum.fromdict(d["build_on_command"]) if d["build_on_command"] is not None else None)