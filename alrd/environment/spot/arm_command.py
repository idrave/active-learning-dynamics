from __future__ import annotations
from alrd.environment.spot.command import Command, CommandEnum
from alrd.environment.spot.utils import CylindricalVelocity
from google.protobuf import wrappers_pb2
from bosdyn.api import geometry_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2
import numpy as np

from dataclasses import dataclass

ARM_MAX_LINEAR_VELOCITY = 0.5

@dataclass
class ArmCylindricalVelocityCommand(Command):
    cmd_type=CommandEnum.ARM_CYLINDRICAL
    velocity: CylindricalVelocity

    def __post_init__(self) -> None:
        cylindrical_coord = geometry_pb2.CylindricalCoordinate(
            r=self.velocity.r,
            theta=self.velocity.theta,
            z=self.velocity.z
        )
        cylindrical_vel = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity(
            linear_velocity=cylindrical_coord,
            max_linear_velocity=wrappers_pb2.DoubleValue(value=ARM_MAX_LINEAR_VELOCITY)
        )
        arm_vel_cmd = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_vel
        )
        arm_cmd = arm_command_pb2.ArmCommand.Request(
            arm_velocity_command=arm_vel_cmd
        )
        sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
        arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
        super().__init__(arm_sync_robot_cmd)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array(self.velocity, dtype=dtype)
    
    @staticmethod
    def fromarray(arr: np.ndarray) -> ArmCylindricalVelocityCommand:
        return ArmCylindricalVelocityCommand(CylindricalVelocity.fromarray(arr))