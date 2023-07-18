from __future__ import annotations
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api.robot_command_pb2 import RobotCommand
from bosdyn.geometry import EulerZXY
from typing import Optional

LocomotionHint = int

def create_mobility_command(vx: float, vy: float, w: float, height: float, pitch: float,
                 locomotion_hint: LocomotionHint, stair_hint: bool, build_on_command: Optional[RobotCommand]) -> RobotCommand:
    orientation = EulerZXY(roll=0.0, pitch=pitch, yaw=0.0)
    mobility_params = RobotCommandBuilder.mobility_params(
        body_height=height, footprint_R_body=orientation, locomotion_hint=locomotion_hint,
        stair_hint=stair_hint)
    cmd = RobotCommandBuilder.synchro_velocity_command(
        v_x=vx, v_y=vy, v_rot=w, params=mobility_params,
        build_on_command=build_on_command)
    return cmd

def create_orientation_command(roll: float, pitch: float, yaw: float, height: float, build_on_command: Optional[RobotCommand]) -> RobotCommand:
    orientation = EulerZXY(roll=roll, pitch=pitch, yaw=yaw)
    cmd = RobotCommandBuilder.synchro_stand_command(
        body_height=height, footprint_R_body=orientation,
        build_on_command=build_on_command)
    return cmd
