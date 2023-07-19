from __future__ import annotations
from bosdyn.api.robot_state_pb2 import RobotState
from bosdyn.client.frame_helpers import get_a_tform_b, get_vision_tform_body, get_odom_tform_body, BODY_FRAME_NAME, HAND_FRAME_NAME
from dataclasses import dataclass
from typing import List
import numpy as np
import textwrap
import copy
import math
from enum import Enum
from scipy.spatial.transform import Rotation as R
from alrd.environment.spot.utils import Vector3D, Quaternion, SE3Velocity, SE3Pose

JOINT_NAMES = [
    'fl.hx',
    'fl.hy',
    'fl.kn',
    'fr.hx',
    'fr.hy',
    'fr.kn',
    'hl.hx',
    'hl.hy',
    'hl.kn',
    'hr.hx',
    'hr.hy',
    'hr.kn',
    'arm0.sh0',
    'arm0.sh1',
    'arm0.hr0',
    'arm0.el0',
    'arm0.el1',
    'arm0.wr0',
    'arm0.wr1',
    'arm0.f1x'
]

JOINT_NAME_POS = {name: i for i, name in enumerate(JOINT_NAMES)}

class Sensor(Enum):
    VISION = 0
    ODOM = 1

@dataclass
class JointState:
    name: str
    position: float
    velocity: float
    acceleration: float
    load: float

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.position, self.velocity, self.acceleration, self.load], dtype=dtype)
    
    def fromarray( name, arr: np.ndarray) -> JointState:
        return JointState(name, *arr)
    
    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            joint_states {{
            name: "{self.name}"
            position {{
            \tvalue: {self.position}
            }}
            velocity {{
            \tvalue: {self.velocity}
            }}
            acceleration {{
            \tvalue: {self.acceleration}
            }}
            load {{
            \tvalue: {self.load}
            }}
            }}""")

@dataclass
class KinematicState:
    time: float
    velocity_of_body_in_vision: SE3Velocity
    velocity_of_body_in_odom: SE3Velocity
    pose_of_body_in_vision: SE3Pose
    pose_of_body_in_odom: SE3Pose
    joint_states: List[JointState]

    def __array__(self, dtype=None) -> np.ndarray:
        joint_states = np.empty((len(JOINT_NAMES) * 4), dtype=dtype)
        for joint in self.joint_states:
            ind = JOINT_NAME_POS[joint.name]
            joint_states[ind*4:ind*4+4] = np.array(joint, dtype=dtype)
        return np.concatenate((np.array([self.time], dtype=dtype), np.array(self.velocity_of_body_in_vision, dtype=dtype),
                               np.array(self.velocity_of_body_in_odom, dtype=dtype), np.array(self.pose_of_body_in_vision, dtype=dtype),
                               np.array(self.pose_of_body_in_odom, dtype=dtype), joint_states), dtype=dtype)
    
    @staticmethod
    def size() -> int:
        return 1 + 6 + 6 + 7 + 7 + len(JOINT_NAMES) * 4
    
    @staticmethod
    def from_robot_state(time, robot_state: RobotState) -> KinematicState:
        kinematic_state = robot_state.kinematic_state
        linear_velocity = kinematic_state.velocity_of_body_in_vision.linear
        angular_velocity = kinematic_state.velocity_of_body_in_vision.angular
        velocity_of_body_in_vision = SE3Velocity(Vector3D(linear_velocity.x, linear_velocity.y, linear_velocity.z),
                                                Vector3D(angular_velocity.x, angular_velocity.y, angular_velocity.z))
        linear_velocity = kinematic_state.velocity_of_body_in_odom.linear
        angular_velocity = kinematic_state.velocity_of_body_in_odom.angular
        velocity_of_body_in_odom = SE3Velocity(Vector3D(linear_velocity.x, linear_velocity.y, linear_velocity.z),
                                                Vector3D(angular_velocity.x, angular_velocity.y, angular_velocity.z))
        vision_body_pose = get_vision_tform_body(kinematic_state.transforms_snapshot)
        pose_of_body_in_vision = SE3Pose(Vector3D(vision_body_pose.position.x, vision_body_pose.position.y, vision_body_pose.position.z),
                                         Quaternion(vision_body_pose.rotation.w, vision_body_pose.rotation.x, vision_body_pose.rotation.y, vision_body_pose.rotation.z))
        odom_body_pose = get_odom_tform_body(kinematic_state.transforms_snapshot)
        pose_of_body_in_odom = SE3Pose(Vector3D(odom_body_pose.position.x, odom_body_pose.position.y, odom_body_pose.position.z),
                                        Quaternion(odom_body_pose.rotation.w, odom_body_pose.rotation.x, odom_body_pose.rotation.y, odom_body_pose.rotation.z))
        joint_states = []
        for joint_state in kinematic_state.joint_states:
            joint_states.append(JointState(joint_state.name, joint_state.position.value,
                                           joint_state.velocity.value, joint_state.acceleration.value, joint_state.load.value))
        return KinematicState(time, velocity_of_body_in_vision, velocity_of_body_in_odom, pose_of_body_in_vision, pose_of_body_in_odom, joint_states)
    
    @staticmethod
    def fromarray( arr: np.ndarray) -> KinematicState:
        joint_names = JOINT_NAMES
        assert len(arr) == KinematicState.size()
        velocity_of_body_in_vision = SE3Velocity.fromarray(arr[1:7])
        velocity_of_body_in_odom = SE3Velocity.fromarray(arr[7:13])
        pose_of_body_in_vision = SE3Pose.fromarray(arr[13:20])
        pose_of_body_in_odom = SE3Pose.fromarray(arr[20:27])
        joint_states = []
        for i, joint in enumerate(joint_names):
            joint_states.append(JointState(joint, *arr[27+i*4:27+i*4+4]))
        return KinematicState(arr[0], velocity_of_body_in_vision, velocity_of_body_in_odom, pose_of_body_in_vision, pose_of_body_in_odom, joint_states)

    def to_str(self) -> str:
        s = ""
        for jointState in self.joint_states:
            if jointState.name[:3] != "arm":
                s += str(jointState) + "\n"

        s += "velocity_of_body_in_vision {\n"
        s += str(self.velocity_of_body_in_vision)
        s += "}\n"

        s += "velocity_of_body_in_odom {\n"
        s += str(self.velocity_of_body_in_odom)
        s += "}\n"

        s += "odom_tform_body {\n"
        s += str(self.pose_of_body_in_odom)
        s += "}\n"

        s += "vision_tform_body {\n"
        s += str(self.pose_of_body_in_vision)
        s += "}"
        return s

@dataclass
class ManipulatorState:
    gripper_open_percentage: float
    is_gripper_holding_item: bool
    estimated_end_effector_force_in_hand: Vector3D
    velocity_of_hand_in_vision: SE3Velocity
    velocity_of_hand_in_odom: SE3Velocity
    pose_of_hand: SE3Pose

    def __array__(self, dtype=None) -> np.ndarray:
        return np.concatenate([
            np.array([self.gripper_open_percentage], dtype=dtype),
            np.array([self.is_gripper_holding_item], dtype=dtype),
            np.array(self.estimated_end_effector_force_in_hand, dtype=dtype),
            np.array(self.velocity_of_hand_in_vision, dtype=dtype),
            np.array(self.velocity_of_hand_in_odom, dtype=dtype),
            np.array(self.pose_of_hand, dtype=dtype)
        ])
    
    @staticmethod
    def fromarray(arr: np.ndarray) -> ManipulatorState:
        return ManipulatorState(arr[0], arr[1], Vector3D.fromarray(arr[2:5]), SE3Velocity.fromarray(arr[5:11]),
                                SE3Velocity.fromarray(arr[11:17]), SE3Pose.fromarray(arr[17:24]))
    
    @staticmethod
    def from_robot_state(robot_state: RobotState) -> ManipulatorState:
        manipulator_state = robot_state.manipulator_state
        gripper_open_percentage = manipulator_state.gripper_open_percentage
        is_gripper_holding_item = manipulator_state.is_gripper_holding_item
        estimated_end_effector_force_in_hand = Vector3D(manipulator_state.estimated_end_effector_force_in_hand.x,
                                                        manipulator_state.estimated_end_effector_force_in_hand.y,
                                                        manipulator_state.estimated_end_effector_force_in_hand.z)
        linear_velocity = manipulator_state.velocity_of_hand_in_vision.linear
        angular_velocity = manipulator_state.velocity_of_hand_in_vision.angular
        velocity_of_hand_in_vision = SE3Velocity(Vector3D(linear_velocity.x, linear_velocity.y, linear_velocity.z),
                                                 Vector3D(angular_velocity.x, angular_velocity.y, angular_velocity.z))
        linear_velocity = manipulator_state.velocity_of_hand_in_odom.linear
        angular_velocity = manipulator_state.velocity_of_hand_in_odom.angular
        velocity_of_hand_in_odom = SE3Velocity(Vector3D(linear_velocity.x, linear_velocity.y, linear_velocity.z),
                                                Vector3D(angular_velocity.x, angular_velocity.y, angular_velocity.z))
        pose_of_hand_in_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, BODY_FRAME_NAME, HAND_FRAME_NAME)
        pose_of_hand = SE3Pose(Vector3D(pose_of_hand_in_body.position.x, pose_of_hand_in_body.position.y, pose_of_hand_in_body.position.z),
                               Quaternion(pose_of_hand_in_body.rotation.w, pose_of_hand_in_body.rotation.x, pose_of_hand_in_body.rotation.y, pose_of_hand_in_body.rotation.z))
        return ManipulatorState(gripper_open_percentage, is_gripper_holding_item, estimated_end_effector_force_in_hand,
                                velocity_of_hand_in_vision, velocity_of_hand_in_odom, pose_of_hand)
    
@dataclass
class SpotState:
    
    kinematic_state: KinematicState
    manipulator_state: ManipulatorState
    """Interface for accessing easily relevant bosdyn RobotState fields"""
    @staticmethod
    def from_robot_state(time, robot_state: RobotState) -> SpotState:
        return SpotState(KinematicState.from_robot_state(time, robot_state),
                         ManipulatorState.from_robot_state(robot_state))

    # properties to ease access to kinematic state
    @property
    def velocity_of_body_in_vision(self):
        linear_velocity = self.kinematic_state.velocity_of_body_in_vision.linear
        angular_velocity = self.kinematic_state.velocity_of_body_in_vision.angular
        return linear_velocity.x, linear_velocity.y, linear_velocity.z, angular_velocity.x, angular_velocity.y, angular_velocity.z
    
    @property
    def velocity_of_body_in_odom(self):
        linear_velocity = self.kinematic_state.velocity_of_body_in_odom.linear
        angular_velocity = self.kinematic_state.velocity_of_body_in_odom.angular
        return linear_velocity.x, linear_velocity.y, linear_velocity.z, angular_velocity.x, angular_velocity.y, angular_velocity.z
    
    @property
    def pose_of_body_in_vision(self):
        vision_body_pose = self.kinematic_state.pose_of_body_in_vision
        return (vision_body_pose.position.x, vision_body_pose.position.y, vision_body_pose.position.z,
                vision_body_pose.rotation.x, vision_body_pose.rotation.y, vision_body_pose.rotation.z, vision_body_pose.rotation.w)

    @property
    def pose_of_body_in_odom(self):
        odom_body_pose = self.kinematic_state.pose_of_body_in_odom
        return (odom_body_pose.position.x, odom_body_pose.position.y, odom_body_pose.position.z,
                odom_body_pose.rotation.x, odom_body_pose.rotation.y, odom_body_pose.rotation.z, odom_body_pose.rotation.w)
    
    def to_str(self) -> str:
        return self.kinematic_state.to_str()
    
def modify_2d_state(oldstate: SpotState, arr: np.ndarray, sensor: Sensor = Sensor.VISION):
    x, y, cos, sin, vx, vy, w = arr
    state = copy.deepcopy(oldstate)
    kinematic = state.kinematic_state
    norm = math.sqrt(cos**2 + sin**2)
    angle = math.atan2(sin/norm, cos/norm)
    if sensor == Sensor.VISION:
        pose_of_body = kinematic.pose_of_body_in_vision
        velocity_of_body = kinematic.velocity_of_body_in_vision
    else:
        pose_of_body = kinematic.pose_of_body_in_odom
        velocity_of_body = kinematic.velocity_of_body_in_odom
    pose_of_body.position.x = x
    pose_of_body.position.y = y
    rotation = pose_of_body.rotation
    quat = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    euler = R.from_quat(quat).as_euler('xyz')
    euler[2] = angle
    quat = R.from_euler('xyz', euler).as_quat()
    pose_of_body.rotation = Quaternion(quat[3], quat[0], quat[1], quat[2])
    velocity_of_body.linear.x = vx
    velocity_of_body.linear.y = vy
    velocity_of_body.angular.z = w
    return state