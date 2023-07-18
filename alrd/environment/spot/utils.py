from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import textwrap
from enum import Enum

DIST_TO_FRONT = 0.55

class Sensor(Enum):
    VISION = 0
    ODOM = 1

class Frame(Enum):
    VISION = 0
    ODOM = 1
    BODY = 2

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)
    
    def fromarray( arr: np.ndarray) -> Vector3D:
        return Vector3D(*arr)

@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=dtype)
    
    def fromarray( arr: np.ndarray) -> Quaternion:
        return Quaternion(*arr)

@dataclass
class SE3Velocity:
    linear: Vector3D
    angular: Vector3D

    def __array__(self, dtype=None) -> np.ndarray:
        return np.concatenate((np.array(self.linear, dtype=dtype), np.array(self.angular, dtype=dtype)), dtype=dtype)
    
    def fromarray( arr: np.ndarray) -> SE3Velocity:
        return SE3Velocity(Vector3D.fromarray(arr[:3]), Vector3D.fromarray(arr[3:]))
    
    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            linear {{
            \tx: {self.linear.x}
            \ty: {self.linear.y}
            \tz: {self.linear.z}
            }}
            angular {{
            \tx: {self.angular.x}
            \ty: {self.angular.y}
            \tz: {self.angular.z}
            }}""")

@dataclass
class CylindricalVelocity:
    r: float # radial velocity
    theta: float # azimuthal velocity
    z: float # vertical velocity

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.r, self.theta, self.z], dtype=dtype)
    
    def fromarray(arr: np.ndarray) -> CylindricalVelocity:
        return CylindricalVelocity(*arr)

@dataclass
class SE3Pose:
    position: Vector3D
    rotation: Quaternion

    def __array__(self, dtype=None) -> np.ndarray:
        return np.concatenate((np.array(self.position, dtype=dtype), np.array(self.rotation, dtype=dtype)), dtype=dtype)
    
    def fromarray(arr: np.ndarray) -> SE3Pose:
        return SE3Pose(Vector3D.fromarray(arr[:3]), Quaternion.fromarray(arr[3:]))
    
    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            position {{
            \tx: {self.position.x}
            \ty: {self.position.y}
            \tz: {self.position.z}
            }}
            rotation {{
            \tw: {self.rotation.w}
            \tx: {self.rotation.x}
            \ty: {self.rotation.y}
            \tz: {self.rotation.z}
            }}""")