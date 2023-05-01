import json
from gym import spaces
from threading import Event, Lock, Thread
from abc import ABC, abstractmethod
import numpy as np
import logging
import time
from alrd.maze import Maze

logger = logging.getLogger(__name__)

class TopicServer(ABC):
    """
    When callback is called, it will set the event.
    When the get_state is called, the event is cleared
    """
    def __init__(self, name):
        self.name = name
        self.event = Event()
        self.lock = Lock()
        self.recent = None

    def callback(self, info):
        with self.lock:
            self.recent = info
            self.event.set()
    
    def get_state(self, blocking=True, timeout=None):
        """
        Returns the latest information received.
        If blocking is True, waits if no information has been received since the last blocking call.
        If timeout is a positive floating number, the method waits at most such number of seconds and returns the latest value available.
        """
        if blocking:
            self.event.wait(timeout=timeout)
        with self.lock: # TODO should nonblocking lock?
            info = self.recent
            if blocking:
                self.event.clear()
        return info


class AttitudeSub(TopicServer):
    def __init__(self, chassis, freq=20) -> None:
        super().__init__('attitude')
        self.yaw = []
        self.pitch = []
        self.roll = []     
        self.chassis = chassis   
        self.chassis.sub_attitude(freq=freq, callback=self.callback)
    
    def callback(self, info):
        super().callback(info)
        yaw, pitch, roll = info
        self.yaw.append(yaw)
        self.pitch.append(pitch)
        self.roll.append(roll)

    def unsubscribe(self):
        self.chassis.unsub_attitude()
    
    def reset(self):
        self.yaw = []
        self.pitch = []
        self.roll = []     
    
    def get_state(self, blocking=True, timeout=None):
        info = super().get_state(blocking=blocking, timeout=timeout)
        yaw, pitch, roll = info
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }

    def to_dict(self):
        return {
            'yaw': self.yaw,
            'pitch': self.pitch,
            'roll': self.roll
        }

class EscSub(TopicServer):
    def __init__(self, chassis, freq=20) -> None:
        super().__init__('esc')
        self.speed = []
        self.angle = []
        self.timestamp = []
        self.state = []
        self.chassis = chassis   
        self.chassis.sub_esc(freq=freq, callback=self.callback)
    
    def callback(self, info):
        speed, angle, timestamp, state = info
        super().callback((speed, angle, timestamp, state))
        self.speed.append(speed)
        self.angle.append(angle)
        self.timestamp.append(timestamp)
        self.state.append(state)

    def unsubscribe(self):
        self.chassis.unsub_esc()
    
    def reset(self):
        self.speed = []
        self.angle = []
        self.timestamp = []
        self.state = []

    def get_state(self, blocking=True, timeout=None):
        info = super().get_state(blocking=blocking, timeout=timeout)
        speed, angle, timestamp, state = info
        return {
            'speed': speed,
            'angle': angle,
            'timestamp': timestamp,
            'state': state
        }

    def to_dict(self):
        return {
            'speed': self.speed,
            'angle': self.angle,
            'timestamp': self.timestamp,
            'state': self.state
        }

class PositionSub(TopicServer):
    def __init__(self, chassis, freq=50) -> None:
        super().__init__('position')
        self.x = []
        self.y = []
        self.z = []
        self.chassis = chassis
        self.chassis.sub_position(cs=0,freq=freq, callback=self.callback) # coordinate system has 0 at current position
    
    def callback(self, info):
        super().callback(info)
        x, y, z = info
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)

    def unsubscribe(self):
        self.chassis.unsub_position()
    
    def reset(self):
        self.x = []
        self.y = []
        self.z = []

    def get_state(self, blocking=True, timeout=None):
        info = super().get_state(blocking=blocking, timeout=timeout)
        x, y, z = info
        return {
            'x': x,
            'y': y,
            'z': z,
        }

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

class PositionSubBox(PositionSub):
    def __init__(self, chassis, x_low, x_high, y_low, y_high, freq=50) -> None:
        super().__init__(chassis, freq=freq)
        self.pos_low = np.array([x_low, y_low])
        self.pos_high = np.array([x_high, y_high])
    
    def is_out_of_bounds(self, x, y):
        position = np.array([x, y])
        comp = np.logical_or(position < self.pos_low, position > self.pos_high)
        return comp.any()
    
    def callback(self, info):
        x, y, z = info
        oob = self.is_out_of_bounds(x, y)
        TopicServer.callback(self, (*info, oob))
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        if oob:
            logger.debug("Robot out of bounds! (low %s, high %s, current %s) Stopping robot..."%(self.pos_low, self.pos_high, (x, y)))
            self.chassis.drive_speed(0.,0.,0.,timeout=0.01)

    def get_state(self, blocking=True, timeout=None):
        info = TopicServer.get_state(self, blocking=blocking)
        x, y, z, oob = info
        return {
            'x': x,
            'y': y,
            'z': z,
            'oob': oob
        }

class VelocitySub(TopicServer):
    def __init__(self, chassis, freq=20) -> None:
        super().__init__('velocity')
        # global (initial) coordinate system
        self.vgx = [] # x-direction speed 
        self.vgy = [] # y-direction speed
        self.vgz = [] # z-direction speed
        # current body coordinate system
        self.vbx = [] # x-direction speed
        self.vby = [] # y-direction speed
        self.vbz = [] # z-direction speed
        self.chassis = chassis
        self.chassis.sub_velocity(freq=freq, callback=self.callback)

    def callback(self, info):
        super().callback(info)
        vgx, vgy, vgz, vbx, vby, vbz = info
        self.vgx.append(vgx) 
        self.vgy.append(vgy)
        self.vgz.append(vgz)
        self.vbx.append(vbx)
        self.vby.append(vby)
        self.vbz.append(vbz)

    def unsubscribe(self):
        self.chassis.unsub_velocity()

    def reset(self):
        self.vgx = []
        self.vgy = []
        self.vgz = []
        self.vbx = []
        self.vby = []
        self.vbz = []

    def get_state(self, blocking=True, timeout=None):
        info = super().get_state(blocking=blocking, timeout=timeout)
        vgx, vgy, vgz, vbx, vby, vbz = info
        return {
            'vgx': vgx,
            'vgy': vgy,
            'vgz': vgz,
            'vbx': vbx,
            'vby': vby,
            'vbz': vbz,
        }

    def to_dict(self):
        return {
            'vgx': self.vgx,
            'vgy': self.vgy,
            'vgz': self.vgz,
            'vbx': self.vbx,
            'vby': self.vby,
            'vbz': self.vbz,
        }

class IMUSub(TopicServer):
    def __init__(self, chassis, freq=20):
        super().__init__('imu')
        self.chassis = chassis
        self.chassis.sub_imu(freq=freq, callback=self.callback)
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []

    def callback(self, info):
        super().callback(info)
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = info
        self.acc_x.append(acc_x)
        self.acc_y.append(acc_y)
        self.acc_z.append(acc_z)
        self.gyro_x.append(gyro_x)
        self.gyro_y.append(gyro_y)
        self.gyro_z.append(gyro_z)

    def unsubscribe(self):
        self.chassis.unsub_imu()

    def reset(self):
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []

    def get_state(self, blocking=True, timeout=None):
        info = super().get_state(blocking=blocking, timeout=timeout)
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = info
        return {
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
        }
    
    def to_dict(self):
        return {
            'acc_x': self.acc_x,
            'acc_y': self.acc_y,
            'acc_z': self.acc_z,
            'gyro_x': self.gyro_x,
            'gyro_y': self.gyro_y,
            'gyro_z': self.gyro_z,
        }

class ChassisSubAbs(ABC):
    def __init__(self, chassis, position_sub: PositionSub, freq=20) -> None:
        self.attribute_sub = AttitudeSub(chassis, freq=freq)
        self.esc_sub = EscSub(chassis, freq=freq)
        self.velocity_sub = VelocitySub(chassis, freq=freq)
        self.imu_sub = IMUSub(chassis, freq=freq)
        self.position_sub = position_sub
        self.freq = freq
    
    def unsubscribe(self):
        self.attribute_sub.unsubscribe()
        self.esc_sub.unsubscribe()
        self.position_sub.unsubscribe()
        self.velocity_sub.unsubscribe()
        self.imu_sub.unsubscribe()
    
    def reset(self):
        self.attribute_sub.reset()
        self.esc_sub.reset()
        self.position_sub.reset()
        self.velocity_sub.reset()
        self.imu_sub.reset()
    
    def get_state(self, blocking=True, timeout=None):
        return {
            'attitude': self.attribute_sub.get_state(blocking=blocking),
            'esc': self.esc_sub.get_state(blocking=blocking),
            'position': self.position_sub.get_state(blocking=blocking),
            'velocity': self.velocity_sub.get_state(blocking=blocking),
            'imu': self.imu_sub.get_state(blocking=blocking),
            'mode': None, # TODO
            'velocity': self.velocity_sub.get_state(blocking=blocking)
        }

    def to_dict(self):
        return {
            'attitude': self.attribute_sub.to_dict(),
            'esc': self.esc_sub.to_dict(),
            'imu': self.imu_sub.to_dict(),
            'mode': None, # TODO
            'position': self.position_sub.to_dict(),
            'status': None, # TODO
            'velocity': self.velocity_sub.to_dict()
        }

class ChassisSub(ChassisSubAbs):
    def __init__(self, chassis, freq=20) -> None:
        super().__init__(chassis, PositionSub(chassis, freq=freq), freq=freq)
    
class ChassisSubBox(ChassisSubAbs):
    def __init__(self, chassis, x_low, x_high, y_low, y_high, freq=20) -> None:
        super().__init__(chassis, PositionSubBox(chassis, x_low, x_high, y_low, y_high, freq=freq), freq=freq)

class VelocityActionSub(TopicServer):
    def __init__(self):
        super().__init__('velocity_action')

    def callback(self, info):
        """
        Must be called to update with latest velocity (vx, vy, vz) commands
        """
        super().callback(info)

    def get_state(self, blocking=True, timeout=None):
        action = super().get_state(blocking=blocking, timeout=timeout)
        if action is None:
            vx, vy, vz = 0, 0, 0
        else:
            vx, vy, vz = action
        return {
            'vx': vx,
            'vy': vy,
            'vz': vz
        }

class MazeMarginChecker(TopicServer):
    def __init__(self, chassis, position_srv: PositionSub, attitude_srv: AttitudeSub, command_sub: VelocityActionSub, maze: Maze, slide_wall=True, freq=20) -> None:
        super().__init__('maze_margin')
        self.chassis = chassis
        self.commands = command_sub
        self.maze = maze
        self.position_srv = position_srv
        self.attitude_srv = attitude_srv
        self.freq = freq
        self.__started = False
        self.__thread = None
        self.slide_wall = slide_wall
    
    def check_action(self, x, y, z, vx, vy, vz):
        if self.slide_wall:
            v_xy_valid = self.maze.clamp_direction((x,y), z, (vx, vy))
        else:
            v_valid = self.maze.valid_move((x,y), z, (vx, vy))
            if v_valid:
                v_xy_valid = (vx, vy)
            else:
                v_xy_valid = (0, 0)
        return np.array([v_xy_valid[0],v_xy_valid[1],vz])
    
    def __run(self):
        while self.__started:
            action = self.commands.get_state(timeout=1/self.freq)
            position = self.position_srv.get_state(blocking=False)
            angle = self.attitude_srv.get_state(blocking=False)['yaw']
            x, y = position['x'], position['y']
            vx, vy, vz = action['vx'], action['vy'], action['vz']
            v_valid = self.drive_speed(x, y, angle, vx, vy, vz)
            self.callback(v_valid)
    
    def start(self):
        self.__started = True
        self.__thread = Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def join(self):
        if self.__started:
            self.__started = False
            logger.debug('joining maze thread')
            self.__thread.join()
            logger.debug('joined maze thread')
    
    def callback(self, info):
        return super().callback(info)

    def get_state(self, blocking=True, timeout=None):
        return {'v_valid': super().get_state(blocking, timeout=timeout)}
        

class GripperSub(TopicServer):
    def __init__(self, gripper, freq=20) -> None:
        super().__init__('gripper')
        self.status = []
        self.gripper = gripper
        self.gripper.sub_status(freq=freq, callback=self.callback)

    def callback(self, gripper_status):
        super().callback(gripper_status)
        self.status.append(gripper_status)

    def unsubscribe(self):
        self.gripper.unsub_status()
    
    def reset(self):
        self.status = []

    def get_state(self, blocking=True, timeout=None):
        status = super().get_state(blocking=blocking, timeout=timeout)
        return {
            'status': status
        }

    def to_dict(self):
        return {
            'status': self.status
        }
    
class ArmSub(TopicServer):
    def __init__(self, arm, freq=20) -> None:
        super().__init__('arm')
        self.pos_x = []
        self.pos_y = []
        self.arm = arm
        self.arm.sub_position(freq=freq, callback=self.callback)

    def callback(self, position):
        super().callback(position)
        pos_x, pos_y = position
        self.pos_x.append(pos_x)
        self.pos_y.append(pos_y)
    
    def unsubscribe(self):
        self.arm.unsub_position()

    def reset(self):
        self.pos_x = []
        self.pos_y = []
        
    def get_state(self, blocking=True, timeout=None):
        x, y = super().get_state(blocking=blocking, timeout=timeout)
        return {
            'x': x,
            'y': y
        }
    
    def to_dict(self):
        return {
            'x': self.pos_x,
            'y': self.pos_y
        }

class RobotSubAbs(ABC):
    def __init__(self, robot, chassis_sub: ChassisSubAbs, freq=20) -> None:
        self.robot = robot
        self.chassis_sub = chassis_sub
        self.gripper_sub = GripperSub(robot.gripper, freq=freq)
        self.arm_sub = ArmSub(robot.robotic_arm, freq=freq)
        self.freq = freq
        logger.debug('Sleep 1s for subscriber to have reliable values')
        time.sleep(1)
    
    def unsubscribe(self):
        self.chassis_sub.unsubscribe()
        self.gripper_sub.unsubscribe()
        self.arm_sub.unsubscribe()

    def reset(self):
        self.chassis_sub.reset()
        self.gripper_sub.reset()
        self.arm_sub.reset()
        
    def get_state(self, blocking=True, timeout=None):
        return {
            'chassis': self.chassis_sub.get_state(blocking=blocking, timeout=timeout),
            'gripper': self.gripper_sub.get_state(blocking=blocking, timeout=timeout),
            'arm': self.arm_sub.get_state(blocking=blocking, timeout=timeout)
        }

    def to_dict(self):
        return {
            'chassis': self.chassis_sub.to_dict(),
            'gripper': self.gripper_sub.to_dict(),
            'arm': self.arm_sub.to_dict()
        }

class RobotSub(RobotSubAbs):
    def __init__(self, robot, freq=20) -> None:
        super().__init__(robot, ChassisSub(robot.chassis, freq=freq), freq=freq)

class RobotSubBox(RobotSubAbs):
    def __init__(self, robot, x_low, x_high, y_low, y_high, freq=20) -> None:
        super().__init__(robot,ChassisSubBox(robot.chassis, x_low, x_high, y_low, y_high, freq=freq), freq=freq)
    