from __future__ import print_function

import argparse
import math
import textwrap
import time
import logging
import traceback
from enum import Enum
from alrd.environment.spot.mobility_command import MobilityCommand

from alrd.utils.xbox.xbox_joystick_factory import XboxJoystickFactory
from alrd.environment.spot.spot import SpotGymBase
from alrd.environment.spot.orientation_command import OrientationCommand

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.util
from bosdyn.api import estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.lease import LeaseClient, ResourceAlreadyClaimedError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.geometry import EulerZXY
from alrd.utils.utils import get_timestamp_str
from pathlib import Path


VELOCITY_BASE_SPEED = 0.5  # m/s
VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
VELOCITY_CMD_DURATION = 0.6  # seconds
HEIGHT_MAX = 0.3  # m
ROLL_OFFSET_MAX = 0.4  # rad
YAW_OFFSET_MAX = 0.7805  # rad
PITCH_OFFSET_MAX = 0.7805  # rad
HEIGHT_CHANGE = 0.1  # m per command


class RobotMode(Enum):
    """RobotMode enum stores the current movement type of the robot.
    """

    Walk = 1
    Sit = 2
    Stand = 3
    Stairs = 4
    Jog = 5
    Amble = 6
    Crawl = 7
    Hop = 8


class SpotXbox(SpotGymBase):
    """SpotXbox class provides mapping between xbox controller commands and Spot API calls.

    Attributes:
        client_name: Common name of this program to use in the SDK calls.
        robot: Instance of the robot.
        command_client: Client for all the robot commands.
        lease_client: Client for the lease management.
        estop_client: Client for the E-Stop functionality.
        estop_keepalive: E-Stop keep-alive object.
        estop_buttons_pressed: Boolean used to determine when E-Stop button combination is released
                               in order to toggle the E-Stop
        mobility_params: Mobility parameters to use in each robot command.
        mode: Current robot movement type as RobotMode enum.
        has_robot_control: Boolean whether program has acquired robot control.
        motors_powered: Boolean whether the robot motors are powered.
        body_height: Current robot body height in meters from normal standing height.
        stand_yaw: Current robot body yaw in radians.
        stand_roll: Current robot body roll in radians.
        stand_pitch: Current robot body pitch in radians.
        stand_height_change: Boolean whether the height is changed in stand mode.
        stand_roll_change: Boolean whether the robot body roll is changed in stand mode.
        stand_pitch_change: Boolean whether the robot body pitch is changed in stand mode.
        stand_yaw_change:Boolean whether the robot body yaw is changed in stand mode.
    """

    def __init__(self):
        super().__init__()
        self.client_name = "XboxControllerClient"
        self.estop_buttons_pressed = False
        self.mobility_params = None

        # Robot state
        self.mode = None

        self.body_height = 0.0
        self.stand_yaw = 0.0
        self.stand_roll = 0.0
        self.stand_pitch = 0.0

        self.stand_height_change = False
        self.stand_roll_change = False
        self.stand_pitch_change = False
        self.stand_yaw_change = False
        self.last_cmd = None
        self.robot_state = None

        self.log_file = None

    def initialize_robot(self, hostname):
        super().initialize_robot(hostname)
        self.mobility_params = spot_command_pb2.MobilityParams(
            locomotion_hint=spot_command_pb2.HINT_AUTO)
        filepath = Path("output/spot") / (get_timestamp_str() + ".txt")
        self.log_file = open(filepath, "w")

        # Print controls
        print(
            textwrap.dedent("""\
| Button Combination | Functionality            |
|--------------------|--------------------------|
| A                  | Walk                     |
| B                  | Stand                    |
| X                  | Sit                      |
| Y                  | Stairs                   |
| LB + :             |                          |
| - D-pad up/down    | Walk height              |
| - D-pad left       | Battery-Change Pose      |
| - D-pad right      | Self right               |
| - Y                | Jog                      |
| - A                | Amble                    |
| - B                | Crawl                    |
| - X                | Hop                      |
|                    |                          |
| If Stand Mode      |                          |
| - Left Stick       |                          |
| -- X               | Rotate body in roll axis |
| -- Y               | Control height           |
| - Right Stick      |                          |
| -- X               | Turn body in yaw axis    |
| -- Y               | Turn body in pitch axis  |
| Else               |                          |
| - Left Stick       | Move                     |
| - Right Stick      | Turn                     |
|                    |                          |
| LB + RB + B        | E-Stop                   |
| Start              | Motor power & Control    |
| Back               | Exit                     |
        """))

        # Describe the necessary steps before one can command the robot.
        print("Before you can command the robot: \n" + \
            "\t1. Acquire a software E-Stop (Left Button + Right Button + B). \n" + \
            "\t2. Obtain a lease and power on the robot's motors (Start button).")

    def _issue_robot_command(self, command, endtime=None):
        """Issues the robot command.

        Args:
            command: Robot command to issue.
        """

        self.last_cmd = command
        super()._issue_robot_command(command, endtime=endtime)

    def _jog(self):
        """Sets robot in Jog mode.
        """

        if self.mode is not RobotMode.Jog:
            self.mode = RobotMode.Jog
            self._reset_height()
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_JOG, stair_hint=0)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            #self._issue_robot_command(cmd)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _amble(self):
        """Sets robot in Amble mode.
        """

        if self.mode is not RobotMode.Amble:
            self.mode = RobotMode.Amble
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_AMBLE, stair_hint=0)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _crawl(self):
        """Sets robot in Crawl mode.
        """

        if self.mode is not RobotMode.Crawl:
            self.mode = RobotMode.Crawl
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_CRAWL, stair_hint=0)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            #self._issue_robot_command(cmd)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _hop(self):
        """Sets robot in Hop mode.
        """

        if self.mode is not RobotMode.Hop:
            self.mode = RobotMode.Hop
            self._reset_height()
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_HOP, stair_hint=0)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            #self._issue_robot_command(cmd)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _stairs(self):
        """Sets robot in Stairs mode.
        """

        if self.mode is not RobotMode.Stairs:
            self.mode = RobotMode.Stairs
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_AUTO, stair_hint=1)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            #self._issue_robot_command(cmd)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _walk(self):
        """Sets robot in Walk mode.
        """

        if self.mode is not RobotMode.Walk:
            self.mode = RobotMode.Walk
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_TROT, stair_hint=0)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            #self._issue_robot_command(cmd)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _stand(self):
        """Sets robot in Stand mode.
        """

        if self.mode is not RobotMode.Stand:
            self.mode = RobotMode.Stand
            self.mobility_params = spot_command_pb2.MobilityParams(
                locomotion_hint=spot_command_pb2.HINT_AUTO, stair_hint=0)

            #cmd = RobotCommandBuilder.synchro_stand_command(params=self.mobility_params)
            #self._issue_robot_command(cmd)
            stand = OrientationCommand(roll=0.0, pitch=0.0, yaw=0.0, height=self.body_height)
            self._issue_robot_command(stand)

    def _sit(self):
        """Sets robot in Sit mode.
        """

        #if self.mode is not RobotMode.Sit:
        #    self.mode = RobotMode.Sit
        #    self.mobility_params = spot_command_pb2.MobilityParams(
        #        locomotion_hint=spot_command_pb2.HINT_AUTO, stair_hint=0)

        #    cmd = RobotCommandBuilder.synchro_sit_command(params=self.mobility_params)
        #    self._issue_robot_command(cmd)
        pass

    def _selfright(self):
        """Executes selfright command, which causes the robot to automatically turn if
        it is on its back.
        """

        #cmd = RobotCommandBuilder.selfright_command()
        #self._issue_robot_command(cmd)
        pass

    def _battery_change_pose(self):
        """Executes the battery-change pose command which causes the robot to sit down if
        standing then roll to its [right]/left side for easier battery changing.
        """

        #cmd = RobotCommandBuilder.battery_change_pose_command(
        #    dir_hint=basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT)
        #self._issue_robot_command(cmd)
        pass

    def _move(self, left_x, left_y, right_x):
        """Commands the robot with a velocity command based on left/right stick values.

        Args:
            left_x: X value of left stick.
            left_y: Y value of left stick.
            right_x: X value of right stick.
        """

        # Stick left_x controls robot v_y
        v_y = -left_x * VELOCITY_BASE_SPEED

        # Stick left_y controls robot v_x
        v_x = left_y * VELOCITY_BASE_SPEED

        # Stick right_x controls robot v_rot
        v_rot = -right_x * VELOCITY_BASE_ANGULAR

        # Recreate mobility_params with the latest information
        self.mobility_params = RobotCommandBuilder.mobility_params(
            body_height=self.body_height, locomotion_hint=self.mobility_params.locomotion_hint,
            stair_hint=self.mobility_params.stair_hint)

        #cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot,
        #                                                   params=self.mobility_params)
        #self._issue_robot_command(cmd, endtime=time.time() + VELOCITY_CMD_DURATION)
        cmd = MobilityCommand(vx=v_x, vy=v_y, w=v_rot, height=self.body_height, pitch=self.stand_pitch,
                              locomotion_hint=self.mobility_params.locomotion_hint, stair_hint=self.mobility_params.stair_hint)
        self._issue_robot_command(cmd, endtime=time.time() + VELOCITY_CMD_DURATION)

    def _orientation_cmd_helper(self, yaw=0.0, roll=0.0, pitch=0.0, height=0.0):
        """Helper function that commands the robot with an orientation command;
        Used by the other orientation functions.

        Args:
            yaw: Yaw of the robot body. Defaults to 0.0.
            roll: Roll of the robot body. Defaults to 0.0.
            pitch: Pitch of the robot body. Defaults to 0.0.
            height: Height of the robot body from normal stand height. Defaults to 0.0.
        """

        if not self.motors_powered:
            return

        #orientation = EulerZXY(yaw, roll, pitch)
        #cmd = RobotCommandBuilder.synchro_stand_command(body_height=height,
        #                                                footprint_R_body=orientation)
        cmd = OrientationCommand(roll=roll, pitch=pitch, yaw=yaw, height=height)
        self._issue_robot_command(cmd, endtime=time.time() + VELOCITY_CMD_DURATION)

    def _change_height(self, direction):
        """Changes robot body height.

        Args:
            direction: 1 to increase height, -1 to decrease height.
        """

        self.body_height = self.body_height + direction * HEIGHT_CHANGE
        self.body_height = min(HEIGHT_MAX, self.body_height)
        self.body_height = max(-HEIGHT_MAX, self.body_height)
        self._orientation_cmd_helper(height=self.body_height)

    def _interp_joy_saturated(self, x, y1, y2):
        """
        Interpolate a value between y1 and y2 based on the position of x between -1 and
        1 (the normalized values the xbox controller classes return.).
        If x is outside [-1, 1], saturate the output correctly to y1 or y2.
        """
        if (x <= -1):
            return y1
        elif (x >= 1):
            return y2

        # Range of x is [-1, 1], so dx is 2.
        slope = (y2 - y1) / 2.0
        return slope * (x + 1) + y1

    def _update_orientation(self, left_x, left_y, right_x, right_y):
        """Updates body orientation in Stand mode.

        Args:
            left_x: X value of left stick.
            left_y: Y value of left stick.
            right_x: X value of right stick.
            right_y: Y value of right stick.
        """

        if left_x != 0.0:
            # Update roll
            self.stand_roll_change = True
            self.stand_roll = self._interp_joy_saturated(left_x, -ROLL_OFFSET_MAX, ROLL_OFFSET_MAX)

        if left_y != 0.0:
            # Update height
            self.stand_height_change = True  # record this change so we reset correctly
            self.body_height = self._interp_joy_saturated(left_y, -HEIGHT_MAX, HEIGHT_MAX)

        if right_x != 0.0:
            # Update yaw
            self.stand_yaw_change = True
            self.stand_yaw = self._interp_joy_saturated(right_x, YAW_OFFSET_MAX, -YAW_OFFSET_MAX)

        if right_y != 0.0:
            # Update pitch
            self.stand_pitch_change = True
            self.stand_pitch = self._interp_joy_saturated(right_y, -PITCH_OFFSET_MAX,
                                                          PITCH_OFFSET_MAX)

        self._orientation_cmd_helper(yaw=self.stand_yaw, roll=self.stand_roll,
                                     pitch=self.stand_pitch, height=self.body_height)

    def _reset_height(self):
        """Resets robot body height to normal stand height.
        """

        self.body_height = 0.0
        self._orientation_cmd_helper(height=self.body_height)
        self.stand_height_change = False

    def _reset_pitch(self):
        """Commands the robot to reset body orientation if tilted up/down.
        Only called in Stand mode.
        """

        self.stand_pitch = 0.0
        self._orientation_cmd_helper(pitch=self.stand_pitch)
        self.stand_pitch_change = False

    def _reset_yaw(self):
        """Commands the robot to reset body orientation if tilted left/right.
        Only called in Stand mode.
        """

        self.stand_yaw = 0.0
        self._orientation_cmd_helper(yaw=self.stand_yaw)
        self.stand_yaw_change = False

    def _reset_roll(self):
        """Commands the robot to reset body orientation if rotated left/right.
        Only called in Stand mode.
        """

        self.stand_roll = 0.0
        self._orientation_cmd_helper(roll=self.stand_roll)
        self.stand_roll_change = False

    def _print_status(self):
        """Prints the current status of the robot: E-Stop, Control, Powered-on, Current Mode.
        """

        # Move cursor back to the start of the line
        print(chr(13), end="")
        if self.estop_keepalive:
            print("E-Stop: Acquired    ", end="")
        else:
            print("E-Stop: Not Acquired", end="")
        if self.has_robot_control:
            print("\tRobot Lease: Acquired    ", end="")
        if self.robot.is_powered_on():
            print("\tRobot Motors: Powered On ", end="")
        if self.mode:
            print("\t\t" + "In Robot Mode: " + self.mode.name, end="")
            num_chars = len(self.mode.name)
            if num_chars < 6:  # 6 is the length of Stairs enum
                print(" " * (6 - num_chars), end="")

    def control_robot(self, frequency):
        """Controls robot from an Xbox controller.

        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        LB + RB + B           -> E-Stop
        A                     -> Walk
        B                     -> Stand
        X                     -> Sit
        Y                     -> Stairs
        D-Pad                 -> cameras
        Back                  -> Exit
        LB +
          D-pad up/down       -> walk height
          D-pad left          -> Battery-Change Pose (roll over)
          D-pad right         -> Self right
          Y                   -> Jog
          A                   -> Amble
          B                   -> Crawl
          X                   -> Hop
        if stand
          Left Stick
            X                 -> rotate body in roll axis
            Y                 -> Control height
          Right Stick
            X                 -> Turn body in yaw axis
            Y                 -> Turn body in pitch axis
        else
          Left Stick          -> Move
          Right Stick         -> Turn
        Start                 -> Motor power and Control

        Args:
            frequency: Max frequency to send commands to robot
        """

        try:
            joy = XboxJoystickFactory.get_joystick()
            currentTime = 0

            while not joy.back():
                start_time = time.time()

                left_x = joy.left_x()
                left_y = joy.left_y()
                right_x = joy.right_x()
                right_y = joy.right_y()

                #handle resets for Stand mode
                if self.mode == RobotMode.Stand:
                    if self.stand_height_change and left_y == 0.0:
                        self._reset_height()
                    if self.stand_roll_change and left_x == 0.0:
                        self._reset_roll()
                    if self.stand_pitch_change and right_y == 0.0:
                        self._reset_pitch()
                    if self.stand_yaw_change and right_x == 0.0:
                        self._reset_yaw()

                # Handle button combinations first
                # If E-Stop button combination is pressed, toggle E-Stop functionality only when
                # buttons are released.
                if joy.left_bumper() and joy.right_bumper() and joy.B():
                    self.estop_buttons_pressed = True
                else:
                    if self.estop_buttons_pressed:
                        if not self.estop_keepalive:
                            self._toggle_estop()
                        self.estop_buttons_pressed = False

                if joy.left_bumper() and joy.dpad_up():
                    self._change_height(1)
                if joy.left_bumper() and joy.dpad_down():
                    self._change_height(-1)
                if joy.left_bumper() and joy.dpad_left():
                    self._battery_change_pose()
                if joy.left_bumper() and joy.dpad_right():
                    self._selfright()

                if joy.Y():
                    if joy.left_bumper():
                        self._jog()
                    else:
                        self._stairs()
                if joy.A():
                    if joy.left_bumper():
                        self._amble()
                    else:
                        self._walk()
                if joy.B() and not joy.right_bumper():
                    if joy.left_bumper():
                        self._crawl()
                    else:
                        self._stand()
                if joy.X():
                    if joy.left_bumper():
                        self._hop()
                    else:
                        self._sit()

                if self.mode == RobotMode.Stand:
                    if left_x != 0.0 or left_y != 0.0 or right_x != 0.0 or right_y != 0.0:
                        self._update_orientation(left_x, left_y, right_x, right_y)
                else:
                    if left_x != 0.0 or left_y != 0.0 or right_x != 0.0:
                        self._move(left_x, left_y, right_x)
                    else:
                        if self.mode == RobotMode.Walk or\
                        self.mode == RobotMode.Amble or\
                        self.mode == RobotMode.Crawl or\
                        self.mode == RobotMode.Jog or\
                        self.mode == RobotMode.Hop:
                            self._move(0.0, 0.0, 0.0)

                if joy.start():
                    self._gain_control()
                    self._power_motors()

                self._print_status()

                self.robot_state = self._read_robot_state()
                if self.mode == RobotMode.Walk:
                    self.print_to_file(currentTime)
                # Make sure we maintain a max frequency of sending commands to the robot
                end_time = time.time()
                delta = 1 / frequency - (end_time - start_time)
                if delta > 0.0:
                    time.sleep(delta)
                currentTime = currentTime + 1/frequency

            joy.close()
        finally:
            # Close out when done
            self._shutdown()

    def print_to_file(self, currentTime):
        self.log_file.write("time {{\n \tvalue: {:.5f} \n}}\n".format(currentTime))
        self.log_file.write(self.last_cmd.to_str())
        self.log_file.write(self.robot_state.to_str())

    def _shutdown(self):
        super()._shutdown()
        if self.log_file is not None:
            self.log_file.close()

def main(argv):
    """Parses command line args.

    Args:
        argv: List of command-line arguments.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=('''
        Use this script to control the Spot robot from an Xbox controller. Press the Back
        controller button to safely power off the robot. Note that the example needs the E-Stop
        to be released. The estop_gui script from the estop SDK example can be used to release
        the E-Stop. Press ctrl-c at any time to safely power off the robot.
        '''))
    parser.add_argument("--max-frequency", default=10, type=int,
                        help="Max frequency in Hz to send commands to robot")
    parser.add_argument("--logging", action='store_true', help="Turn on logging output")

    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if options.verbose else logging.INFO)
    if options.logging:
        bosdyn.client.util.setup_logging(options.verbose)

    max_frequency = options.max_frequency
    if max_frequency <= 0:
        max_frequency = 10

    controller = SpotXbox()
    try:
        controller.initialize_robot(options.hostname)
        print("Initialized robot")
        controller.control_robot(max_frequency)
    except Exception as e:
        traceback.print_exc()
        controller._shutdown()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])