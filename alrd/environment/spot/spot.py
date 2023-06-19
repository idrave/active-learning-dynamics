from __future__ import print_function

import argparse
import math
import textwrap
import traceback
from dataclasses import dataclass, field
import time
from typing import Optional, Tuple
from enum import Enum
from pathlib import Path

from alrd.environment.spot.command import Command, OrientationCommand, CommandEnum, MobilityCommand
from alrd.environment.spot.robot_state import SpotState
from alrd.utils import get_timestamp_str

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.util
from bosdyn.api import estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.lease import LeaseClient, ResourceAlreadyClaimedError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, block_for_trajectory_cmd, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.geometry import EulerZXY

import gym
import logging
import threading
import numpy as np
import signal
import queue

##### Boundaries of the environment #####
MINX = -1.0
MAXX = 1.0
MINY = -1.0
MAXY = 1.0
MAXSPEED = 1.6
MARGIN = 0.15
MAXTIMEOUT = MARGIN / MAXSPEED
TIMEOUT_TOLERANCE = 1
STAND_TIMEOUT = 10.
POSE_TIMEOUT = 20.
READ_STATE_SLEEP_PERIOD = 0.01
STEPWAIT = 0.5 # maximum fraction of the command period to wait in step fucnction 
RESET_POS_TOLERANCE = 0.1 # tolerance for the robot to be considered in the reset position
RESET_ANGLE_TOLERANCE = np.pi/18 # tolerance for the robot to be considered in the reset angle
COMMAND_DURATION = 0.6
SHUTDOWN_TIMEOUT = 10.0
MAX_MAIN_WAIT = 1.0

class HandleSignal:
    def __init__(self, handler):
        self.handler = handler
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logging.info('Closed with sigint')
        self.handler()

class SpotGymBase(object):
    """SpotGym class initializes the robot and provides methods to control it and read the state.

    Attributes:
        client_name: Common name of this program to use in the SDK calls.
        robot: Instance of the robot.
        command_client: Client for all the robot commands.
        lease_client: Client for the lease management.
        estop_client: Client for the E-Stop functionality.
        estop_keepalive: E-Stop keep-alive object.
        has_robot_control: Boolean whether program has acquired robot control.
        motors_powered: Boolean whether the robot motors are powered.
    """

    def __init__(self):
        super().__init__()
        self.client_name = "SpotGym"
        self.robot = None
        self.command_client = None
        self.lease_client = None
        self.lease_keep_alive = None
        self.estop_client = None
        self.estop_keepalive = None

        # Robot state
        self.has_robot_control = False
        self.motors_powered = False

        self._handler = HandleSignal(self._handle_signal)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def initialize_robot(self, hostname):
        """Initializes SDK from hostname.

        Args:
            hostname: String hostname of the robot.
        """

        sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = sdk.create_robot(hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

    def _start(self):
        if self.estop_keepalive is None:
            self._toggle_estop()
        self._gain_control()
        self._power_motors()

    def _shutdown(self):
        """Returns lease to power off.
        """
        if not self.motors_powered or \
        not self.has_robot_control or \
        not self.estop_keepalive or \
        not self.robot.is_powered_on():
            self.logger.info("Robot is already off!")
            return

        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        self.motors_powered = False
        if self.lease_keep_alive:
            self.lease_keep_alive.shutdown()
        self.has_robot_control = False
        self._toggle_estop()
        self.logger.info("Robot powered off")
    
    def _handle_signal(self):
        self.logger.error("Received sigint shutting down robot")
        self._shutdown()

    def _toggle_estop(self):
        """Toggles on/off E-Stop.
        """

        if not self.estop_keepalive:
            if self.estop_client.get_status().stop_level == estop_pb2.ESTOP_LEVEL_NONE:
                print("Taking E-Stop from another controller")

            #register endpoint with 9 second timeout
            estop_endpoint = bosdyn.client.estop.EstopEndpoint(client=self.estop_client,
                                                               name=self.client_name,
                                                               estop_timeout=9.0)
            estop_endpoint.force_simple_setup()

            self.estop_keepalive = bosdyn.client.estop.EstopKeepAlive(estop_endpoint)
        else:
            self.estop_keepalive.stop()
            self.estop_keepalive.shutdown()
            self.estop_keepalive = None
            #sys.exit('E-Stop')

    def _gain_control(self):
        """Acquires lease of the robot to gain control.
        """

        if self.has_robot_control or not self.estop_keepalive:
            return
        try:
            self.lease_client.acquire()
        except ResourceAlreadyClaimedError as exc:
            print("Another controller " + exc.response.lease_owner.client_name +
                  " has a lease. Close that controller"
                  ", wait a few seconds and press the Start button again.")
            return
        else:
            self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
                self.lease_client, return_at_exit=True)
            self.has_robot_control = True

    def _power_motors(self):
        """Powers the motors on in the robot.
        """

        if self.motors_powered or \
        not self.has_robot_control or \
        not self.estop_keepalive or \
        self.robot.is_powered_on():
            return

        self.robot.power_on(timeout_sec=20)
        self.robot.is_powered_on()
        self.motors_powered = True

    def _issue_robot_command(self, command: Command, endtime=None, blocking=False) -> None:
        """Check that the lease has been acquired and motors are powered on before issuing a command.

        Args:
            command: RobotCommand message to be sent to the robot.
            endtime: Time (in the local clock) that the robot command should stop.
        """
        if not self.has_robot_control:
            print("Must have control by acquiring a lease before commanding the robot.")
            return
        if not self.motors_powered:
            print("Must have motors powered on before commanding the robot.")
            return

        self.command_client.robot_command_async(command.cmd, end_time_secs=endtime)
    
    def _issue_blocking_stand_command(self) -> bool:
        try:
            blocking_stand(self.command_client, timeout_sec=STAND_TIMEOUT, update_frequency=0.1)
            return True
        except Exception as e:
            self.logger.error("Failed to stand robot:")
            traceback.print_exc()
            return False

    def _issue_goal_pose_command(self, x, y, theta) -> bool:
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=x, goal_y=y, goal_heading=theta, frame_name=frame_helpers.VISION_FRAME_NAME
        )
        cmd_id = self.command_client.robot_command_async(cmd, end_time_secs=time.time()+POSE_TIMEOUT).result() # TODO: need to adjust end time accordingly
        return block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=POSE_TIMEOUT)

    def _read_robot_state(self, timeout=None) -> Optional[SpotState]:
        # Here we wait till the future is done (Method: Check-until-done) or timeout passes.
        check_until_done_future = self.robot_state_client.get_robot_state_async()
        start = time.time()
        while not check_until_done_future.done() or (timeout and time.time() - start > timeout):
            time.sleep(READ_STATE_SLEEP_PERIOD)
        if check_until_done_future.done():
            return SpotState(check_until_done_future.result())
        else:
            return None

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

class StateMachineAction(Enum):
    SHUTDOWN_DONE = 0
    SHUTDOWN = 1
    STOP_DONE = 2
    STOP = 3
    #MONITOR_STOP = 4
    #MONITOR_WAITING = 5
    RESET_DONE = 3
    RESET = 4
    CHECK_DONE = 5
    STEP_DONE = 6
    STEP = 7

class State(Enum):
    READY = 0
    RUNNING = 1
    STOPPING = 2
    STOPPED = 3
    RESETTING = 4
    WAITING = 5
    SHUTTING_DOWN = 6
    SHUTDOWN = 7

@dataclass
class StateMachineRequest:
    action: StateMachineAction
    def __init__(self, action: StateMachineAction, value=None):
        self.action = action
        self.__event = threading.Event()
        self.__feedback = None
        self.__value = value
    
    @property
    def value(self):
        return self.__value

    def set_feedback(self, feedback):
        self.__feedback = feedback
        self.__event.set()

    def wait(self, timeout=None) -> Optional[bool]:
        finished = self.__event.wait(timeout=timeout)
        return self.__feedback if finished else None

class StopFeedback(Enum):
    FAILED = 0
    SUCCESS = 1
    ONGOING = 2

class SpotGymStateMachine(SpotGymBase):
    """
    Assumes only one thread is calling issue command and reset.
    """
    def __init__(self, monitor_freq=30.):
        super().__init__()
        self.__state = State.SHUTDOWN
        self.__main = threading.Thread(target=self.__main_loop)
        self.__cmd_thread = threading.Thread(target=self.__cmd_loop)
        self.__reset_thread = threading.Thread(target=self.__reset_loop)
        self.__stop_thread = threading.Thread(target=self.__stop_loop)
        self.__shutdown_thread = threading.Thread(target=self.__shutdown_loop)
        self.__monitor_thread = threading.Thread(target=self.__bounds_srv)
        self.__cmd_event = threading.Event()
        self.__reset_event = threading.Event()
        self.__stop_event = threading.Event()
        self.__shutdown_event = threading.Event()
        self.__check_bounds = threading.Event()
        self.__queue = queue.PriorityQueue()
        self.__current_cmd = None
        self.__isopen = False
        self.__freq = monitor_freq
    
    @property
    def isopen(self):
        return self.__isopen
    
    @property
    def state(self):
        return self.__state

    def start(self):
        if self.isopen:
            return
        assert self.__state == State.SHUTDOWN
        self.logger.info("Starting state machine...")
        self._start()
        self._issue_blocking_stand_command()
        self.__cmd_event.clear()
        self.__reset_event.clear()
        self.__stop_event.clear()
        self.__shutdown_event.clear()
        self.__check_bounds.clear()
        self.__cmd_thread.start()
        self.__reset_thread.start()
        self.__stop_thread.start()
        self.__monitor_thread.start()
        self.__shutdown_thread.start()
        self.__main.start()
        self.__isopen = True
    
    def wait_threads(self):
        self.__cmd_event.set()
        self.__reset_event.set()
        self.__stop_event.set()
        self.__shutdown_event.set()
        self.__check_bounds.set()
        self.logger.info("Waiting for commmand thread to exit...")
        self.__cmd_thread.join()
        self.logger.info("Waiting for reset thread to exit...")
        self.__reset_thread.join()
        self.logger.info("Waiting for stop thread to exit...")
        self.__stop_thread.join()
        self.logger.info("Waiting for monitor thread to exit...")
        self.__monitor_thread.join()
        self.logger.info("Waiting for shutdown thread to exit...")
        self.__shutdown_thread.join()

    def close(self):
        if not self.isopen:
            return
        result = self._issue_shutdown(timeout=SHUTDOWN_TIMEOUT)
        if not result:
            self.abrupt_close()
            return
        self.__main.join()
        # make threads wake up and exit
        self.wait_threads()
        assert self.__state == State.SHUTDOWN
        self.__isopen = False
        self.logger.info("Closed")
    
    def abrupt_close(self):
        """
        Close when shutdown or main loop fail.
        """
        if not self.isopen:
            return
        self._shutdown()
        self.__state = State.SHUTDOWN
        # make threads wake up and exit
        self.wait_threads()
        self.__isopen = False
        self.logger.info("Closed")

    def _handle_signal(self):
        self.close()

    def __cmd_loop(self):
        try:
            self.__cmd_event.wait()
            while self.__state != State.SHUTDOWN:
                cmd, timelength = self.__current_cmd
                self._issue_robot_command(cmd, endtime=time.time()+COMMAND_DURATION)
                time.sleep(timelength)
                start_read = time.time()
                new_state = self._read_robot_state(timeout=timelength * STEPWAIT)
                read_time = time.time() - start_read
                if new_state is not None:
                    oob = self.is_in_bounds(new_state)
                else:
                    oob = False
                self.__cmd_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.STEP_DONE, (new_state, read_time, oob)))
                self.__cmd_event.wait()
        except Exception as e:
            self.logger.error("Error in command loop: {}".format(e))
            self._issue_shutdown()

    def _issue_command(self, cmd: Command, timelength: float) -> Tuple[Optional[SpotState], float, bool]:
        request = StateMachineRequest(StateMachineAction.STEP, (cmd, timelength))
        self.__queue.put(request)
        return request.wait()
    
    def __reset_loop(self):
        try:
            self.__reset_event.wait()
            while self.__state != State.SHUTDOWN:
                result = self._issue_goal_pose_command(0, 0, 0)
                self.__reset_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.RESET_DONE, result))
                self.__reset_event.wait()
        except Exception as e:
            self.logger.error("Error in reset loop:")
            traceback.print_exc()
            self._issue_shutdown()
    
    def _issue_reset(self):
        request = StateMachineRequest(StateMachineAction.RESET)
        self.__queue.put(request)
        return request.wait()

    def __stop_loop(self):
        try:
            self.__stop_event.wait()
            while self.__state != State.SHUTDOWN:
                result = self._issue_blocking_stand_command()
                self.__stop_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.STOP_DONE, result))
                self.__stop_event.wait()
        except Exception as e:
            self.logger.error("Error in stop loop:")
            traceback.print_exc()
            self._issue_shutdown()
    
    def _issue_stop(self) -> StopFeedback:
        request = StateMachineRequest(StateMachineAction.STOP)
        self.__queue.put(request)
        return request.wait()
    
    def __shutdown_loop(self):
        try:
            self.__shutdown_event.wait()
            while self.__state != State.SHUTDOWN:
                self.logger.info('Shutdown service received shutdown signal')
                self._shutdown()
                self.__shutdown_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.SHUTDOWN_DONE))
                self.__shutdown_event.wait()
        except Exception as e:
            self.logger.error("Error in shutdown loop:")
            traceback.print_exc()
            self.abrupt_close()

    def _issue_shutdown(self, timeout=None):
        request = StateMachineRequest(StateMachineAction.SHUTDOWN)
        self.__queue.put(request)
        return request.wait(timeout=timeout)

    def is_in_bounds(self, state: SpotState) -> bool:
        x, y, _, _, _, _, _ = state.pose_of_body_in_vision
        return x > MINX and x < MAXX and y > MINY and y < MAXY
    
    def __bounds_srv(self): 
        try:
            self.__check_bounds.wait()
            while self.__state not in {State.SHUTDOWN, State.SHUTTING_DOWN}:
                new_state = self._read_robot_state(timeout = MAXTIMEOUT - 1/self.__freq)
                self.__check_bounds.clear()
                if new_state is None:
                    self.__queue.put(StateMachineRequest(StateMachineAction.CHECK_DONE, None))
                elif not self.is_in_bounds(new_state):
                    self.__queue.put(StateMachineRequest(StateMachineAction.CHECK_DONE, False))
                else:
                    self.__queue.put(StateMachineRequest(StateMachineAction.CHECK_DONE, True))
                self.__check_bounds.wait()
        except Exception as e:
            self.logger.error("Error in bound monitoring loop")
            traceback.print_exc()
            self._issue_shutdown()

    def __main_loop(self):
        latest_order = None
        self.__state = State.STOPPED
        last_read = None
        check_ongoing = False
        try:
            while self.__state != State.SHUTDOWN:
                wait = MAX_MAIN_WAIT
                if self.state in {State.RUNNING, State.READY} and not check_ongoing:
                    if last_read is None:
                        wait = max(0, 1/self.__freq - (time.time() - last_read))
                    else:
                        wait = 1/self.__freq
                    if wait <= 0:
                        self.__check_bounds.set()
                        check_ongoing = True
                        continue
                try:
                    request = self.__queue.get(timeout=wait)
                except queue.Empty:
                    continue
                if request.action == StateMachineAction.STEP:
                    if self.__state == State.READY:
                        self.__state = State.RUNNING
                        self.__current_cmd = request.value
                        self.__cmd_event.set()
                        latest_order = request
                    else:
                        request.set_feedback(False)
                elif request.action == StateMachineAction.STEP_DONE:
                    if self.__state == State.RUNNING:
                        new_state, timer, oob = request.value
                        last_read = time.time()
                        latest_order.set_feedback((new_state, timer, oob))
                        latest_order = None
                        if new_state is None or oob: # next state could not be read or out of bounds
                            if oob:
                                self.logger.info("Robot out of bounds after step. Stopping.")
                            else:
                                self.logger.info("Robot state could not be read after step. Stopping.")
                            self.__state = State.STOPPING
                            self.__stop_event.set()
                        else:
                            self.__state = State.READY
                elif request.action == StateMachineAction.CHECK_DONE:
                    check_ongoing = False
                    last_read = time.time()
                    if self.__state in {State.RUNNING, State.READY}:
                        shouldstop = False
                        if request.value is False:
                            shouldstop = True
                            self.logger.info("Robot out of bounds. Stopping.")
                        elif request.value is None:
                            shouldstop = True
                            self.logger.info("Robot state could not be read. Stopping.")
                        if shouldstop:
                            self.__state = State.STOPPING
                            self.__stop_event.set()
                            if latest_order is not None:
                                latest_order.set_feedback((None, None, True))
                                latest_order = None
                    elif self.__state == State.WAITING:
                        latest_order.set_feedback(request.value)
                        latest_order = None
                        self.__state = State.READY
                        last_read = None
                elif request.action == StateMachineAction.RESET:
                    if self.__state in {State.STOPPED, State.READY}:
                        self.__state = State.RESETTING
                        self.__reset_event.set()
                        latest_order = request
                    else:
                        request.set_feedback(False)
                elif request.action == StateMachineAction.RESET_DONE:
                    if self.__state == State.RESETTING:
                        if request.value:
                            if check_ongoing:
                                self.__state = State.WAITING
                            else:
                                latest_order.set_feedback(request.value)
                                latest_order = None
                                self.__state = State.READY
                                last_read = None
                        else:
                            self.logger.warning("Reset failed. Stopping robot.")
                            self.__state = State.STOPPING
                            self.__stop_event.set()
                elif request.action == StateMachineAction.STOP:
                    if self.__state == State.READY:
                        self.__state = State.STOPPING
                        self.__stop_event.set()
                        latest_order = request
                    elif self.__state == State.STOPPED:
                        request.set_feedback(True)
                    elif self.__state == State.STOPPING:
                        assert latest_order is None # Note: we enter here only if reset is called after check_done triggered a stop
                        latest_order = request
                    else:
                        request.set_feedback(False)
                elif request.action == StateMachineAction.STOP_DONE:
                    if self.__state == State.STOPPING:
                        if request.value:
                            self.__state = State.STOPPED
                            if latest_order is not None:
                                latest_order.set_feedback(True)
                                latest_order = None
                        else:
                            self.logger.warning("Stop failed. Shutting down robot.")
                            if latest_order is not None:
                                latest_order.set_feedback(False)
                                latest_order = None
                            self._issue_shutdown()
                elif request.action == StateMachineAction.SHUTDOWN:
                    if self.__state != State.SHUTTING_DOWN:
                        self.__state = State.SHUTTING_DOWN
                        self.__shutdown_event.set()
                        if latest_order is not None:
                            latest_order.set_feedback(False)
                        latest_order = request
                    else:
                        request.set_feedback(False)
                elif request.action == StateMachineAction.SHUTDOWN_DONE:
                    self.__state = State.SHUTDOWN
                    latest_order.set_feedback(True)
                    latest_order = None
                else:
                    raise ValueError("Invalid action {}".format(request.action))
        except Exception as e:
            self.logger.error("Error in main loop")
            traceback.print_exc()
            self.abrupt_close()
            if latest_order is not None:
                latest_order.set_feedback(False)
        
    def _handle_signal(self): # TODO: may want to have close function that works for these cases
        super()._handle_signal()
        self.__state = State.SHUTDOWN

class SpotGym(SpotGymStateMachine, gym.Env):
    def __init__(self, hostname, cmd_freq, monitor_freq=30.):
        """
        hostname: The hostname of the robot.
        monitor_freq: Environment's maximum state monitoring frequency for checking position boundaries.
        cmd_freq: Environment's maximum action frequency. Commands will take at least 1/cmd_freq seconds to execute.
        """
        super().__init__()
        assert 1/monitor_freq < MAXTIMEOUT, "Monitor frequency must be higher than 1/{} Hz to ensure safe navigation".format(MAXTIMEOUT)
        self.__cmd_freq = cmd_freq
        self.log_file = None
    
    def initialize_robot(self, hostname):
        super().initialize_robot(hostname)
    
    def start(self):
        super().start()
        filepath = Path("output/spot") / (get_timestamp_str() + ".txt")
        self.log_file = open(filepath, "w")
        self.logger.info(f"Printing to {self.log_file}")

    def close(self):
        super().close()
        self.log_file.close()
    
    def print_to_file(self, command: MobilityCommand, state: SpotState, currentTime):
        self.log_file.write("time {{\n \tvalue: {:.5f} \n}}\n".format(currentTime))
        self.log_file.write(command.to_str())
        self.log_file.write(state.to_str())
    
    def _shutdown(self):
        super()._shutdown()
        if self.log_file is not None:
            self.log_file.close()

    @property
    def should_reset(self):
        return self.state == State.STOPPED
    
    def _step(self, cmd: Command) -> Optional[Tuple[SpotState, float, float]]:
        """
        Apply the command for as long as the command period specified.
        Returns:
            new_state: The new state of the robot after the command is applied.
            cmd_time: The time it took to issue the command + read the state.
            read_time: The time it took to read the state of the robot.
        """
        if not self.isopen or self.should_reset:
            self.logger.warning("Environment is closed or should be reset but step was called.")
            return None
        start_cmd = time.time()
        result = self._issue_command(cmd, 1/self.__cmd_freq)
        if result is None:
            return None
        next_state, read_time, oob = result
        if next_state is None or oob:
            return None 
        elif cmd.cmd_type == CommandEnum.MOBILITY:
            self.print_to_file(cmd, next_state, time.time())
        cmd_time = time.time() - start_cmd
        return next_state, cmd_time, read_time
    
    def _reset(self) -> Optional[SpotState]:
        """
        Reset the robot to the origin.
        """
        if not self.isopen:
            self.logger.warning("Environment is closed but reset was called.")
            return None
            
        self.logger.info("Reset called, stopping robot...")
        result = self._issue_stop()
        if not result:
            self.logger.error("Reset stop failed")
            return None
        input("Press enter to reset the robot to the origin... ")
        # reset position
        result = self._issue_reset()
        if not result:
            self.logger.error("Failed to reset robot position")
            return None
        input("Reset done. Press enter to continue.")
        new_state = self._read_robot_state()
        return new_state