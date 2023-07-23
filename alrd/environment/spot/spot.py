from __future__ import annotations 

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass, asdict 
from enum import Enum
import yaml
from typing import Optional, Tuple

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.util
import numpy as np
from alrd.environment.spot.command import Command 
from alrd.environment.spot.robot_state import SpotState
from alrd.environment.spot.utils import MAX_SPEED, get_hitbox
from alrd.utils import change_frame_2d, Frame2D
from bosdyn.api import estop_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.lease import LeaseClient, ResourceAlreadyClaimedError
from bosdyn.client.robot_command import (RobotCommandBuilder,
                                         RobotCommandClient,
                                         block_for_trajectory_cmd,
                                         blocking_sit, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from scipy.spatial.transform import Rotation as R

@dataclass
class SpotEnvironmentConfig(yaml.YAMLObject):
    yaml_tag=u'!SpotEnvironmentConfig'
    # robot hostname
    hostname: str
    # Boundaries of the environment
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    # Starting pose of the robot relative to the environment initialization pose
    start_x: float
    start_y: float
    start_angle: float

    def __post_init__(self):
        assert self.min_x < self.max_x, "min_x must be less than max_x"
        assert self.min_y < self.max_y, "min_y must be less than max_y"
        assert self.start_x >= self.min_x and self.start_x <= self.max_x, "start_x must be within the environment boundaries"
        assert self.start_y >= self.min_y and self.start_y <= self.max_y, "start_y must be within the environment boundaries"

##### Fixed environment parameters #####
MARGIN = 0.16                       # Margin to the walls within which the robot is stopped (m)
CHECK_TIMEOUT = MARGIN / MAX_SPEED  # Maximum time the boundary check will wait for state reading (s)
STAND_TIMEOUT = 10.0                # Maximum time to wait for the robot to stand up (s)
POSE_TIMEOUT = 10.0                 # Maximum time to wait for the robot to reach a pose (s)
SHUTDOWN_TIMEOUT = 10.0             # Maximum time to wait for the robot to shut down (s)
READ_STATE_SLEEP_PERIOD = 0.01      # Period defining how often we check if a state read request is finished (s)
EXPECTED_STATE_READ_TIME = 0.02     # Expected time for the robot to read its state (s)
COMMAND_DURATION = 0.5              # Duration of regular commands sent to spot (while not replaced by new command) (s)
MAX_MAIN_WAIT = 10.0                # Maximum time the state machine main loop sleeps

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

        blocking_sit(self.command_client, timeout_sec=SHUTDOWN_TIMEOUT, update_frequency=10.)
        self.robot.power_off(cut_immediately=False, timeout_sec=SHUTDOWN_TIMEOUT)
        self.motors_powered = False
        if self.lease_keep_alive:
            self.lease_keep_alive.shutdown()
        self.has_robot_control = False
        self._toggle_estop()
        self.logger.info("Robot powered off")
    
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
        cmd_id = self.command_client.robot_command_async(cmd, end_time_secs=time.time()+POSE_TIMEOUT).result() 
        return block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=POSE_TIMEOUT)

    def _read_robot_state(self, timeout=None) -> Optional[SpotState]:
        # Here we wait till the future is done (Method: Check-until-done) or timeout passes.
        check_until_done_future = self.robot_state_client.get_robot_state_async()
        start = time.time()
        while not check_until_done_future.done() and (timeout is None or time.time() - start < timeout):
            time.sleep(READ_STATE_SLEEP_PERIOD)
        if check_until_done_future.done():
            return SpotState.from_robot_state(time.time(), check_until_done_future.result())
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
    RESET_DONE = 4
    RESET = 5
    CHECK_DONE = 6
    STEP_DONE = 7
    STEP = 8

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

class State(Enum):
    READY = 0
    RUNNING = 1
    STOPPING = 2
    STOPPED = 3
    RESETTING = 4
    WAITING = 5
    SHUTTING_DOWN = 6
    SHUTDOWN = 7

@dataclass(order=True)
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

    def wait(self, timeout=None, sleep_period=0.1) -> Optional[bool]:
        """Wait for the request to be finished.
        Params:
            timeout: timeout in seconds
            sleep_period: maximum period the program waits continuously in seconds. In Windows the program will only react to SIGINT when it leaves the wait."""
        assert timeout is None or timeout > 0, "timeout must be positive"
        assert sleep_period > 0, "sleep_period must be positive"
        start = time.time()
        wait = min(timeout, sleep_period) if timeout is not None else sleep_period
        finished = self.__event.wait(timeout=wait)
        while not finished and (timeout is None or time.time() - start < timeout):
            wait = min(timeout - (time.time() - start), sleep_period) if timeout is not None else sleep_period
            finished = self.__event.wait(timeout=wait)
        return self.__feedback if finished else None

class StopFeedback(Enum):
    FAILED = 0
    SUCCESS = 1
    ONGOING = 2

class SpotGymStateMachine(SpotGymBase):
    """
    Assumes only one thread is calling issue command and reset.
    """
    def __init__(self, config: SpotEnvironmentConfig, monitor_freq=30.):
        assert 1/monitor_freq > EXPECTED_STATE_READ_TIME, "monitor_freq is too high"
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
        self.__body_start_frame = None
        self.__reset_pose = None
        self.config = config
        self.initialize_robot(config.hostname)
    
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
        # determine body pose in vision frame, which is used to reset the robot
        state = self._read_robot_state()
        assert state is not None
        x, y, _, qx, qy, qz, qw = state.pose_of_body_in_vision
        theta = R.from_quat([qx, qy, qz, qw]).as_euler('xyz')[2]
        self.__body_start_frame = Frame2D(x, y, theta)
        reset_point = self.__body_start_frame.inverse(np.array([ self.config.start_x, self.config.start_y ]))
        reset_angle = self.__body_start_frame.angle + self.config.start_angle
        self.__reset_pose = (reset_point[0], reset_point[1], reset_angle)
        self.logger.info("Start position: {}".format(self.__body_start_frame))
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
    
    @property
    def body_start_frame(self):
        return self.__body_start_frame
    
    def wait_threads(self):
        self.__main.join()
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
        if self.state != State.SHUTDOWN:
            success, _ = self._issue_shutdown()
            if not success:
                self.logger.error("Failed to shutdown robot.")
                if self.state != State.SHUTDOWN:
                    self.__state = State.SHUTDOWN
        else:
            success = True
        # make threads wake up and exit
        self.wait_threads()
        self.__isopen = False
        if not success:
            self._shutdown()
        self.logger.info("Closed")

    def __cmd_loop(self):
        try:
            self.__cmd_event.wait()
            while self.__state != State.SHUTDOWN:
                cmd, timelength = self.__current_cmd
                start = time.time()
                self._issue_robot_command(cmd, endtime=time.time()+COMMAND_DURATION)
                time.sleep(max(0.,timelength - EXPECTED_STATE_READ_TIME))
                start_read = time.time()
                new_state = self._read_robot_state(timeout=COMMAND_DURATION - (time.time() - start))
                read_time = time.time() - start_read
                if new_state is not None:
                    oob = not self.is_in_bounds(new_state)
                else:
                    oob = False
                self.__cmd_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.STEP_DONE, (new_state, read_time, oob)))
                self.__cmd_event.wait()
        except Exception as e:
            self.logger.error("Error in command loop")
            traceback.print_exc()
            self._issue_shutdown()

    def _issue_command(self, cmd: Command, timelength: float) -> Tuple[bool, Optional[Tuple[Optional[SpotState], Optional[float], bool]]]:
        request = StateMachineRequest(StateMachineAction.STEP, (cmd, timelength))
        self.__queue.put(request)
        response = request.wait()
        if response is None:
            return False, None
        return response
    
    def __reset_loop(self):
        try:
            self.__reset_event.wait()
            while self.__state != State.SHUTDOWN:
                result = self._issue_goal_pose_command(*self.__reset_pose) 
                self.__reset_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.RESET_DONE, result))
                self.__reset_event.wait()
        except Exception as e:
            self.logger.error("Error in reset loop:")
            traceback.print_exc()
            self._issue_shutdown()
    
    def _issue_reset(self) -> Tuple[bool, None]:
        request = StateMachineRequest(StateMachineAction.RESET)
        self.__queue.put(request)
        response = request.wait()
        if response is None:
            return False, None
        return response

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
    
    def _issue_stop(self) -> bool:
        request = StateMachineRequest(StateMachineAction.STOP)
        self.__queue.put(request)
        response = request.wait()
        if response is None:
            return False, None
        return response
    
    def __shutdown_loop(self):
        try:
            self.__shutdown_event.wait()
            while self.__state != State.SHUTDOWN:
                self.logger.info('Shutdown service received shutdown signal')
                self._shutdown()
                self.__shutdown_event.clear()
                self.__queue.put(StateMachineRequest(StateMachineAction.SHUTDOWN_DONE, True))
                self.__shutdown_event.wait()
        except Exception as e:
            self.logger.error("Error in shutdown loop:")
            traceback.print_exc()
            self.__state = State.SHUTDOWN
            self._shutdown()

    def _issue_shutdown(self) -> Tuple[bool, None]:
        request = StateMachineRequest(StateMachineAction.SHUTDOWN)
        self.__queue.put(request)
        result = request.wait()
        return result if result is not None else (False, None)

    def is_in_bounds(self, state: SpotState) -> bool:
        x, y, _, qx, qy, qz, qw = state.pose_of_body_in_vision
        angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        box = get_hitbox(x, y, angle)
        box = self.body_start_frame.transform(box)
        min_x, min_y = np.min(box, axis=0)
        max_x, max_y = np.max(box, axis=0)
        return min_x > self.config.min_x + MARGIN and max_x < self.config.max_x - MARGIN and \
                min_y > self.config.min_y + MARGIN and max_y < self.config.max_y - MARGIN
    
    def __bounds_srv(self): 
        try:
            self.__check_bounds.wait()
            while self.__state not in {State.SHUTDOWN, State.SHUTTING_DOWN}:
                new_state = self._read_robot_state(timeout = CHECK_TIMEOUT - 1/self.__freq + EXPECTED_STATE_READ_TIME)
                self.__check_bounds.clear()
                if new_state is None:
                    self.logger.info('Bounds srv: timed out')
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
                    if last_read is not None:
                        wait = max(0, 1/self.__freq - (time.time() - last_read) - EXPECTED_STATE_READ_TIME)
                    else:
                        wait = 1/self.__freq - EXPECTED_STATE_READ_TIME
                    if wait <= 0:
                        self.__check_bounds.set()
                        check_ongoing = True
                        continue
                try:
                    request = self.__queue.get(timeout=wait)
                except queue.Empty:
                    continue
                assert isinstance(request, StateMachineRequest)
                if request.action == StateMachineAction.STEP:
                    if self.__state == State.READY:
                        self.__state = State.RUNNING
                        self.__current_cmd = request.value
                        self.__cmd_event.set()
                        latest_order = request
                    else:
                        request.set_feedback((False, None))
                elif request.action == StateMachineAction.STEP_DONE:
                    if self.__state == State.RUNNING:
                        new_state, timer, oob = request.value
                        if new_state is not None:
                            last_read = time.time()
                        latest_order.set_feedback((True, (new_state, timer, oob)))
                        latest_order = None
                        if new_state is None or oob: # next state could not be read or out of bounds
                            if oob:
                                self.logger.info(f"Robot out of bounds after step. state {new_state.pose_of_body_in_vision}. origin {self.__body_start_frame}. Stopping.")
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
                            if latest_order is not None:
                                latest_order.set_feedback((False, (None, None, True)))
                                latest_order = None
                        elif request.value is None:
                            shouldstop = True
                            self.logger.info("Robot state could not be checked. Stopping.")
                            if latest_order is not None:
                                latest_order.set_feedback((False, (None, None, False)))
                                latest_order = None
                        if shouldstop:
                            self.__state = State.STOPPING
                            self.__stop_event.set()
                    elif self.__state == State.WAITING:
                        latest_order.set_feedback((request.value, None))
                        latest_order = None
                        self.__state = State.READY
                        last_read = None
                elif request.action == StateMachineAction.RESET:
                    if self.__state in {State.STOPPED, State.READY}:
                        self.__state = State.RESETTING
                        self.logger.debug('State Machine: Starting reset')
                        self.__reset_event.set()
                        latest_order = request
                    else:
                        request.set_feedback((False, None))
                elif request.action == StateMachineAction.RESET_DONE:
                    self.logger.debug('State Machine: Received reset done')
                    if self.__state == State.RESETTING:
                        if request.value:
                            if check_ongoing:
                                self.__state = State.WAITING
                            else:
                                self.logger.debug(f'State Machine: sending feedback to reset {request.value}')
                                latest_order.set_feedback((request.value, None))
                                latest_order = None
                                self.__state = State.READY
                                last_read = None
                        else:
                            self.logger.warning("Reset failed. Stopping robot.")
                            self.__state = State.STOPPING
                            self.__stop_event.set()
                    self.logger.debug(f'State Machine: state after reset done {self.__state}')
                elif request.action == StateMachineAction.STOP:
                    self.logger.debug('State Machine: received stop instruction')
                    if self.__state == State.READY:
                        self.__state = State.STOPPING
                        self.__stop_event.set()
                        latest_order = request
                    elif self.__state == State.STOPPED:
                        self.logger.debug('State Machine: Already stopped')
                        request.set_feedback((True, None))
                    elif self.__state == State.STOPPING:
                        assert latest_order is None # Note: we enter here only if reset is called after check_done triggered a stop
                        latest_order = request
                    else:
                        request.set_feedback((False, None))
                elif request.action == StateMachineAction.STOP_DONE:
                    if self.__state == State.STOPPING:
                        if request.value:
                            self.__state = State.STOPPED
                            if latest_order is not None:
                                latest_order.set_feedback((True, None))
                                latest_order = None
                        else:
                            self.logger.warning("Stop failed. Shutting down robot.")
                            if latest_order is not None:
                                latest_order.set_feedback((False, None))
                                latest_order = None
                            self.__state = State.SHUTTING_DOWN
                            self.__shutdown_event.set()
                elif request.action == StateMachineAction.SHUTDOWN:
                    if self.__state != State.SHUTTING_DOWN:
                        self.__state = State.SHUTTING_DOWN
                        self.__shutdown_event.set()
                        if latest_order is not None:
                            latest_order.set_feedback((False, None))
                        latest_order = request
                    else:
                        request.set_feedback((False, None))
                elif request.action == StateMachineAction.SHUTDOWN_DONE:
                    self.__state = State.SHUTDOWN
                    latest_order.set_feedback((True, None))
                    latest_order = None
                else:
                    raise ValueError("Invalid action {}".format(request.action))
        except Exception as e:
            self.logger.error("Error in main loop")
            traceback.print_exc()
            if latest_order is not None:
                latest_order.set_feedback((False, None))
            self.__state = State.SHUTDOWN
            self._shutdown()