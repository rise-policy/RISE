"""
Flexiv robot ethernet communication python wrapper.
"""

import time
import numpy as np

from device.robot import flexivrdk


class ModeMap:
    idle = "IDLE"
    cart_impedance_online = "NRT_CARTESIAN_MOTION_FORCE"


class FlexivRobot:
    """
    Flexiv Robot Control Class.
    """

    logger_name = "FlexivRobot"

    def __init__(self, robot_ip_address, pc_ip_address):
        """
        Initialize.

        Args:
            robot_ip_address: robot_ip address string
            pc_ip_address: pc_ip address string

        Raises:
            RuntimeError: error occurred when ip_address is None.
        """
        self.mode = flexivrdk.Mode
        self.robot_states = flexivrdk.RobotStates()
        self.plan_info = flexivrdk.PlanInfo()
        self.robot_ip_address = robot_ip_address
        self.pc_ip_address = pc_ip_address
        self.init_robot()


            
    def init_robot(self):
        self.robot = flexivrdk.Robot(self.robot_ip_address, self.pc_ip_address)
        # Clear fault on robot server if any
        if self.robot.isFault():
            print("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            self.robot.clearFault()
            time.sleep(2)
            # Check again
            if self.robot.isFault():
                print("Fault cannot be cleared, exiting ...")
                return
            print("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        print("Enabling robot ...")
        self.robot.enable()

        # Wait for the robot to become operational
        seconds_waited = 0
        while not self.robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                print("Still waiting for robot to become operational, please check that the robot 1) has no fault, 2) is in [Auto (remote)] mode.")

        print("Robot is now operational")

    def enable(self, max_time=10):
        """Enable robot after emergency button is released."""
        self.robot.enable()
        tic = time.time()
        while not self.is_operational():
            if time.time() - tic > max_time:
                return "Robot enable failed"
            time.sleep(0.01)
        return

    def _get_robot_status(self):
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states

    def mode_mapper(self, mode):
        assert mode in ModeMap.__dict__.keys(), "unknown mode name: %s" % mode
        return getattr(self.mode, getattr(ModeMap, mode))

    def get_control_mode(self):
        return self.robot.getMode()

    def set_control_mode(self, mode):
        control_mode = self.mode_mapper(mode)
        self.robot.setMode(control_mode)

    def switch_mode(self, mode, sleep_time=0.01):
        """switch to different control modes.

        Args:
            mode: 'idle', 'cart_impedance_online'
            sleep_time: sleep time to control mode switch time

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if self.get_control_mode() == self.mode_mapper(mode):
            return

        while self.get_control_mode() != self.mode_mapper("idle"):
            self.set_control_mode("idle")
            time.sleep(sleep_time)
        while self.get_control_mode() != self.mode_mapper(mode):
            self.set_control_mode(mode)
            time.sleep(sleep_time)

        print("[Robot] Set mode: {}".format(str(self.get_control_mode())))

    def clear_fault(self):
        self.robot.clearFault()

    def is_fault(self):
        """Check if robot is in FAULT state."""
        return self.robot.isFault()

    def is_stopped(self):
        """Check if robot is stopped."""
        return self.robot.isStopped()

    def is_connected(self):
        """return if connected.

        Returns: True/False
        """
        return self.robot.isConnected()

    def is_operational(self):
        """Check if robot is operational."""
        return self.robot.isOperational()
        
    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcpPose)

    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcpVel)

    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().q)

    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().dq)

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.stop()
        while self.get_control_mode() != self.mode_mapper("idle"):
            time.sleep(0.005)

    def set_max_contact_wrench(self, max_wrench):
        self.switch_mode("cart_impedance_online")
        self.robot.setMaxContactWrench(max_wrench)

    def send_impedance_online_pose(self, tcp):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("cart_impedance_online")
        self.robot.sendCartesianMotionForce(np.array(tcp))

    def send_tcp_pose(self, tcp):
        """
        Send tcp pose.
        """
        self.send_impedance_online_pose(tcp)
