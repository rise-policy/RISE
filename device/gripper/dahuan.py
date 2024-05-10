'''
Control script for Dahuan gripper using modbus connection.
'''

import time
import serial
import numpy as np
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu


class DahuanModbusGripper:
    """
    Control Dahuan Gripper AG-95 using modbus.
    Features include:
            - set force
            - set width
            - Get grasp status.
    """
    def __init__(self, port = '/dev/ttyUSB0'):
        '''
        Initialization.
        
        Parameters
        ----------
        port: str, optional, default: '/dev/ttyUSB0', the port of the Dahuan Gripper.
        '''
        super(DahuanModbusGripper, self).__init__()
        self.master = modbus_rtu.RtuMaster(
            serial.Serial(
                port=port, baudrate=115200, bytesize=8, parity="N", stopbits=1
            )
        )
        self.master.open()
        assert self.master._is_opened, "[Gripper] Port {} needs permission".format(port)
        self.master.set_timeout(2.0)
        self.master.set_verbose(True)
        self.cur_width = 1000
        self.cur_force = 100
        self.init_open()
        self.set_force(100)

    def init_open(self):
        '''
        Initialization the gripper.
        '''
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 0x0100, 2, 0x0001)
        return_data = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0200, 1)
        while return_data != 1:
            return_data = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0200, 1)[0]
            time.sleep(0.1)
        print("[Gripper] Gripper initialized succeed!")

    def set_force(self, force_percent):
        '''
        Set the gripper force.
        
        Parameters
        ----------
        force_percent: int, required, between 20 and 100, the force percent; the corresponding real-world force range is 45N ~ 160N.
        '''
        assert force_percent > 20 and force_percent <= 100
        return_data = self.master.execute(
            1, cst.WRITE_SINGLE_REGISTER, 0x0101, 2, force_percent
        )[1]
        assert return_data == force_percent
        self.force = force_percent

    def set_width(self, width_permillage):
        '''
        Set the gripper width

        Parameters
        ----------
        width_permillage: int, required, between 0 and 1000, the width permillage; the corresponding real-world width range is 0mm ~ 95mm.
        '''
        assert width_permillage >= 0 and width_permillage <= 1000
        return_data = self.master.execute(
            1, cst.WRITE_SINGLE_REGISTER, 0x0103, 2, width_permillage
        )[1]
        assert return_data == width_permillage
        self.width = width_permillage

    def open_gripper(self):
        '''
        Open the gripper.
        '''
        self.set_width(1000)

    def close_gripper(self):
        '''
        Close the gripper.
        '''
        self.set_width(0)

    def get_info(self):
        '''
        Get the current gripper information, including width, current and status.
        '''
        width = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0202, 1)
        current = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0204, 1)
        status = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0x0201, 1)

        return_data = [width[0], current[0], status[0]]
        return return_data
