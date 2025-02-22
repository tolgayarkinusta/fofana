"""Mock serial port for testing."""
class SerialException(Exception):
    pass

class MAVLink:
    """Mock MAVLink class."""
    def __init__(self, port):
        self.port = port
        
    def command_long_send(self, target_system, target_component, command, confirmation,
                         param1, param2, param3, param4, param5, param6, param7):
        """Mock command_long_send."""
        if command == 400:  # MAV_CMD_COMPONENT_ARM_DISARM
            self.port._armed = bool(param1)
        return True

class Serial:
    def __init__(self, port, baudrate=9600, timeout=None, dsrdtr=None, rtscts=None, xonxoff=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.dsrdtr = dsrdtr
        self.rtscts = rtscts
        self.xonxoff = xonxoff
        self.is_open = True
        self.written_data = []
        self.target_system = 1
        self.target_component = 1
        self._armed = False
        self.mav = MAVLink(self)
        
    def write(self, data):
        """Handle MAVLink messages."""
        self.written_data.append(data)
        # Parse command ID from MAVLink message
        if len(data) > 8:  # MAVLink v1 message
            command_id = int.from_bytes(data[8:10], byteorder='little')
            if command_id == 400:  # MAV_CMD_COMPONENT_ARM_DISARM
                self._armed = bool(data[10])  # First param is arm (1) / disarm (0)
        return len(data)
        
    def read(self, size=1):
        """Return mock MAVLink messages."""
        # Mock heartbeat message
        heartbeat = b'\xfe\t\x00\x00\x00\x00\x00\x00\x00\x03\x04\x03\x00\x00\x00\x00\x00\x7f'
        # Mock servo output message
        servo = b'\xfe\x15\x00\x00\x00\x00\x00\x00\x00\x24\x00\xdc\x05\xdc\x05\xdc\x05\xdc\x05\xdc\x05\xdc\x05\xdc\x05\xdc\x05\x00\x00\x7f'
        return heartbeat if size < 20 else servo
        
    def inWaiting(self):
        """Return number of bytes waiting."""
        return 18  # Length of mock heartbeat message
        
    def close(self):
        """Close the serial port."""
        self.is_open = False
        
    def flush(self):
        """Flush buffers."""
        pass
        
    def reset_input_buffer(self):
        """Clear input buffer."""
        pass
        
    def reset_output_buffer(self):
        """Clear output buffer."""
        pass
        
    def motors_armed_wait(self):
        """Wait for motors to arm."""
        return True
        
    def motors_disarmed_wait(self):
        """Wait for motors to disarm."""
        return True
        
    def wait_heartbeat(self):
        """Wait for heartbeat message."""
        return True
        
    def recv_match(self, type=None, blocking=False):
        """Mock message matching."""
        if type == 'SERVO_OUTPUT_RAW':
            return {'servo1_raw': 1500, 'servo2_raw': 1500}
        return None
