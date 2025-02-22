"""Mock serial port for testing."""
class SerialException(Exception):
    pass

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
        
    def write(self, data):
        self.written_data.append(data)
        return len(data)
        
    def read(self, size=1):
        # Mock MAVLink heartbeat message
        return b'\xfe\t\x00\x00\x00\x00\x00\x00\x00\x03\x04\x03\x00\x00\x00\x00\x00\x7f'
        
    def inWaiting(self):
        return 18  # Length of mock heartbeat message
        
    def close(self):
        self.is_open = False
        
    def flush(self):
        pass
        
    def reset_input_buffer(self):
        pass
        
    def reset_output_buffer(self):
        pass
