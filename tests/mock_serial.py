"""Mock serial port for testing."""
class Serial:
    def __init__(self, port, baudrate=9600, timeout=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
        self.written_data = []
        
    def write(self, data):
        self.written_data.append(data)
        return len(data)
        
    def read(self, size=1):
        return b'\x00' * size
        
    def close(self):
        self.is_open = False
        
    def flush(self):
        pass
