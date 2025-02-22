"""Mock ZED SDK for testing."""
import numpy as np

class Camera:
    def __init__(self):
        self.is_open = False
        
    def open(self, init_params):
        self.is_open = True
        return ERROR_CODE.SUCCESS
        
    def close(self):
        self.is_open = False
        
    def grab(self, runtime_params):
        return ERROR_CODE.SUCCESS
        
    def retrieve_image(self, mat, view):
        mat.data = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
    def retrieve_measure(self, mat, measure):
        if measure == MEASURE.DEPTH:
            mat.data = np.random.rand(720, 1280).astype(np.float32)
        elif measure == MEASURE.XYZRGBA:
            mat.data = np.random.rand(720, 1280, 4).astype(np.float32)

class InitParameters:
    def __init__(self):
        self.camera_resolution = None
        self.depth_mode = None
        self.coordinate_units = None
        self.sdk_cuda_ctx = False

class RuntimeParameters:
    def __init__(self):
        self.sensing_mode = None

class Mat:
    def __init__(self):
        self.data = None
        
    def get_data(self):
        import torch
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data).cuda()
        return self.data

class RESOLUTION:
    HD720 = 2

class DEPTH_MODE:
    ULTRA = 1

class UNIT:
    METER = 1

class SENSING_MODE:
    STANDARD = 1

class VIEW:
    LEFT = 1

class MEASURE:
    DEPTH = 1
    XYZRGBA = 2

class ERROR_CODE:
    SUCCESS = 0
    FAILURE = 1
