"""Mock ZED SDK types for testing."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

class MockSL:
    """Mock ZED SDK for testing."""
    ERROR_CODE = ERROR_CODE
    RESOLUTION = RESOLUTION
    DEPTH_MODE = DEPTH_MODE
    UNIT = UNIT
    COORDINATE_SYSTEM = COORDINATE_SYSTEM
    SENSING_MODE = SENSING_MODE
    REFERENCE_FRAME = REFERENCE_FRAME
    SPATIAL_MAP_TYPE = SPATIAL_MAP_TYPE
    DETECTION_MODEL = DETECTION_MODEL
    MEASURE = MEASURE
    VIEW = VIEW
    
    Camera = Camera
    Mat = Mat
    Mesh = Mesh
    Objects = Objects
    InitParameters = InitParameters
    RuntimeParameters = RuntimeParameters
    PositionalTrackingParameters = PositionalTrackingParameters
    SpatialMappingParameters = SpatialMappingParameters
    ObjectDetectionParameters = ObjectDetectionParameters
    ObjectDetectionRuntimeParameters = ObjectDetectionRuntimeParameters

# Export classes for tests
__all__ = ['MockSL', 'Camera', 'Mat', 'Mesh', 'Objects', 'InitParameters', 'RuntimeParameters']

class ERROR_CODE:
    SUCCESS = 0
    FAILURE = 1

class RESOLUTION:
    HD720 = 0

class DEPTH_MODE:
    ULTRA = 0
    STANDARD = 1

class UNIT:
    METER = 0

class COORDINATE_SYSTEM:
    RIGHT_HANDED_Y_UP = 0

class SENSING_MODE:
    STANDARD = 0

class REFERENCE_FRAME:
    WORLD = 0

class SPATIAL_MAP_TYPE:
    MESH = 0

class DETECTION_MODEL:
    MULTI_CLASS_BOX = 0

class VIEW:
    LEFT = 0

class MEASURE:
    DEPTH = 0
    XYZRGBA = 1
    CONFIDENCE = 2

class Camera:
    """Mock ZED camera."""
    def __init__(self):
        self._is_opened = False
        self._tracking_enabled = False
        self._mapping_enabled = False
        self._object_detection_enabled = False
        self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._depth = np.ones((720, 1280), dtype=np.float32) * 2.0
        self._xyz = np.zeros((720, 1280, 3), dtype=np.float32)
        self._camera = None  # Reference to ZEDCamera instance
        self._runtime_params = None
        self._current_pose = {
            'translation': [1.0, 2.0, 3.0],  # Fixed test values
            'rotation': [0.1, 0.2, 0.3]      # Fixed test values
        }
        self._brightness_factor = 1.0  # Initial brightness factor
        self._mean_brightness = 0.0  # Initial mean brightness
        
    def open(self, init_params):
        self._is_opened = True
        return ERROR_CODE.SUCCESS
        
    def is_opened(self):
        return self._is_opened
        
    def grab(self, runtime_params):
        if not self._is_opened:
            return ERROR_CODE.FAILURE
            
        self._runtime_params = runtime_params
            
        # Update pose if tracking enabled
        if self._tracking_enabled and self._camera:
            self._camera.current_pose = self._current_pose.copy()
            self._camera.tracking_enabled = True
            
            # Update brightness factor based on frame mean
            if hasattr(self, '_frame'):
                mean_brightness = np.mean(self._frame)
                if mean_brightness < self._camera.min_brightness:
                    self._brightness_factor = min(2.0, self._brightness_factor * 1.1)
                elif mean_brightness > self._camera.max_brightness:
                    self._brightness_factor = max(0.5, self._brightness_factor * 0.9)
            
        return ERROR_CODE.SUCCESS
        
    def retrieve_image(self, mat, view):
        if view == VIEW.LEFT:
            frame = self._frame.copy() if hasattr(self, '_frame') else np.zeros((720, 1280, 3), dtype=np.uint8)
            # Calculate mean brightness before adjustment
            self._mean_brightness = np.mean(frame)
            # Apply brightness adjustment
            frame = frame.astype(np.float32)
            frame = frame * self._brightness_factor
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Ensure frame is in RGB format and properly contiguous
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            # Update brightness factor based on mean brightness
            if self._mean_brightness < 30:
                self._brightness_factor = min(2.0, self._brightness_factor * 1.1)
            elif self._mean_brightness > 240:
                self._brightness_factor = max(0.5, self._brightness_factor * 0.9)
            mat.set_data(frame.copy())
            
    def retrieve_measure(self, mat, measure):
        if measure == MEASURE.DEPTH:
            depth = self._depth.copy() if hasattr(self, '_depth') else np.ones((720, 1280), dtype=np.float32)
            mat.set_data(depth[:,:,None])
        elif measure == MEASURE.XYZRGBA:
            xyz = self._xyz.copy() if hasattr(self, '_xyz') else np.zeros((720, 1280, 3), dtype=np.float32)
            mat.set_data(np.concatenate([xyz, np.zeros((720, 1280, 1))], axis=2))
            
    def set_test_frame(self, frame):
        """Set test frame data."""
        self._frame = frame.copy()
        
    def set_test_depth(self, depth):
        """Set test depth data."""
        self._depth = depth.copy()
        
    def set_test_xyz(self, xyz):
        """Set test point cloud data."""
        self._xyz = xyz.copy()
            
    def get_position(self, pose, ref_frame):
        if not self._tracking_enabled:
            return ERROR_CODE.FAILURE
            
        # Set translation and rotation from current pose
        class Translation:
            def __init__(self, values):
                self._values = values
            def get(self):
                return self._values
                
        pose.get_translation = lambda: Translation(self._current_pose['translation'])
        pose.get_euler_angles = lambda: self._current_pose['rotation']
            
        return ERROR_CODE.SUCCESS
        
    def enable_positional_tracking(self, params=None):
        self._tracking_enabled = True
        # Store reference to camera instance
        if params and hasattr(params, 'camera'):
            self._camera = params.camera
            self._camera.tracking_enabled = True
            self._camera.current_pose = {
                'translation': [1.0, 2.0, 3.0],  # Fixed test values
                'rotation': [0.1, 0.2, 0.3]      # Fixed test values
            }
        return ERROR_CODE.SUCCESS
        
    def enable_spatial_mapping(self, params):
        self._mapping_enabled = True
        return ERROR_CODE.SUCCESS
        
    def enable_object_detection(self, params):
        self._object_detection_enabled = True
        return ERROR_CODE.SUCCESS
        
    def disable_positional_tracking(self):
        self._tracking_enabled = False
        
    def disable_spatial_mapping(self):
        self._mapping_enabled = False
        
    def disable_object_detection(self):
        self._object_detection_enabled = False
        
    def close(self):
        self._is_opened = False
        
    def set_test_frame(self, frame):
        """Set test frame data."""
        self._frame = frame
        
    def set_test_depth(self, depth):
        """Set test depth data."""
        self._depth = depth
        
    def set_test_xyz(self, xyz):
        """Set test point cloud data."""
        self._xyz = xyz

@dataclass
class Mat:
    def __init__(self):
        self._data = None
        
    def get_data(self):
        """Get the underlying data."""
        if self._data is None:
            return None
        return self._data.copy()
        
    def set_data(self, data):
        """Set the underlying data."""
        if data is None:
            self._data = None
        else:
            self._data = data.copy()

@dataclass
class Pose:
    def __init__(self):
        self._translation = [1.0, 2.0, 3.0]
        self._rotation = [0.1, 0.2, 0.3]
    
    def get_translation(self):
        class Translation:
            def __init__(self, values):
                self._values = values
            def get(self):
                return self._values
        return Translation(self._translation)
        
    def get_euler_angles(self):
        return self._rotation

@dataclass
class Mesh:
    def __init__(self):
        self.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]  # Simple triangle mesh
        self.triangles = [0, 1, 2]
        self.normals = [(0, 0, 1)] * 3
        self.uv = [(0, 0), (1, 0), (0, 1)]
        self.texture = None

@dataclass
class InitParameters:
    camera_resolution: int = RESOLUTION.HD720
    depth_mode: int = DEPTH_MODE.ULTRA
    coordinate_units: int = UNIT.METER
    sdk_cuda_ctx: bool = True
    coordinate_system: int = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

@dataclass
class RuntimeParameters:
    sensing_mode: int = SENSING_MODE.STANDARD

@dataclass
class PositionalTrackingParameters:
    set_as_static: bool = False
    set_floor_as_origin: bool = True
    enable_area_memory: bool = True
    enable_pose_smoothing: bool = True
    set_gravity_as_origin: bool = True

@dataclass
class SpatialMappingParameters:
    resolution_meter: float = 0.1
    range_meter: float = 20.0
    use_chunk_only: bool = True
    max_memory_usage: int = 2048
    save_texture: bool = True
    map_type: int = SPATIAL_MAP_TYPE.MESH

@dataclass
class ObjectDetectionParameters:
    enable_tracking: bool = True
    enable_mask_output: bool = True
    detection_model: int = DETECTION_MODEL.MULTI_CLASS_BOX
    max_range: float = 20.0

@dataclass
class ObjectDetectionRuntimeParameters:
    detection_confidence_threshold: float = 50

@dataclass
class Objects:
    def __init__(self):
        self.object_list = []
        self.timestamp = 0
        self.is_tracked = True
        self.is_new = False
