"""Mock ZED SDK types for testing."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

class ERROR_CODE:
    SUCCESS = 0
    FAILURE = 1

class RESOLUTION:
    HD720 = 0
    HD1080 = 1

class DEPTH_MODE:
    ULTRA = 0
    STANDARD = 1

class UNIT:
    METER = 0

class COORDINATE_SYSTEM:
    RIGHT_HANDED_Y_UP = 0
    RIGHT_HANDED_Z_UP = 1

class SENSING_MODE:
    STANDARD = 0

class REFERENCE_FRAME:
    WORLD = 0

class SPATIAL_MAP_TYPE:
    MESH = 0

class DETECTION_MODEL:
    MULTI_CLASS_BOX = 0

class OBJECT_FILTERING_MODE:
    NMS3D = 0

class VIEW:
    LEFT = 0

class MEASURE:
    DEPTH = 0
    XYZRGBA = 1
    CONFIDENCE = 2

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

class Pose:
    """Mock ZED SDK Pose class."""
    def __init__(self):
        self._translation = [1.0, 2.0, 3.0]
        self._rotation = [0.1, 0.2, 0.3]
        self._timestamp = 0
        self._confidence = 100
        
    def get_translation(self):
        """Get position vector."""
        class Translation:
            def __init__(self, values):
                self._values = values
            def get(self):
                return self._values
        return Translation(self._translation)
        
    def get_euler_angles(self):
        """Get rotation angles in radians."""
        return self._rotation
        
    def get_pose_data(self):
        """Get full pose data."""
        return {
            'translation': self._translation,
            'rotation': self._rotation,
            'timestamp': self._timestamp,
            'confidence': self._confidence
        }

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
    depth_mode: int = DEPTH_MODE.STANDARD
    coordinate_units: int = UNIT.METER
    sdk_cuda_ctx: bool = True
    coordinate_system: int = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    depth_minimum_distance: float = 0.3
    depth_maximum_distance: float = 40.0

@dataclass
class RuntimeParameters:
    sensing_mode: int = SENSING_MODE.STANDARD
    confidence_threshold: int = 50
    texture_confidence_threshold: int = 90

@dataclass
class PositionalTrackingParameters:
    set_as_static: bool = False
    set_floor_as_origin: bool = True
    enable_area_memory: bool = True
    enable_pose_smoothing: bool = True
    set_gravity_as_origin: bool = True

@dataclass
class MeshFilterParameters:
    remove_duplicate_vertices: bool = True
    min_vertex_dist_meters: float = 0.01

@dataclass
class SpatialMappingParameters:
    resolution_meter: float = 0.1
    range_meter: float = 20.0
    use_chunk_only: bool = True
    max_memory_usage: int = 2048
    save_texture: bool = True
    map_type: int = SPATIAL_MAP_TYPE.MESH
    set_gravity_as_origin: bool = True
    enable_mesh_optimization: bool = True
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        self.mesh_filter_params = MeshFilterParameters()

@dataclass
class ObjectDetectionParameters:
    enable_tracking: bool = True
    enable_mask_output: bool = True
    detection_model: int = DETECTION_MODEL.MULTI_CLASS_BOX
    max_range: float = 40.0
    filtering_mode: int = OBJECT_FILTERING_MODE.NMS3D
    confidence_threshold: int = 50

@dataclass
class ObjectDetectionRuntimeParameters:
    detection_confidence_threshold: int = 50

@dataclass
class Objects:
    def __init__(self):
        """Initialize with mock buoy data."""
        self.object_list = [
            # Navigation gate buoys (Taylor Made Sur-Mark)
            type('Object', (), {
                'position': (-2, 1, 5),
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True,
                'label': 'red_buoy'
            }),
            type('Object', (), {
                'position': (2, 1, 5),
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True,
                'label': 'green_buoy'
            }),
            # Yellow buoy (endangered species)
            type('Object', (), {
                'position': (0, 0.15, 40),
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 90,
                'tracking_state': True,
                'label': 'yellow_buoy'
            })
        ]
        self.timestamp = 0
        self.is_tracked = True
        self.is_new = False

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
        self._pose = Pose()
        self._objects = Objects()
        
    def open(self, init_params):
        self._is_opened = True
        return ERROR_CODE.SUCCESS
        
    def is_opened(self):
        return self._is_opened
        
    def grab(self, runtime_params):
        if not self._is_opened:
            return ERROR_CODE.FAILURE
        return ERROR_CODE.SUCCESS
        
    def retrieve_image(self, mat, view):
        if view == VIEW.LEFT:
            mat.set_data(self._frame.copy())
            
    def retrieve_measure(self, mat, measure):
        if measure == MEASURE.DEPTH:
            mat.set_data(self._depth.copy())
        elif measure == MEASURE.XYZRGBA:
            mat.set_data(self._xyz.copy())
            
    def enable_positional_tracking(self, params=None):
        self._tracking_enabled = True
        return ERROR_CODE.SUCCESS
        
    def enable_spatial_mapping(self, params):
        self._mapping_enabled = True
        return ERROR_CODE.SUCCESS
        
    def enable_object_detection(self, params):
        self._object_detection_enabled = True
        return ERROR_CODE.SUCCESS
        
    def get_position(self, pose, reference_frame):
        """Get current position."""
        pose._translation = self._pose._translation
        pose._rotation = self._pose._rotation
        return ERROR_CODE.SUCCESS
        
    def retrieve_objects(self, objects, runtime_params):
        """Get detected objects."""
        if not self._object_detection_enabled:
            return ERROR_CODE.FAILURE
        objects.object_list = self._objects.object_list
        return ERROR_CODE.SUCCESS
        
    def close(self):
        self._is_opened = False

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
    OBJECT_FILTERING_MODE = OBJECT_FILTERING_MODE
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
