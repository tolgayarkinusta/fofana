"""Mock ZED SDK types for testing."""
from dataclasses import dataclass, field
import numpy as np

class ERROR_CODE:
    SUCCESS = 0
    FAILURE = 1

class RESOLUTION:
    HD720 = 0
    HD1080 = 1

class DEPTH_MODE:
    NONE = 0
    PERFORMANCE = 1
    QUALITY = 2
    ULTRA = 3
    NEURAL = 4
    NEURAL_PLUS = 5

class UNIT:
    MILLIMETER = 0
    METER = 1

class COORDINATE_SYSTEM:
    IMAGE = 0
    RIGHT_HANDED_Y_UP = 1
    RIGHT_HANDED_Z_UP = 2

class SENSING_MODE:
    STANDARD = 0

class REFERENCE_FRAME:
    WORLD = 0  # The transform of sl.Pose will contain the motion with reference to the world frame
    CAMERA = 1  # The transform of sl.Pose will contain the motion with reference to the previous camera frame

class SPATIAL_MAP_TYPE:
    MESH = 0
    FUSED_POINT_CLOUD = 1

class MAPPING_RESOLUTION:
    HIGH = 0    # Creates a detailed geometry, requires lots of memory
    MEDIUM = 1  # Small variations in the geometry will disappear
    LOW = 2     # Keeps only huge variations, useful for outdoor purposes

class MAPPING_RANGE:
    SHORT = 0   # Only depth close to the camera will be used
    MEDIUM = 1  # Medium depth range
    LONG = 2    # Takes into account objects that are far
    AUTO = 3    # Depth range will be computed based on current camera state

class DETECTION_MODEL:
    MULTI_CLASS_BOX_FAST = 0  # Default model for general purpose detection
    MULTI_CLASS_BOX_MEDIUM = 1  # Better accuracy but slower
    MULTI_CLASS_BOX_ACCURATE = 2  # Best accuracy but slowest
    PERSON_HEAD_BOX = 3  # Specialized for head detection

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
    depth_mode: int = DEPTH_MODE.QUALITY
    coordinate_units: int = UNIT.METER
    coordinate_system: int = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    sdk_verbose: int = 1
    sdk_gpu_id: int = -1
    depth_minimum_distance: float = 0.3
    depth_maximum_distance: float = 40.0
    enable_image_enhancement: bool = True
    camera_fps: int = 30
    depth_stabilization: int = 1  # Default value for depth stabilization (1-100, 0 to disable)
    enable_right_side_measure: bool = False
    camera_disable_self_calib: bool = False
    optional_settings_path: str = ""
    sensors_required: bool = False
    enable_image_validity_check: bool = False

@dataclass
class RuntimeParameters:
    enable_depth: bool = True
    confidence_threshold: int = 95
    texture_confidence_threshold: int = 100
    remove_saturated_areas: bool = True
    measure3D_reference_frame: int = REFERENCE_FRAME.WORLD
    enable_fill_mode: bool = False

@dataclass
class PositionalTrackingParameters:
    set_as_static: bool = False
    set_floor_as_origin: bool = True
    enable_area_memory: bool = True
    enable_pose_smoothing: bool = True
    set_gravity_as_origin: bool = True

@dataclass
class SpatialMappingParameters:
    """Class containing a set of parameters for the spatial mapping module."""
    def __init__(self):
        """Initialize spatial mapping parameters optimized for marine environment."""
        self._resolution = MAPPING_RESOLUTION.LOW  # Better for outdoor/marine
        self._range = MAPPING_RANGE.LONG  # Extended range for marine environment
        self.use_chunk_only = True  # Memory efficient mapping
        self.max_memory_usage = 2048  # 2GB memory limit
        self.save_texture = True  # Enable texture for visualization
        self.map_type = SPATIAL_MAP_TYPE.MESH  # Mesh for better water surface mapping
        self.reverse_vertex_order = False  # Default vertex order
        
    def set_resolution(self, resolution):
        """Set spatial mapping resolution."""
        self._resolution = resolution
        
    def set_range(self, range_value):
        """Set spatial mapping range."""
        self._range = range_value

@dataclass
class BatchParameters:
    """Default batch parameters."""
    pass

@dataclass
class Resolution:
    """Resolution class."""
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

@dataclass
class ObjectDetectionParameters:
    enable_tracking: bool = True
    enable_segmentation: bool = False
    detection_model: int = DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    max_range: float = -1.0
    filtering_mode: int = OBJECT_FILTERING_MODE.NMS3D
    prediction_timeout_s: float = 0.2
    allow_reduced_precision_inference: bool = False
    instance_module_id: int = 0
    batch_trajectories_parameters: BatchParameters = field(default_factory=BatchParameters)
    fused_objects_group_name: str = ""
    custom_onnx_file: str = ""
    custom_onnx_dynamic_input_shape: Resolution = field(default_factory=Resolution)

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
                'label': 'red_buoy',
                'type': 'navigation_gate',
                'distance': 5.0
            }),
            type('Object', (), {
                'position': (2, 1, 5),
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True,
                'label': 'green_buoy',
                'type': 'navigation_gate',
                'distance': 5.0
            }),
            # Yellow buoy (endangered species)
            type('Object', (), {
                'position': (0, 0.15, 10),  # Closer for better detection
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 90,
                'tracking_state': True,
                'label': 'yellow_buoy',
                'type': 'yellow_buoy'
            })
        ]
        self.timestamp = 0
        self.is_tracked = True
        self.is_new = False

class POSITIONAL_TRACKING_STATE:
    """Mock ZED SDK positional tracking states."""
    OFF = "OFF"  # Tracking is disabled
    OK = "OK"  # Tracking is working normally
    FPS_TOO_LOW = "FPS_TOO_LOW"  # FPS too low for accurate tracking
    SEARCHING_FLOOR_PLANE = "SEARCHING_FLOOR_PLANE"  # Looking for floor plane
    UNAVAILABLE = "UNAVAILABLE"  # Tracking unavailable

class Camera:
    """Mock ZED camera."""
    def __init__(self):
        self._is_opened = False
        self._tracking_enabled = False
        self._mapping_enabled = False
        self._object_detection_enabled = False
        self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._depth = np.ones((720, 1280), dtype=np.float32) * 2.0
        self._xyz = np.zeros((720, 1280, 4), dtype=np.float32)  # XYZRGBA format
        self._pose = Pose()
        self._objects = Objects()
        self._tracking_state = POSITIONAL_TRACKING_STATE.OFF
        
    def open(self, init_params):
        """Open camera with initialization parameters."""
        self._is_opened = True
        self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._depth = np.ones((720, 1280), dtype=np.float32) * 2.0
        self._xyz = np.zeros((720, 1280, 4), dtype=np.float32)
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
            data = self._xyz.copy()
            # Add RGB values (white by default)
            data[..., 3] = 255  # Alpha channel
            mat.set_data(data)
            
    def enable_positional_tracking(self, params=None):
        self._tracking_enabled = True
        self._tracking_state = POSITIONAL_TRACKING_STATE.OK
        return ERROR_CODE.SUCCESS
        
    def get_positional_tracking_status(self):
        """Get current positional tracking status."""
        if not self._tracking_enabled:
            return "OFF"
        if self._tracking_state == POSITIONAL_TRACKING_STATE.OFF:
            return "OFF"
        elif self._tracking_state == POSITIONAL_TRACKING_STATE.OK:
            return "OK"
        elif self._tracking_state == POSITIONAL_TRACKING_STATE.FPS_TOO_LOW:
            return "FPS_TOO_LOW"
        elif self._tracking_state == POSITIONAL_TRACKING_STATE.SEARCHING_FLOOR_PLANE:
            return "SEARCHING_FLOOR_PLANE"
        else:
            return "UNAVAILABLE"
        
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
        
    def disable_spatial_mapping(self):
        """Disable spatial mapping."""
        self._mapping_enabled = False
        return ERROR_CODE.SUCCESS
        
    def disable_object_detection(self):
        """Disable object detection."""
        self._object_detection_enabled = False
        return ERROR_CODE.SUCCESS
        
    def disable_positional_tracking(self):
        """Disable positional tracking."""
        self._tracking_enabled = False
        return ERROR_CODE.SUCCESS
        
    def extract_whole_spatial_map(self, mesh):
        """Extract spatial map data."""
        mesh.vertices = self._pose._translation
        mesh.triangles = [0, 1, 2]
        mesh.normals = [(0, 0, 1)] * 3
        return ERROR_CODE.SUCCESS
        
    def get_confidence_map(self):
        """Get depth confidence map."""
        return np.ones((720, 1280), dtype=np.uint8) * 100  # Full confidence


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
    MAPPING_RESOLUTION = MAPPING_RESOLUTION
    MAPPING_RANGE = MAPPING_RANGE
    DETECTION_MODEL = DETECTION_MODEL
    OBJECT_FILTERING_MODE = OBJECT_FILTERING_MODE
    MEASURE = MEASURE
    VIEW = VIEW
    POSITIONAL_TRACKING_STATE = POSITIONAL_TRACKING_STATE
    
    Camera = Camera
    Mat = Mat
    Mesh = Mesh
    Objects = Objects
    Pose = Pose
    InitParameters = InitParameters
    RuntimeParameters = RuntimeParameters
    PositionalTrackingParameters = PositionalTrackingParameters
    SpatialMappingParameters = SpatialMappingParameters
    ObjectDetectionParameters = ObjectDetectionParameters
    ObjectDetectionRuntimeParameters = ObjectDetectionRuntimeParameters
    BatchParameters = BatchParameters
    Resolution = Resolution

# Export classes for tests
__all__ = ['MockSL', 'Camera', 'Mat', 'Mesh', 'Objects', 'InitParameters', 'RuntimeParameters']
