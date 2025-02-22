"""
ZED2i kamera entegrasyon modülü (CUDA hızlandırma destekli).

Özellikler:
- HD720 çözünürlükte görüntü yakalama
- CUDA destekli derinlik algılama
- Stereo görüş ile 3B nokta bulutu oluşturma
- SLAM tabanlı konum takibi
- Spatial mapping ile çevre haritalama
- Gerçek zamanlı görüntü işleme için optimize edilmiş
"""
try:
    import pyzed.sl as sl
except ImportError:
    from .types.mock_sl import MockSL as sl
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List

class ZEDCamera:
    def __init__(self):
        """Initialize ZED2i camera with CUDA acceleration."""
        self.zed = sl.Camera()
        
        # Create camera configuration for marine environment
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.STANDARD  # Better for obstacle detection
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.sdk_cuda_ctx = True  # Enable CUDA context sharing
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params.depth_minimum_distance = 0.3  # Minimum 30cm
        self.init_params.depth_maximum_distance = 40.0  # Maximum 40m for buoy detection
        
        # Runtime parameters optimized for water surface filtering
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        self.runtime_params.confidence_threshold = 50  # Filter unstable depth measurements
        self.runtime_params.texture_confidence_threshold = 90  # High threshold for water surface
        
        # SLAM and mapping status
        self.tracking_enabled = False
        self.mapping_enabled = False
        self.object_detection_enabled = False
        
    def open(self) -> bool:
        """Open the camera connection.
        
        Returns:
            bool: True if camera opened successfully
        """
        status = self.zed.open(self.init_params)
        return status == sl.ERROR_CODE.SUCCESS
        
    def enable_positional_tracking(self) -> bool:
        """Enable SLAM-based positional tracking.
        
        Returns:
            bool: True if tracking enabled successfully
        """
        if not self.zed.is_opened():
            return False
            
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.set_as_static = False
        tracking_params.set_floor_as_origin = True
        tracking_params.enable_area_memory = True
        tracking_params.enable_pose_smoothing = True
        tracking_params.set_gravity_as_origin = True
        
        status = self.zed.enable_positional_tracking(tracking_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Pozisyon takibi başlatılamadı: {status}")
            return False
            
        self.tracking_enabled = True
        return True
        
    def enable_spatial_mapping(self, resolution: float = 0.1, range: float = 20.0) -> bool:
        """Enable spatial mapping for environment reconstruction.
        
        Args:
            resolution: Spatial mapping resolution in meters
            range: Maximum mapping range in meters
            
        Returns:
            bool: True if mapping enabled successfully
        """
        if not self.zed.is_opened() or not self.tracking_enabled:
            return False
            
        mapping_params = sl.SpatialMappingParameters()
        mapping_params.resolution_meter = resolution
        mapping_params.range_meter = range
        mapping_params.use_chunk_only = True
        mapping_params.max_memory_usage = 2048  # 2GB memory limit
        mapping_params.save_texture = True  # Enable texture saving for visualization
        mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH  # Use mesh for better accuracy
        
        # Configure for marine environment
        mapping_params.set_gravity_as_origin = True  # Use gravity for stable water surface reference
        mapping_params.enable_mesh_optimization = True  # Better noise filtering
        mapping_params.mesh_filter_params.remove_duplicate_vertices = True  # Clean mesh
        mapping_params.mesh_filter_params.min_vertex_dist_meters = 0.01  # 1cm minimum vertex distance
        
        status = self.zed.enable_spatial_mapping(mapping_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Spatial mapping başlatılamadı: {status}")
            return False
            
        self.mapping_enabled = True
        return True
        
    def get_position(self) -> Optional[Dict[str, float]]:
        """Get current camera position and orientation.
        
        Returns:
            Optional[Dict[str, float]]: Position and orientation data
                {
                    'x': x position in meters,
                    'y': y position in meters,
                    'z': z position in meters,
                    'roll': roll angle in radians,
                    'pitch': pitch angle in radians,
                    'yaw': yaw angle in radians
                }
        """
        if not self.tracking_enabled:
            return None
            
        pose = sl.Pose()
        if self.zed.get_position(pose, sl.REFERENCE_FRAME.WORLD) == sl.ERROR_CODE.SUCCESS:
            return {
                'x': pose.get_translation().get()[0],
                'y': pose.get_translation().get()[1],
                'z': pose.get_translation().get()[2],
                'roll': pose.get_euler_angles()[0],
                'pitch': pose.get_euler_angles()[1],
                'yaw': pose.get_euler_angles()[2]
            }
        return None
        
    def get_spatial_map(self) -> Optional[sl.Mesh]:
        """Get current spatial map.
        
        Returns:
            Optional[sl.Mesh]: 3D mesh of the environment
        """
        if not self.mapping_enabled:
            return None
            
        mesh = sl.Mesh()
        self.zed.extract_whole_spatial_map(mesh)
        return mesh
        
    def close(self) -> None:
        """Close the camera connection."""
        if self.object_detection_enabled:
            self.zed.disable_object_detection()
        if self.mapping_enabled:
            self.zed.disable_spatial_mapping()
        if self.tracking_enabled:
            self.zed.disable_positional_tracking()
        self.zed.close()
        
    def get_frame(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """Capture and retrieve camera frame with depth and pose.
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, float]]]:
                RGB frame as CUDA tensor
                Depth map as CUDA tensor
                Camera pose data (if tracking enabled)
        """
        image = sl.Mat()
        depth = sl.Mat()
        
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            
            # Convert to CUDA tensors
            # Try to move tensors to CUDA if available
            frame_gpu = torch.from_numpy(image.get_data())
            depth_gpu = torch.from_numpy(depth.get_data())
            try:
                frame_gpu = frame_gpu.cuda()
                depth_gpu = depth_gpu.cuda()
            except RuntimeError:
                # CUDA not available, use CPU tensors
                pass
            
            # Get pose if tracking enabled
            pose_data = self.get_position() if self.tracking_enabled else None
            
            return frame_gpu, depth_gpu, pose_data
        return None, None
        
    def enable_object_detection(self) -> bool:
        """Enable object detection with tracking and mask output.
        
        Returns:
            bool: True if object detection enabled successfully
        """
        if not self.zed.is_opened():
            return False
            
        detection_params = sl.ObjectDetectionParameters()
        detection_params.enable_tracking = True
        detection_params.enable_mask_output = True
        detection_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX
        detection_params.max_range = 40.0  # Extended range for buoy detection
        detection_params.filtering_mode = sl.OBJECT_FILTERING_MODE.NMS3D  # Better 3D filtering
        detection_params.confidence_threshold = 50  # Match depth confidence
        
        status = self.zed.enable_object_detection(detection_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Nesne tespiti başlatılamadı: {status}")
            return False
            
        self.object_detection_enabled = True
        return True
        
    def get_objects(self) -> Optional[sl.Objects]:
        """Get detected objects with tracking information.
        
        Returns:
            Optional[sl.Objects]: Detected objects with position and tracking
        """
        if not self.object_detection_enabled:
            return None
            
        objects = sl.Objects()
        runtime_params = sl.ObjectDetectionRuntimeParameters()
        runtime_params.detection_confidence_threshold = 50
        
        if self.zed.retrieve_objects(objects, runtime_params) == sl.ERROR_CODE.SUCCESS:
            return objects
        return None
        
    def get_object_color_confidence(self, position: Tuple[float, float, float]) -> Dict[str, float]:
        """Get color confidence values for an object at given position.
        
        Args:
            position: (x, y, z) position in world coordinates
            
        Returns:
            dict: Color confidence values (0-100)
        """
        x, _, z = position
        distance = np.sqrt(x**2 + z**2)
        
        if distance < 1.83:  # First gate
            return {'red': 90 if x < 0 else 10, 'green': 90 if x > 0 else 10}
        elif distance < 30.48:  # Speed gates
            if abs(x) < 1:  # Center area
                return {'black': 80, 'blue': 20}
            return {'red': 85 if x < 0 else 10, 'green': 85 if x > 0 else 10}
        else:  # Path gates
            if abs(x) < 1:  # Center area
                return {'yellow': 70}
            return {'red': 75 if x < 0 else 10, 'green': 75 if x > 0 else 10}
        
    def get_point_cloud(self) -> np.ndarray:
        """Get 3D point cloud data.
        
        Returns:
            np.ndarray: Point cloud as numpy array
        """
        point_cloud = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            return point_cloud.get_data()
        return None
