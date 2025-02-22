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
from typing import Dict, Optional, Tuple
import numpy as np
import torch
try:
    import pyzed.sl as sl
except ImportError:
    from .types.mock_sl import MockSL as sl

class ZEDCamera:
    def __init__(self):
        """Initialize ZED2i camera with CUDA acceleration."""
        self.zed = sl.Camera()
        
        # Create camera configuration for marine environment
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.QUALITY  # Better for untextured surfaces like water
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        self.init_params.depth_minimum_distance = -1.0  # Use SDK default
        self.init_params.depth_maximum_distance = -1.0  # Use SDK default
        self.init_params.enable_image_enhancement = True  # Enhance image quality
        self.init_params.camera_fps = 30  # Standard FPS for marine applications
        self.init_params.sdk_verbose = 1  # Enable SDK verbose mode
        self.init_params.sdk_gpu_id = -1  # Auto-select most powerful GPU
        self.init_params.depth_stabilization = 1  # Enable depth stabilization (1-100, 0 to disable)
        self.init_params.enable_right_side_measure = False  # Not needed for our use case
        self.init_params.camera_disable_self_calib = False  # Enable auto calibration
        self.init_params.optional_settings_path = ""  # Use default settings
        self.init_params.sensors_required = False  # Don't require additional sensors
        self.init_params.enable_image_validity_check = False  # Skip validity check for performance
        
        # Runtime parameters optimized for water surface filtering
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_depth = True  # Enable depth computation
        self.runtime_params.confidence_threshold = 95  # Default confidence threshold
        self.runtime_params.texture_confidence_threshold = 100  # Maximum texture confidence for water
        self.runtime_params.remove_saturated_areas = True  # Handle water reflections
        self.runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD  # Use world reference frame for consistent mapping
        self.runtime_params.enable_fill_mode = False  # Disable fill mode for better accuracy
        
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
            print("Kamera açık değil!")
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
            # Cleanup on failure
            self.zed.disable_positional_tracking()
            return False
            
        # Wait briefly for tracking to initialize
        import time
        time.sleep(0.5)
        
        # Verify tracking state
        tracking_status = self.zed.get_positional_tracking_status()
        if tracking_status == sl.POSITIONAL_TRACKING_STATE.OFF:
            print("Pozisyon takibi etkin değil! Önce pozisyon takibini etkinleştirin.")
            self.zed.disable_positional_tracking()
            return False
        elif tracking_status != sl.POSITIONAL_TRACKING_STATE.OK:
            print(f"Pozisyon takibi başlatılamadı! Durum: {tracking_status}")
            self.zed.disable_positional_tracking()
            return False
            
        self.tracking_enabled = True
        return True
        
    def enable_spatial_mapping(self) -> bool:
        """Spatial mapping'i etkinleştirir."""
        if not self.zed.is_opened():
            print("Kamera açık değil!")
            return False
            
        if not self.enable_positional_tracking():
            return False
            
        # Verify tracking is actually running
        tracking_status = self.zed.get_positional_tracking_status()
        if tracking_status != sl.POSITIONAL_TRACKING_STATE.OK:
            print(f"Pozisyon takibi hazır değil! Durum: {tracking_status}")
            return False
            
        mapping_params = sl.SpatialMappingParameters()
        mapping_params.set_resolution(sl.MAPPING_RESOLUTION.LOW)  # Better for outdoor/marine
        mapping_params.set_range(sl.MAPPING_RANGE.LONG)  # Extended range for marine environment
        mapping_params.use_chunk_only = True  # Memory efficient mapping
        mapping_params.max_memory_usage = 2048  # 2GB memory limit
        mapping_params.save_texture = True  # Enable texture for visualization
        mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH  # Mesh for better water surface mapping
        mapping_params.reverse_vertex_order = False  # Default vertex order
        
        status = self.zed.enable_spatial_mapping(mapping_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Spatial mapping başlatılamadı: {status}")
            self.disable_positional_tracking()
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
            trans = pose.get_translation().get()
            rot = pose.get_euler_angles()
            return {
                'x': trans[0],
                'y': trans[1],
                'z': trans[2],
                'roll': rot[0],
                'pitch': rot[1],
                'yaw': rot[2]
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
        if self.zed.extract_whole_spatial_map(mesh) == sl.ERROR_CODE.SUCCESS:
            return mesh
        return None
        
    def close(self) -> None:
        """Close the camera connection."""
        if self.object_detection_enabled:
            self.zed.disable_object_detection()
            self.object_detection_enabled = False
        if self.mapping_enabled:
            self.zed.disable_spatial_mapping()
            self.mapping_enabled = False
        if self.tracking_enabled:
            self.zed.disable_positional_tracking()
            self.tracking_enabled = False
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
        detection_params.enable_tracking = True  # Enable object tracking
        detection_params.enable_segmentation = False  # Default value
        detection_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_FAST  # Default model
        detection_params.max_range = -1.0  # Use SDK default
        detection_params.filtering_mode = sl.OBJECT_FILTERING_MODE.NMS3D  # Better 3D filtering
        detection_params.prediction_timeout_s = 0.2  # Fast prediction for real-time
        detection_params.allow_reduced_precision_inference = False  # Full precision for accuracy
        detection_params.instance_module_id = 0  # Default instance module
        # Use default values for optional parameters
        detection_params.batch_trajectories_parameters = sl.BatchParameters()  # Default batch parameters
        detection_params.fused_objects_group_name = ""  # No fused objects group
        detection_params.custom_onnx_file = ""  # No custom ONNX model
        detection_params.custom_onnx_dynamic_input_shape = sl.Resolution(512, 512)  # Default input shape
        
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
            np.ndarray: Point cloud as XYZRGBA numpy array
        """
        point_cloud = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            data = point_cloud.get_data()
            if data is not None:
                # Add alpha channel if needed
                if data.shape[-1] == 3:
                    alpha = np.ones((*data.shape[:-1], 1), dtype=data.dtype)
                    data = np.concatenate([data, alpha], axis=-1)
            return data
        return None
        
    def get_depth_map(self) -> Optional[np.ndarray]:
        """Get depth map data.
        
        Returns:
            Optional[np.ndarray]: Depth map as numpy array
        """
        depth = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            return depth.get_data()
        return None
        
    def get_confidence_map(self) -> Optional[np.ndarray]:
        """Get depth confidence map.
        
        Returns:
            Optional[np.ndarray]: Confidence values (0-100)
        """
        if not self.zed.is_opened():
            return None
        return self.zed.get_confidence_map()
