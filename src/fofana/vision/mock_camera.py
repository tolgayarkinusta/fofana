"""Mock ZED camera implementation for testing."""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from .types.mock_sl import (
    Camera, Mat, Mesh, Objects, InitParameters, RuntimeParameters,
    PositionalTrackingParameters, SpatialMappingParameters,
    ObjectDetectionParameters, ObjectDetectionRuntimeParameters,
    ERROR_CODE
)

class MockZEDCamera:
    def __init__(self):
        self.is_open = False
        self.tracking_enabled = False
        self.mapping_enabled = False
        self.detection_enabled = False
        
        # Initialize mock objects
        self.init_params = InitParameters()
        self.runtime_params = RuntimeParameters()
        self.camera = Camera()
        
    def open(self, init_params: Optional[InitParameters] = None) -> ERROR_CODE:
        """Open camera with optional initialization parameters."""
        self.is_open = True
        if init_params:
            self.init_params = init_params
        return ERROR_CODE.SUCCESS
        
    def enable_positional_tracking(self, params: Optional[PositionalTrackingParameters] = None) -> ERROR_CODE:
        """Enable positional tracking with optional parameters."""
        self.tracking_enabled = True
        return ERROR_CODE.SUCCESS
        
    def enable_spatial_mapping(self, params: Optional[SpatialMappingParameters] = None) -> ERROR_CODE:
        """Enable spatial mapping with optional parameters."""
        self.mapping_enabled = True
        return ERROR_CODE.SUCCESS
        
    def enable_object_detection(self, params: Optional[ObjectDetectionParameters] = None) -> ERROR_CODE:
        """Enable object detection with optional parameters."""
        self.detection_enabled = True
        return ERROR_CODE.SUCCESS
        
    def get_frame(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Return mock frame data on CUDA."""
        frame = torch.zeros((1080, 1920, 3), dtype=torch.uint8).cuda()
        depth = torch.zeros((1080, 1920), dtype=torch.float32).cuda()
        pose = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        return frame, depth, pose
        
    def get_objects(self) -> Optional[Objects]:
        """Return mock detected objects."""
        if not self.detection_enabled:
            return None
            
        objects = Objects()
        objects.object_list = [
            # Navigation gate buoys (Taylor Made Sur-Mark)
            type('Object', (), {
                'position': (-2, 1, 5),
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True
            }),
            type('Object', (), {
                'position': (2, 1, 5),
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True
            }),
            
            # Speed gate buoys (Polyform A-2)
            type('Object', (), {
                'position': (-4, 0.3, 20),
                'dimensions': (0.254, 0.3048, 0.254),
                'confidence': 85,
                'tracking_state': True
            }),
            type('Object', (), {
                'position': (4, 0.3, 20),
                'dimensions': (0.254, 0.3048, 0.254),
                'confidence': 85,
                'tracking_state': True
            }),
            
            # Yellow buoy (endangered species)
            type('Object', (), {
                'position': (0, 0.15, 40),
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 90,
                'tracking_state': True
            }),
            
            # Stationary vessel
            type('Object', (), {
                'position': (2, 0.5, 35),
                'dimensions': (1.0, 0.8, 2.0),
                'confidence': 85,
                'tracking_state': True
            })
        ]
        return objects
        
    def get_object_color_confidence(self, position: Tuple[float, float, float]) -> Dict[str, float]:
        """Return mock color confidence values."""
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
            
    def get_spatial_map(self) -> Optional[Mesh]:
        """Return mock spatial map."""
        if not self.mapping_enabled:
            return None
        return Mesh()
        
    def get_point_cloud(self) -> np.ndarray:
        """Return mock point cloud."""
        return np.zeros((1080, 1920, 4), dtype=np.float32)  # XYZRGBA format
        
    def get_depth_map(self) -> np.ndarray:
        """Return mock depth map."""
        return np.ones((1080, 1920), dtype=np.float32) * 2.0  # 2m depth
        
    def get_confidence_map(self) -> np.ndarray:
        """Return mock confidence map."""
        return np.ones((1080, 1920), dtype=np.uint8) * 100  # High confidence
        
    def close(self):
        """Close camera and disable all features."""
        self.is_open = False
        self.tracking_enabled = False
        self.mapping_enabled = False
        self.detection_enabled = False
