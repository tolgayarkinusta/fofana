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
        
    def open(self, init_params: Optional[InitParameters] = None) -> bool:
        """Open camera with optional initialization parameters."""
        if init_params:
            self.init_params = init_params
        self.is_open = True
        self.camera.open(self.init_params)
        return True
        
    def enable_positional_tracking(self, params: Optional[PositionalTrackingParameters] = None) -> bool:
        """Enable positional tracking with optional parameters."""
        self.tracking_enabled = True
        return True
        
    def enable_spatial_mapping(self, params: Optional[SpatialMappingParameters] = None) -> bool:
        """Enable spatial mapping with optional parameters."""
        self.mapping_enabled = True
        return True
        
    def enable_object_detection(self, params: Optional[ObjectDetectionParameters] = None) -> bool:
        """Enable object detection with optional parameters."""
        self.detection_enabled = True
        return True
        
    def get_frame(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Return mock frame data."""
        frame = torch.zeros((720, 1280, 3), dtype=torch.uint8)
        depth = torch.zeros((720, 1280), dtype=torch.float32)
        
        # Try to move to CUDA if available
        try:
            frame = frame.cuda()
            depth = depth.cuda()
        except RuntimeError:
            # CUDA not available, use CPU tensors
            pass
            
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
                'position': (-0.91, 1, 1.5),  # Within 6ft
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True,
                'type': 'navigation_gate',
                'distance': 1.5,
                'label': 'red_buoy'
            }),
            type('Object', (), {
                'position': (0.91, 1, 1.5),  # Within 6ft
                'dimensions': (0.4572, 0.9906, 0.4572),
                'confidence': 90,
                'tracking_state': True,
                'type': 'navigation_gate',
                'distance': 1.5,
                'label': 'green_buoy'
            }),
            
            # Speed gate buoys (Polyform A-2)
            type('Object', (), {
                'position': (-2, 0.3, 15),  # Between 6ft and 100ft
                'dimensions': (0.254, 0.3048, 0.254),
                'confidence': 85,
                'tracking_state': True,
                'type': 'speed_gate',
                'distance': 15.0,
                'label': 'red_buoy'
            }),
            type('Object', (), {
                'position': (2, 0.3, 15),
                'dimensions': (0.254, 0.3048, 0.254),
                'confidence': 85,
                'tracking_state': True,
                'type': 'speed_gate',
                'distance': 15.0,
                'label': 'green_buoy'
            }),
            type('Object', (), {
                'position': (0, 0.3, 15),
                'dimensions': (0.254, 0.3048, 0.254),
                'confidence': 85,
                'tracking_state': True,
                'type': 'speed_gate',
                'distance': 15.0,
                'label': 'black_buoy'
            }),
            
            # Path gate buoys (Polyform A-0)
            type('Object', (), {
                'position': (-3, 0.15, 35),  # Beyond 100ft
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 75,
                'tracking_state': True,
                'type': 'path_gate',
                'distance': 35.0,
                'label': 'red_buoy'
            }),
            type('Object', (), {
                'position': (3, 0.15, 35),
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 75,
                'tracking_state': True,
                'type': 'path_gate',
                'distance': 35.0,
                'label': 'green_buoy'
            }),
            type('Object', (), {
                'position': (0, 0.15, 35),
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 70,
                'tracking_state': True,
                'type': 'path_gate',
                'distance': 35.0,
                'label': 'yellow_buoy'
            }),
            
            # Yellow obstacle buoys (Polyform A-0)
            type('Object', (), {
                'position': (-1, 0.15, 25),
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 75,
                'tracking_state': True,
                'type': 'path_gate',
                'distance': 25.0,
                'label': 'yellow_buoy'
            }),
            type('Object', (), {
                'position': (1, 0.15, 25),
                'dimensions': (0.203, 0.1524, 0.203),
                'confidence': 75,
                'tracking_state': True,
                'type': 'path_gate',
                'distance': 25.0,
                'label': 'yellow_buoy'
            }),
            
            # Stationary vessel
            type('Object', (), {
                'position': (2, 0.5, 35),
                'dimensions': (1.0, 0.8, 2.0),
                'confidence': 85,
                'tracking_state': True,
                'type': 'vessel'
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
