"""Mock ZED camera implementation for testing."""
import numpy as np
from typing import Dict, Optional, Tuple

class MockZEDCamera:
    def __init__(self):
        self.is_open = False
        self.tracking_enabled = False
        self.mapping_enabled = False
        self.detection_enabled = False
        
    def open(self) -> bool:
        self.is_open = True
        return True
        
    def enable_positional_tracking(self) -> bool:
        self.tracking_enabled = True
        return True
        
    def enable_spatial_mapping(self) -> bool:
        self.mapping_enabled = True
        return True
        
    def enable_object_detection(self) -> bool:
        self.detection_enabled = True
        return True
        
    def get_frame(self):
        """Return mock frame data."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((1080, 1920), dtype=np.float32)
        pose = {'position': [0, 0, 0], 'rotation': [0, 0, 0]}
        return frame, depth, pose
        
    def get_objects(self):
        """Return mock detected objects."""
        if not self.detection_enabled:
            return None
            
        class Object:
            def __init__(self, pos, dim, conf):
                self.position = pos
                self.dimensions = dim
                self.confidence = conf
                
        class Objects:
            def __init__(self, objs):
                self.object_list = objs
                
        # Simulate different buoy types
        objects = [
            # Navigation gate buoys (Taylor Made)
            Object((-2, 1, 5), (0.4, 0.99, 0.4), 90),  # Red buoy
            Object((2, 1, 5), (0.4, 0.99, 0.4), 90),   # Green buoy
            
            # Speed gate buoys (Polyform A-2)
            Object((-4, 0.3, 20), (0.25, 0.3, 0.25), 85),  # Red buoy
            Object((4, 0.3, 20), (0.25, 0.3, 0.25), 85),   # Green buoy
            Object((0, 0.3, 25), (0.25, 0.3, 0.25), 80),   # Black buoy
            
            # Path gate buoys (Polyform A-0)
            Object((-3, 0.15, 35), (0.2, 0.15, 0.2), 75),  # Red buoy
            Object((3, 0.15, 35), (0.2, 0.15, 0.2), 75),   # Green buoy
            Object((0, 0.15, 40), (0.2, 0.15, 0.2), 70)    # Yellow buoy
        ]
        return Objects(objects)
        
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
            
    def close(self):
        self.is_open = False
        self.tracking_enabled = False
        self.mapping_enabled = False
        self.detection_enabled = False
