"""
Şamandıra tespit ve takip modülü.

Özellikler:
- ZED2i kamera ile şamandıra tespiti
- Renkli şamandıraların tespiti (kırmızı, yeşil, sarı, siyah)
- 3B konum ve boyut bilgisi
- Şamandıra takibi ve sınıflandırma
- RoboBoat 2025 şamandıra özelliklerine göre filtreleme
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import pyzed.sl as sl

class BuoyDetector:
    def __init__(self, camera):
        """Initialize buoy detector.
        
        Args:
            camera: ZEDCamera instance for object detection
        """
        self.camera = camera
        
        # Buoy specifications (in meters)
        self.buoy_specs = {
            'navigation': {  # Taylor Made Sur-Mark
                'height': 0.9906,  # 39 inches
                'diameter': 0.4572  # 18 inches
            },
            'mapping': {    # Polyform A-0
                'height': 0.1524,  # 0.5 feet
                'diameter': 0.203   # 20.3 cm
            }
        }
        
    def detect_buoys(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """Detect buoys using ZED object detection.
        
        Args:
            frame: RGB image as numpy array (for visualization only)
            
        Returns:
            Dict[str, List[Dict]]: Detected buoys by color with position and dimensions
                {
                    'red': [{'position': (x,y,z), 'dimensions': (w,h,d), 'confidence': float}],
                    'green': [...],
                    'yellow': [...],
                    'black': [...]
                }
        """
        objects = self.camera.get_objects()
        if objects is None:
            return {'red': [], 'green': [], 'yellow': [], 'black': []}
            
        buoys = {
            'red': [],
            'green': [],
            'yellow': [],
            'black': []
        }
        
        for obj in objects.object_list:
            if obj.confidence < 50:  # Filter low confidence detections
                continue
                
            # Get 3D position and dimensions
            position = obj.position
            dimensions = obj.dimensions
            
            # Classify buoy by size and add to appropriate list
            buoy_type = self._classify_buoy(dimensions)
            if buoy_type:
                buoys[buoy_type].append({
                    'position': (position[0], position[1], position[2]),
                    'dimensions': (dimensions[0], dimensions[1], dimensions[2]),
                    'confidence': obj.confidence
                })
                
        return buoys
        
    def _classify_buoy(self, dimensions: Tuple[float, float, float]) -> Optional[str]:
        """Classify buoy based on dimensions.
        
        Args:
            dimensions: (width, height, depth) in meters
            
        Returns:
            Optional[str]: Buoy color classification or None if not a buoy
        """
        height = dimensions[1]  # Assuming Y is up
        diameter = max(dimensions[0], dimensions[2])  # Max of width/depth
        
        # Check navigation channel buoys (Taylor Made Sur-Mark)
        nav_spec = self.buoy_specs['navigation']
        if (0.8 * nav_spec['height'] <= height <= 1.2 * nav_spec['height'] and
            0.8 * nav_spec['diameter'] <= diameter <= 1.2 * nav_spec['diameter']):
            # Classify as red/green based on position (left/right of camera)
            return 'red' if dimensions[0] < 0 else 'green'
            
        # Check mapping buoys (Polyform A-0)
        map_spec = self.buoy_specs['mapping']
        if (0.8 * map_spec['height'] <= height <= 1.2 * map_spec['height'] and
            0.8 * map_spec['diameter'] <= diameter <= 1.2 * map_spec['diameter']):
            # Use object color for yellow/black classification
            # This requires custom training data to distinguish colors
            return 'yellow'  # For now, assume all small buoys are yellow
            
        return None
