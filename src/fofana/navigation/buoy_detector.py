"""
Şamandıra tespit ve takip modülü.

Özellikler:
- ZED2i kamera ile şamandıra tespiti
- Renkli şamandıraların tespiti (kırmızı, yeşil, sarı, siyah)
- 3B konum ve boyut bilgisi
- Şamandıra takibi ve sınıflandırma
- RoboBoat 2025 şamandıra özelliklerine göre filtreleme
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

class BuoyDetector:
    """Şamandıra tespit ve takip modülü.

    Özellikler:
    - ZED2i kamera ile şamandıra tespiti
    - Renkli şamandıraların tespiti (kırmızı, yeşil, sarı, siyah)
    - 3B konum ve boyut bilgisi
    - Şamandıra takibi ve sınıflandırma
    - RoboBoat 2025 şamandıra özelliklerine göre filtreleme
    """
    def __init__(self, camera):
        """Initialize buoy detector.
        Args:
            camera: ZEDCamera instance for object detection
        """
        self.camera = camera

        # Detection parameters
        self.depth_min = 0.3  # Minimum 30cm
        self.depth_max = 40.0  # Maximum 40m
        self.confidence_threshold = 50  # Minimum confidence score

        # Buoy specifications (in meters)
        self.buoy_specs = {
            'navigation_gate': {  # İlk geçit - Taylor Made Sur-Mark
                'height': 0.9906,  # 39 inches
                'diameter': 0.4572,  # 18 inches
                'model': {'red': '950410', 'green': '950400'}
            },
            'path_gate': {    # Diğer geçitler - Polyform A-0
                'height': 0.1524,  # 0.5 feet
                'diameter': 0.203,  # 20.3 cm
                'model': 'A-0'
            },
            'speed_gate': {   # Hız parkuru - Polyform A-2
                'height': 0.3048,  # 1 foot
                'diameter': 0.254,  # estimated from A-2 specs
                'model': 'A-2'
            }
        }

    def detect_buoys(self, _: np.ndarray) -> Dict[str, List[Dict]]:
        """ZED nesne tespiti kullanarak şamandıraları tespit eder.

        Args:
            _: RGB görüntü (numpy dizisi, kullanılmıyor)

        Returns:
            Dict[str, List[Dict]]: Renklere göre tespit edilen şamandıralar
        """
        objects = self.camera.get_objects()
        if objects is None:
            return {'red': [], 'green': [], 'yellow': [], 'black': [], 'blue': []}

        buoys = {
            'red': [], 'green': [], 'yellow': [], 'black': [], 'blue': []
        }

        # Track buoys by position and type to avoid duplicates
        detected_positions = {}  # type -> set of (x,z) positions

        for obj in objects.object_list:
            # Skip low confidence detections
            if obj.confidence < self.confidence_threshold:
                continue

            # Get object properties
            position = obj.position
            dimensions = obj.dimensions
            distance = np.sqrt(position[0]**2 + position[2]**2)

            # Get buoy type from object or infer from dimensions
            if hasattr(obj, 'type'):
                buoy_type = obj.type
            else:
                buoy_type = self._get_buoy_type(position, dimensions)

            # Skip if similar buoy of same type already detected
            pos_key = (round(position[0], 1), round(position[2], 1))
            if buoy_type not in detected_positions:
                detected_positions[buoy_type] = set()
            elif pos_key in detected_positions[buoy_type]:
                continue

            # Get color from label or classify based on position
            if hasattr(obj, 'label'):
                color = obj.label.split('_')[0]
            else:
                color = self._classify_color(position)

            if color in buoys:
                # Add buoy if dimensions match expected type
                expected_type = self._get_buoy_type(position, dimensions)
                if buoy_type == expected_type:
                    buoys[color].append({
                        'position': position,
                        'dimensions': dimensions,
                        'confidence': obj.confidence,
                        'distance': distance,
                        'type': buoy_type,
                        'specs': self.buoy_specs[buoy_type]
                    })
                    detected_positions[buoy_type].add(pos_key)

        return buoys

    def _get_buoy_type(
            self,
            position: Tuple[float, float, float],
            dimensions: Optional[Tuple[float, float, float]] = None
    ) -> str:
        """Konuma ve boyutlara göre şamandıra tipini belirler.

        Args:
            position: (x, y, z) metre cinsinden konum
            dimensions: Optional (genişlik, yükseklik, derinlik) metre cinsinden

        Returns:
            str: Şamandıra tipi ('navigation_gate', 'path_gate', 'speed_gate')
        """
        if dimensions:
            height = dimensions[1]
            diameter = max(dimensions[0], dimensions[2])

            # Navigation gate buoys (Taylor Made Sur-Mark)
            nav_specs = self.buoy_specs['navigation_gate']
            height_ok = 0.8 * nav_specs['height'] <= height <= 1.2 * nav_specs['height']
            diameter_ok = 0.8 * nav_specs['diameter'] <= diameter <= 1.2 * nav_specs['diameter']
            if height_ok and diameter_ok:
                return 'navigation_gate'

            # Speed gate buoys (Polyform A-2)
            speed_specs = self.buoy_specs['speed_gate']
            height_ok = 0.8 * speed_specs['height'] <= height <= 1.2 * speed_specs['height']
            diameter_ok = (0.8 * speed_specs['diameter'] <=
                         diameter <= 1.2 * speed_specs['diameter'])
            if height_ok and diameter_ok:
                return 'speed_gate'

            # Path gate buoys (Polyform A-0)
            path_specs = self.buoy_specs['path_gate']
            height_ok = 0.8 * path_specs['height'] <= height <= 1.2 * path_specs['height']
            diameter_ok = (0.8 * path_specs['diameter'] <=
                         diameter <= 1.2 * path_specs['diameter'])
            if height_ok and diameter_ok:
                return 'path_gate'

        # Fallback to position-based classification
        distance = np.sqrt(position[0]**2 + position[2]**2)

        if distance < 1.83:  # 6ft içinde
            return 'navigation_gate'
        if distance < 30.48:  # 100ft içinde
            return 'speed_gate'
        return 'path_gate'

    def detect_obstacles(self, _: np.ndarray) -> Dict[str, List[Dict]]:
        """Detect all obstacles including buoys and vessels.

        Args:
            _: RGB görüntü (numpy dizisi, kullanılmıyor)

        Returns:
            Dict[str, List[Dict]]: Tespit edilen engeller kategorilere göre
        """
        obstacles = {
            'yellow_buoys': [],
            'stationary_vessels': [],
            'other': []
        }

        # Get objects from camera
        objects = self.camera.get_objects()
        if not objects:
            return obstacles

        # Track yellow buoys by position to avoid duplicates
        detected_yellow_positions = set()

        # Process each detected object
        for obj in objects.object_list:
            if obj.confidence < self.confidence_threshold:
                continue

            position = obj.position
            dimensions = obj.dimensions
            distance = np.sqrt(position[0]**2 + position[2]**2)

            # Skip if similar yellow buoy already detected
            pos_key = (round(position[0], 1), round(position[2], 1))

            # Check for yellow buoys first
            is_yellow = False
            if hasattr(obj, 'label') and obj.label == 'yellow_buoy':
                is_yellow = True
            else:
                # Check color confidence for yellow
                color_confidence = self.camera.get_object_color_confidence(position)
                if color_confidence.get('yellow', 0) > self.confidence_threshold:
                    is_yellow = True

            # Add yellow buoy if dimensions match Polyform A-0 and not duplicate
            if is_yellow and pos_key not in detected_yellow_positions:
                path_specs = self.buoy_specs['path_gate']
                height_ok = (0.8 * path_specs['height'] <=
                          dimensions[1] <=
                          1.2 * path_specs['height'])
                diameter_ok = (0.8 * path_specs['diameter'] <=
                            max(dimensions[0], dimensions[2]) <=
                            1.2 * path_specs['diameter'])
                if height_ok and diameter_ok:
                    obstacles['yellow_buoys'].append({
                        'position': position,
                        'dimensions': (0.203, 0.1524, 0.203),  # Polyform A-0 dimensions
                        'confidence': obj.confidence,
                        'distance': distance
                    })
                    detected_yellow_positions.add(pos_key)
                    continue

            # Check for vessels
            if hasattr(obj, 'type') and obj.type == 'vessel':
                obstacles['stationary_vessels'].append({
                    'position': position,
                    'dimensions': dimensions,
                    'confidence': obj.confidence,
                    'distance': distance
                })
                continue

            # Add other obstacles
            obstacles['other'].append({
                'position': position,
                'dimensions': dimensions,
                'confidence': obj.confidence,
                'distance': distance
            })

        return obstacles

    def _classify_color(self, position: Tuple[float, float, float]) -> Optional[str]:
        """Konuma göre renk sınıflandırması yapar.

        Args:
            position: (x, y, z) metre cinsinden konum

        Returns:
            Optional[str]: Renk sınıflandırması veya None
        """
        color_confidence = self.camera.get_object_color_confidence(position)
        if not color_confidence:
            return None

        # En yüksek güvenilirlikli rengi seç
        max_confidence = 0
        max_color = None

        for color, confidence in color_confidence.items():
            if confidence > max_confidence:
                max_confidence = confidence
                max_color = color

        return max_color if max_confidence > self.confidence_threshold else None
