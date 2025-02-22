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
import os

# Only import pyzed if not using mock
if not os.getenv('MOCK_ZED'):
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
        
    def detect_buoys(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """ZED nesne tespiti kullanarak şamandıraları tespit eder.
        
        Args:
            frame: RGB görüntü (numpy dizisi)
            
        Returns:
            Dict[str, List[Dict]]: Renklere göre tespit edilen şamandıralar (konum ve boyutlarıyla)
        """
        objects = self.camera.get_objects()
        if objects is None:
            return {'red': [], 'green': [], 'yellow': [], 'black': [], 'blue': []}
            
        buoys = {
            'red': [], 'green': [], 'yellow': [], 'black': [], 'blue': []
        }
        
        for obj in objects.object_list:
            if obj.confidence < 50:  # Düşük güvenilirlikli tespitleri filtrele
                continue
                
            position = obj.position
            dimensions = obj.dimensions
            
            color = self._classify_buoy(dimensions, position)
            if color:
                distance = np.sqrt(position[0]**2 + position[2]**2)
                buoy_type = self._get_buoy_type(position)
                
                buoys[color].append({
                    'position': tuple(position),
                    'dimensions': tuple(dimensions),
                    'confidence': obj.confidence,
                    'distance': distance,  # Başlangıçtan mesafe
                    'type': buoy_type,  # Şamandıra tipi (navigation/path/speed)
                    'specs': self.buoy_specs[buoy_type]  # Şamandıra özellikleri
                })
                
        return buoys
        
    def _get_buoy_type(self, position: Tuple[float, float, float]) -> str:
        """Konuma göre şamandıra tipini belirler.
        
        Args:
            position: (x, y, z) metre cinsinden konum
            
        Returns:
            str: Şamandıra tipi ('navigation_gate', 'path_gate', 'speed_gate')
        """
        distance = np.sqrt(position[0]**2 + position[2]**2)
        
        if distance < 1.83:  # 6ft içinde
            return 'navigation_gate'
        elif distance < 30.48:  # 100ft içinde
            return 'speed_gate'
        else:
            return 'path_gate'
        
    def _classify_buoy(self, dimensions: Tuple[float, float, float], position: Tuple[float, float, float]) -> Optional[str]:
        """Şamandıra boyutları ve konumuna göre sınıflandırma yapar.
        
        Args:
            dimensions: (genişlik, yükseklik, derinlik) metre cinsinden
            position: (x, y, z) başlangıç noktasından metre cinsinden konum
            
        Returns:
            Optional[str]: Şamandıra renk sınıflandırması veya None
        """
        height = dimensions[1]  # Y yukarı yönde
        distance_from_start = np.sqrt(position[0]**2 + position[2]**2)  # X-Z düzleminde mesafe
        diameter = max(dimensions[0], dimensions[2])  # Genişlik/derinlikten maksimum
        
        # Mesafe ve boyutlara göre şamandıra tipini kontrol et
        if distance_from_start < 1.83:  # İlk geçit (6ft içinde)
            specs = self.buoy_specs['navigation_gate']
            if (0.8 * specs['height'] <= height <= 1.2 * specs['height'] and
                0.8 * specs['diameter'] <= diameter <= 1.2 * specs['diameter']):
                return self._classify_color(position)
                
        elif distance_from_start < 30.48:  # Hız parkuru (100ft içinde)
            specs = self.buoy_specs['speed_gate']
            if (0.8 * specs['height'] <= height <= 1.2 * specs['height']):
                return self._classify_color(position)
                
        else:  # Diğer geçitler
            specs = self.buoy_specs['path_gate']
            if (0.8 * specs['height'] <= height <= 1.2 * specs['height'] and
                0.8 * specs['diameter'] <= diameter <= 1.2 * specs['diameter']):
                return self._classify_color(position)
                
        return None  # Boyutlar uyuşmuyorsa şamandıra değil
        
    def _classify_color(self, position: Tuple[float, float, float]) -> str:
        """Şamandıra rengini konuma göre sınıflandırır.
        
        Args:
            position: (x, y, z) metre cinsinden konum
            
        Returns:
            str: Renk sınıflandırması ('red', 'green', 'yellow', 'black')
        """
        # ZED kamera renk tespiti ve konum bilgisini kullan
        color_confidence = self.camera.get_object_color_confidence(position)
        
        # İlk geçit için (6ft içinde)
        distance = np.sqrt(position[0]**2 + position[2]**2)
        if distance < 1.83:  # İlk geçit
            # Kameranın sol/sağına göre kırmızı/yeşil
            return 'red' if position[0] < 0 else 'green'
            
        # Hız parkuru için (100ft içinde)
        elif distance < 30.48:
            if color_confidence.get('black', 0) > 50:
                return 'black'
            elif color_confidence.get('blue', 0) > 50:
                return 'blue'
            # Kameranın sol/sağına göre kırmızı/yeşil
            return 'red' if position[0] < 0 else 'green'
            
        # Diğer geçitler için
        else:
            if color_confidence.get('yellow', 0) > 50:
                return 'yellow'
            # Kameranın sol/sağına göre kırmızı/yeşil
            return 'red' if position[0] < 0 else 'green'
            
        return None
