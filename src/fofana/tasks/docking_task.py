"""
Yanaşma görevi modülü.

Bu modül şu işlevleri içerir:
- Doğru renk/şekildeki yanaşma yerini tespit etme
- Güvenli yanaşma kontrolü
- Dolu yanaşma yerlerinden kaçınma
"""
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..core.mavlink_controller import USVController
from ..vision.camera import ZEDCamera
from ..vision.shape_detector import ShapeDetector

@dataclass
class DockingConfig:
    """Yanaşma görevi yapılandırması."""
    shape: str  # Hedef şekil: 'circle', 'triangle', 'square', 'plus'
    color: str  # Hedef renk: 'red', 'green', 'blue'
    approach: str  # Yanaşma yönü: 'forward', 'backward'
    
    def __post_init__(self):
        """Yapılandırmayı doğrula."""
        valid_shapes = {'circle', 'triangle', 'square', 'plus'}
        valid_colors = {'red', 'green', 'blue'}
        valid_approaches = {'forward', 'backward'}
        
        if self.shape not in valid_shapes:
            raise ValueError(f"Geçersiz şekil: {self.shape}. Şunlardan biri olmalı: {valid_shapes}")
        if self.color not in valid_colors:
            raise ValueError(f"Geçersiz renk: {self.color}. Şunlardan biri olmalı: {valid_colors}")
        if self.approach not in valid_approaches:
            raise ValueError(f"Geçersiz yaklaşma yönü: {self.approach}. Şunlardan biri olmalı: {valid_approaches}")

class DockingTask:
    def __init__(self, control_queue: mp.Queue, status_queue: mp.Queue):
        """Yanaşma görevini başlat.
        
        Args:
            control_queue: Görev kontrol komutları için kuyruk
            status_queue: Durum güncellemeleri için kuyruk
        """
        self.control_queue = control_queue
        self.status_queue = status_queue
        self.usv = USVController()
        self.camera = ZEDCamera()
        self.shape_detector = ShapeDetector()
        self.running = False
        self.config = None  # DockingConfig instance
        self.start_time = None
        
    def configure(self, shape: str, color: str, approach: str = 'forward') -> bool:
        """Hedef yanaşma yerini yapılandır.
        
        Args:
            shape: Hedef şekil ('circle', 'triangle', 'square', 'plus')
            color: Hedef renk ('red', 'green', 'blue')
            approach: Yanaşma yönü ('forward', 'backward')
            
        Returns:
            bool: Yapılandırma başarılı ise True
        """
        try:
            self.config = DockingConfig(shape, color, approach)
            return True
        except ValueError as e:
            self._send_status_update(f"Yapılandırma hatası: {str(e)}")
            return False
            
    def run(self) -> None:
        """Yanaşma görevini çalıştır."""
        if not self.config:
            self._send_status_update("Görev yapılandırılmamış")
            return
            
        self.running = True
        self.start_time = time.time()
        
        try:
            # Donanımı başlat
            if not self.camera.open():
                self._send_status_update("Kamera başlatılamadı")
                return
                
            self.camera.enable_positional_tracking()
            self.camera.enable_object_detection()
            self.usv.arm_vehicle()
            
            while self.running:
                # Kontrol kuyruğunu kontrol et
                if not self.control_queue.empty():
                    cmd = self.control_queue.get()
                    if cmd == "stop":
                        break
                
                # Kamera görüntüsünü al
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                # Hedef yanaşma yerini bul
                dock_info = self._find_target_dock(frame)
                if dock_info:
                    # Yanaşma yerine git
                    if self._navigate_to_dock(frame, dock_info):
                        elapsed_time = time.time() - self.start_time
                        self._send_status_update(
                            f"Yanaşma başarılı! Süre: {elapsed_time:.1f}s"
                        )
                        break
                
        finally:
            self.cleanup()
            
    def _find_target_dock(self, frame) -> Optional[Dict]:
        """Hedef yanaşma yerini bul.
        
        Args:
            frame: Kamera görüntüsü
            
        Returns:
            Optional[Dict]: Yanaşma yeri bilgileri (bulunursa)
        """
        # Şekilleri tespit et
        shapes = self.shape_detector.detect_shapes(frame)
        if not shapes:
            return None
            
        # Hedef şekil ve rengi ara
        for shape in shapes:
            if (shape['type'] == self.config.shape and 
                shape['color'] == self.config.color):
                return {
                    'position': shape['position'],
                    'distance': shape['distance'],
                    'confidence': shape['confidence'],
                    'dimensions': shape['dimensions']
                }
                
        return None
        
    def _navigate_to_dock(self, frame, dock_info: Dict) -> bool:
        """Yanaşma yerine git.
        
        Args:
            frame: Kamera görüntüsü
            dock_info: Yanaşma yeri bilgileri
            
        Returns:
            bool: Yanaşma başarılı ise True
        """
        # Yanaşma yeri konumunu al
        dock_x, dock_y = dock_info['position']
        frame_width = frame.shape[1]
        
        # Hizalama hatasını hesapla
        center_error = (dock_x - frame_width/2) / (frame_width/2)  # [-1, 1]
        
        # Temel oransal kontrol
        base_speed = 30  # Temel ileri hız
        turn_scale = 15  # Maksimum dönüş ayarı
        
        # Motor hızlarını hesapla
        if dock_info['distance'] < 0.5:  # Yanaşma mesafesine gelindi
            left_speed = right_speed = 0
            return True
        else:
            # İleri yanaşma
            if self.config.approach == 'forward':
                left_speed = base_speed - center_error * turn_scale
                right_speed = base_speed + center_error * turn_scale
            # Geri yanaşma
            else:
                left_speed = -(base_speed + center_error * turn_scale)
                right_speed = -(base_speed - center_error * turn_scale)
                
        # Motor hızlarını ayarla
        self.usv.set_motor_speed(5, left_speed)   # Sol motor
        self.usv.set_motor_speed(6, right_speed)  # Sağ motor
        
        return False
        
    def _send_status_update(self, message: str) -> None:
        """Durum güncellemesi gönder."""
        self.status_queue.put({
            "status": message,
            "config": self.config.__dict__ if self.config else None,
            "running": self.running,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        })
        
    def cleanup(self) -> None:
        """Kaynakları temizle."""
        self.usv.stop_motors()
        self.usv.disarm_vehicle()
        self.camera.close()
        self.running = False
        self.start_time = None
