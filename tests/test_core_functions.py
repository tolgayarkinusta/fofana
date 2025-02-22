"""RoboBoat 2025 temel fonksiyon testleri."""
import pytest
import numpy as np
from fofana.core.mavlink_controller import USVController
from fofana.vision.camera import ZEDCamera
from fofana.navigation.buoy_detector import BuoyDetector
from fofana.navigation.path_planner import PathPlanner
from fofana.tasks.task_manager import TaskManager

def test_motor_control():
    """Motor kontrol testi."""
    controller = USVController()
    
    # Arm testi
    controller.arm_vehicle()
    
    # İleri hareket testi
    controller.set_motor_speed('left', 1600)  # %20 ileri
    controller.set_motor_speed('right', 1600)
    
    # Geri hareket testi
    controller.set_motor_speed('left', 1400)  # %20 geri
    controller.set_motor_speed('right', 1400)
    
    # Dönüş testi
    controller.set_motor_speed('left', 1600)  # Sağa dönüş
    controller.set_motor_speed('right', 1400)
    
    # Stop testi
    controller.stop_motors()
    
    # Disarm testi
    controller.disarm_vehicle()

def test_camera_integration():
    """ZED2i kamera entegrasyon testi."""
    camera = ZEDCamera()
    
    # Bağlantı testi
    assert camera.open() == True
    
    # Görüntü alma testi
    frame, depth = camera.get_frame()
    assert frame is not None
    assert depth is not None
    
    # CUDA kontrolü
    assert frame.device.type == 'cuda'
    
    # Nokta bulutu testi
    point_cloud = camera.get_point_cloud()
    assert point_cloud is not None
    
    camera.close()

def test_buoy_detection():
    """Şamandıra tespit testi."""
    camera = ZEDCamera()
    detector = BuoyDetector()
    
    camera.open()
    frame, _ = camera.get_frame()
    
    # Şamandıra tespiti
    buoys = detector.detect_buoys(frame)
    
    # Kırmızı-yeşil şamandıra kontrolü
    assert len(buoys['red']) > 0
    assert len(buoys['green']) > 0
    
    # Sarı şamandıra kontrolü
    assert len(buoys['yellow']) > 0
    
    camera.close()

def test_navigation():
    """Navigasyon testi."""
    controller = USVController()
    planner = PathPlanner(controller)
    
    # Test koordinatları
    test_red_buoys = [(100, 300, 20), (500, 300, 20)]
    test_green_buoys = [(100, 500, 20), (500, 500, 20)]
    
    # Şamandıralar arası geçiş testi
    planner.navigate_through_gates(test_red_buoys, test_green_buoys)
    
    # Motor hızları kontrolü
    left_speed, right_speed = controller.get_motor_speeds()
    assert -100 <= left_speed <= 100
    assert -100 <= right_speed <= 100

def test_task_management():
    """Görev yönetimi testi."""
    manager = TaskManager()
    
    # Görev başlatma testi
    assert manager.start_task('navigation') == True
    
    # Durum kontrolü
    state = manager.get_task_state()
    assert state['task'] == 'navigation'
    assert state['state'] == 'running'
    
    # Görev durdurma testi
    manager.stop_task()
    state = manager.get_task_state()
    assert state['state'] == 'idle'
