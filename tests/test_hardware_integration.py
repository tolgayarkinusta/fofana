"""RoboBoat 2025 donanım entegrasyon testleri."""
import pytest
import time
import sys
import os
from unittest.mock import patch, MagicMock

# Add mock modules to path
sys.path.insert(0, os.path.dirname(__file__))
import mock_zed_sdk
import mock_serial

sys.modules['pyzed.sl'] = mock_zed_sdk
sys.modules['serial'] = mock_serial

from fofana.main import RoboBoat2025Runner
from fofana.core.mavlink_controller import USVController
from fofana.vision.camera import ZEDCamera

def test_zed_camera_cuda():
    """ZED2i kamera CUDA entegrasyon testi."""
    camera = ZEDCamera()
    assert camera.open() == True
    
    # CUDA kontrolü
    frame, depth = camera.get_frame()
    assert frame is not None
    assert frame.is_cuda  # CUDA tensor kontrolü
    assert depth.is_cuda  # Derinlik verisi CUDA kontrolü
    
    # Performans testi
    start_time = time.time()
    for _ in range(100):
        frame, depth = camera.get_frame()
    fps = 100 / (time.time() - start_time)
    assert fps >= 30  # En az 30 FPS bekleniyor
    
    camera.close()

def test_motor_pwm_control():
    """Motor PWM kontrol testi."""
    with patch('fofana.core.mavlink_controller.mavutil.mavlink_connection') as mock_mavlink:
        controller = USVController()
        controller.arm_vehicle()
        
        # İleri hareket testi
        controller.set_motor_speed('left', 1600)  # %20 ileri
        controller.set_motor_speed('right', 1600)
        
        # Geri hareket testi
        controller.set_motor_speed('left', 1400)  # %20 geri
        controller.set_motor_speed('right', 1400)
        
        # Sağa dönüş testi
        controller.set_motor_speed('left', 1600)
        controller.set_motor_speed('right', 1400)
        
        # Sola dönüş testi
        controller.set_motor_speed('left', 1400)
        controller.set_motor_speed('right', 1600)
        
        controller.stop_motors()
        controller.disarm_vehicle()
        
        # Verify commands were sent
        assert mock_mavlink.called

def test_full_task_sequence():
    """Tam görev sırası testi."""
    runner = RoboBoat2025Runner()
    
    # Tüm görevleri çalıştır
    runner.run_all_tasks()
    
    # Her görevin tamamlandığını kontrol et
    assert runner._run_navigation_task() == True
    assert runner._run_mapping_task() == True
    assert runner._run_docking_task() == True
    assert runner._run_rescue_task() == True
    assert runner._run_return_home_task() == True
