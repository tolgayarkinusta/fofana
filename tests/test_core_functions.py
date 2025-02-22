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
    assert camera.enable_positional_tracking() == True
    assert camera.enable_spatial_mapping() == True
    
    # Görüntü alma testi
    frame, depth, pose = camera.get_frame()
    assert frame is not None, "Frame alınamadı"
    assert depth is not None, "Depth alınamadı"
    assert pose is not None, "Pose alınamadı"
    
    # CUDA kontrolü
    assert frame.is_cuda, "Frame CUDA'da değil"
    assert depth.is_cuda, "Depth CUDA'da değil"
    
    # Nokta bulutu ve derinlik testi
    point_cloud = camera.get_point_cloud()
    assert point_cloud is not None, "Point cloud alınamadı"
    assert point_cloud.shape[-1] == 4, "Point cloud XYZRGBA formatında değil"
    
    # Spatial mapping testi
    mesh = camera.get_spatial_map()
    assert mesh is not None, "Mesh alınamadı"
    assert hasattr(mesh, 'vertices'), "Mesh vertices içermiyor"
    
    camera.close()

def test_buoy_detection():
    """Şamandıra tespit testi."""
    camera = ZEDCamera()
    detector = BuoyDetector(camera)
    
    camera.open()
    camera.enable_object_detection()
    frame, _, _ = camera.get_frame()
    
    # Şamandıra tespiti
    buoys = detector.detect_buoys(frame)
    
    # Kırmızı-yeşil şamandıra kontrolü (Taylor Made Sur-Mark)
    assert len(buoys['red']) > 0, "Kırmızı şamandıra tespit edilemedi"
    assert len(buoys['green']) > 0, "Yeşil şamandıra tespit edilemedi"
    
    # İlk geçit şamandıra boyutları kontrolü
    nav_buoy = buoys['red'][0]
    assert nav_buoy['type'] == 'navigation_gate', "Yanlış şamandıra tipi"
    assert abs(nav_buoy['dimensions'][1] - 0.9906) < 0.1, "Yanlış şamandıra yüksekliği"
    assert abs(nav_buoy['dimensions'][0] - 0.4572) < 0.1, "Yanlış şamandıra çapı"
    
    # Sarı şamandıra kontrolü (Polyform A-0)
    assert len(buoys['yellow']) > 0, "Sarı şamandıra tespit edilemedi"
    yellow_buoy = buoys['yellow'][0]
    assert yellow_buoy['type'] == 'path_gate', "Yanlış sarı şamandıra tipi"
    assert abs(yellow_buoy['dimensions'][1] - 0.1524) < 0.1, "Yanlış sarı şamandıra yüksekliği"
    
    camera.close()

def test_navigation():
    """Navigasyon testi."""
    camera = ZEDCamera()
    controller = USVController()
    planner = PathPlanner(controller, camera)
    
    # Kamera ve tespit sistemini başlat
    camera.open()
    camera.enable_positional_tracking()
    camera.enable_object_detection()
    
    # Test frame al
    frame, depth, pose = camera.get_frame()
    
    # Costmap güncelleme testi
    planner.update_costmap()
    assert planner.costmap is not None, "Costmap oluşturulamadı"
    
    # Engel tespiti testi
    obstacles = planner.detect_obstacles()
    assert len(obstacles['yellow_buoys']) > 0, "Sarı şamandıralar tespit edilemedi"
    assert len(obstacles['stationary_vessels']) > 0, "Sabit tekneler tespit edilemedi"
    
    # Güvenlik mesafesi kontrolü
    yellow_pos = obstacles['yellow_buoys'][0]['position']
    x, y = planner._world_to_costmap(yellow_pos)
    assert planner.costmap[y, x] >= 0.8, "Sarı şamandıra güvenlik mesafesi yetersiz"
    
    camera.close()

def test_task_management():
    """Görev yönetimi testi."""
    camera = ZEDCamera()
    controller = USVController()
    manager = TaskManager(camera=camera, controller=controller)
    
    # Görev başlatma testi
    assert manager.start_task('navigation'), "Navigasyon görevi başlatılamadı"
    
    # Durum kontrolü
    state = manager.get_task_state('navigation')
    assert state['running'], "Görev çalışmıyor"
    assert state['progress'] >= 0, "Görev ilerlemesi hatalı"
    
    # Çoklu görev testi
    assert manager.start_task('mapping'), "Haritalama görevi başlatılamadı"
    states = manager.get_all_task_states()
    assert len(states) >= 2, "Çoklu görev çalışmıyor"
    
    # Görev durdurma testi
    manager.stop_all_tasks()
    states = manager.get_all_task_states()
    assert not any(s['running'] for s in states.values()), "Görevler durdurulamadı"
