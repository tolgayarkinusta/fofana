"""Test complete system integration for RoboBoat 2025."""
import pytest
import time
import numpy as np
from fofana.vision.camera import ZEDCamera
from fofana.navigation.buoy_detector import BuoyDetector
from fofana.navigation.path_planner import PathPlanner
from fofana.core.mavlink_controller import USVController
from fofana.tasks.task_manager import TaskManager

def test_system_initialization():
    """Test complete system initialization."""
    # Initialize camera
    camera = ZEDCamera()
    assert camera.open(), "Failed to open camera"
    assert camera.enable_positional_tracking(), "Failed to enable tracking"
    assert camera.enable_spatial_mapping(), "Failed to enable mapping"
    assert camera.enable_object_detection(), "Failed to enable detection"
    
    # Initialize USV controller
    usv = USVController()
    usv.arm_vehicle()
    
    # Initialize buoy detector
    detector = BuoyDetector(camera)
    
    # Initialize path planner
    planner = PathPlanner(usv, camera)
    
    # Verify camera performance
    frames = 0
    start_time = time.time()
    while frames < 30:  # Test 30 frames
        frame, depth, pose = camera.get_frame()
        if frame is not None:
            frames += 1
            assert frame.is_cuda, "Frame not on GPU"
            assert depth.is_cuda, "Depth not on GPU"
            
    duration = time.time() - start_time
    fps = frames / duration
    assert fps >= 30.0, f"Low frame rate: {fps:.1f} FPS"
    
    # Test buoy detection
    frame, _, _ = camera.get_frame()
    buoys = detector.detect_buoys(frame)
    assert isinstance(buoys, dict), "Invalid buoy detection result"
    assert all(k in buoys for k in ['red', 'green', 'yellow', 'black']), "Missing buoy types"
    
    # Test path planning
    planner.update_costmap()
    assert planner.costmap is not None, "Failed to generate costmap"
    
    # Cleanup
    camera.close()
    usv.disarm_vehicle()

def test_task_execution():
    """Test task execution and multiprocessing."""
    task_manager = TaskManager()
    
    # Test navigation task
    assert task_manager.start_task('navigation'), "Failed to start navigation"
    time.sleep(1)  # Wait for task to initialize
    
    status = task_manager.get_task_state('navigation')
    assert status['running'], "Navigation task not running"
    
    task_manager.stop_task('navigation')
    
    # Test mapping task
    assert task_manager.start_task('mapping'), "Failed to start mapping"
    time.sleep(1)
    
    status = task_manager.get_task_state('mapping')
    assert status['running'], "Mapping task not running"
    
    task_manager.stop_task('mapping')
    
    # Test rescue task
    params = {'targets': {'orange': 3, 'black': 3}}
    assert task_manager.start_task('rescue', params), "Failed to start rescue"
    time.sleep(1)
    
    status = task_manager.get_task_state('rescue')
    assert status['running'], "Rescue task not running"
    
    task_manager.stop_task('rescue')
    
    # Cleanup
    task_manager.stop_all_tasks()

def test_buoy_specifications():
    """Test buoy detection with competition specifications."""
    camera = ZEDCamera()
    detector = BuoyDetector(camera)
    
    camera.open()
    camera.enable_object_detection()
    
    # Test navigation channel buoys (39in height, 18in diameter)
    frame, _, _ = camera.get_frame()
    buoys = detector.detect_buoys(frame)
    
    for buoy in buoys['red'] + buoys['green']:
        height = buoy['dimensions'][1]  # Y up
        diameter = max(buoy['dimensions'][0], buoy['dimensions'][2])
        
        # Check dimensions (with 20% tolerance)
        assert 0.8 * 0.9906 <= height <= 1.2 * 0.9906, f"Invalid height: {height}m"
        assert 0.8 * 0.4572 <= diameter <= 1.2 * 0.4572, f"Invalid diameter: {diameter}m"
    
    # Test mapping buoys (0.5ft height, 20.3cm diameter)
    for buoy in buoys['yellow']:
        height = buoy['dimensions'][1]
        diameter = max(buoy['dimensions'][0], buoy['dimensions'][2])
        
        assert 0.8 * 0.1524 <= height <= 1.2 * 0.1524, f"Invalid height: {height}m"
        assert 0.8 * 0.203 <= diameter <= 1.2 * 0.203, f"Invalid diameter: {diameter}m"
    
    camera.close()
