"""Test complete system integration for RoboBoat 2025."""
import pytest
import time
import numpy as np
import sys
import os

# Use mock camera for testing
from fofana.vision.types.mock_sl import MockSL as sl
from fofana.vision.mock_camera import MockZEDCamera
from fofana.navigation.buoy_detector import BuoyDetector
from fofana.navigation.path_planner import (
    PathPlanner, FIRST_GATE_SPACING_MAX, GATE_WIDTH_MIN, GATE_WIDTH_MAX
)
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

def test_obstacle_avoidance():
    """Test obstacle avoidance with yellow buoys and vessels."""
    camera = MockZEDCamera()
    detector = BuoyDetector(camera)
    planner = PathPlanner(USVController(), camera)
    
    # Test yellow buoy detection
    frame = camera.get_frame()[0]  # Get RGB frame
    obstacles = detector.detect_obstacles(frame)
    assert len(obstacles['yellow_buoys']) > 0, "No yellow buoys detected"
    
    # Verify costmap generation
    planner.update_costmap()
    assert planner.costmap is not None, "Failed to generate costmap"
    
    # Check safety margins
    yellow_pos = obstacles['yellow_buoys'][0]['position']
    x, y = planner._world_to_costmap(yellow_pos)
    assert planner.costmap[y, x] >= 0.8, "Insufficient safety margin for endangered species"
    
    # Test vessel detection
    assert len(obstacles['stationary_vessels']) > 0, "No vessels detected"
    vessel_pos = obstacles['stationary_vessels'][0]['position']
    x, y = planner._world_to_costmap(vessel_pos)
    assert planner.costmap[y, x] >= 0.9, "Insufficient safety margin for vessels"
    
    # Cleanup
    camera.close()

def test_buoy_specifications():
    """Test buoy detection with competition specifications."""
    camera = ZEDCamera()
    detector = BuoyDetector(camera)
    
    camera.open()
    camera.enable_object_detection()
    
    # Get test frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Mock frame
    buoys = detector.detect_buoys(frame)
    
    # Test navigation gate buoys (Taylor Made Sur-Mark)
    nav_buoys = [b for b in buoys['red'] + buoys['green'] 
                 if b['type'] == 'navigation_gate']
    assert len(nav_buoys) == 2, "Should detect 2 navigation gate buoys"
    
    for buoy in nav_buoys:
        height = buoy['dimensions'][1]  # Y up
        diameter = max(buoy['dimensions'][0], buoy['dimensions'][2])
        distance = buoy['distance']
        
        # Verify dimensions (20% tolerance)
        assert 0.8 * 0.9906 <= height <= 1.2 * 0.9906, \
            f"Invalid navigation buoy height: {height}m"
        assert 0.8 * 0.4572 <= diameter <= 1.2 * 0.4572, \
            f"Invalid navigation buoy diameter: {diameter}m"
        assert distance < FIRST_GATE_SPACING_MAX, \
            f"Navigation buoy too far: {distance}m"
            
    # Test speed gate buoys (Polyform A-2)
    speed_buoys = [b for b in buoys['red'] + buoys['green'] + buoys['black']
                   if b['type'] == 'speed_gate']
    assert len(speed_buoys) == 3, "Should detect 3 speed gate buoys"
    
    for buoy in speed_buoys:
        height = buoy['dimensions'][1]
        diameter = max(buoy['dimensions'][0], buoy['dimensions'][2])
        distance = buoy['distance']
        
        assert 0.8 * 0.3048 <= height <= 1.2 * 0.3048, \
            f"Invalid speed buoy height: {height}m"
        assert 0.8 * 0.254 <= diameter <= 1.2 * 0.254, \
            f"Invalid speed buoy diameter: {diameter}m"
        assert 1.83 <= distance <= 30.48, \
            f"Speed buoy at wrong distance: {distance}m"
            
    # Test path gate buoys (Polyform A-0)
    path_buoys = [b for b in buoys['red'] + buoys['green'] + buoys['yellow']
                  if b['type'] == 'path_gate']
    assert len(path_buoys) == 3, "Should detect 3 path gate buoys"
    
    for buoy in path_buoys:
        height = buoy['dimensions'][1]
        diameter = max(buoy['dimensions'][0], buoy['dimensions'][2])
        distance = buoy['distance']
        
        assert 0.8 * 0.1524 <= height <= 1.2 * 0.1524, \
            f"Invalid path buoy height: {height}m"
        assert 0.8 * 0.203 <= diameter <= 1.2 * 0.203, \
            f"Invalid path buoy diameter: {diameter}m"
        assert distance > 30.48, \
            f"Path buoy too close: {distance}m"
            
    camera.close()
