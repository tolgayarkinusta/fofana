"""Test system integration."""
import numpy as np
import pytest
from fofana.vision.mock_camera import MockZEDCamera
from fofana.navigation.buoy_detector import BuoyDetector
from fofana.navigation.path_planner import PathPlanner
from fofana.core.mavlink_controller import USVController

# Constants
FIRST_GATE_SPACING_MAX = 1.83  # 6ft in meters
SPEED_GATE_MIN = 1.83  # 6ft in meters
SPEED_GATE_MAX = 30.48  # 100ft in meters
PATH_GATE_MIN = 30.48  # 100ft in meters

def test_system_initialization():
    """Test basic system initialization."""
    camera = MockZEDCamera()
    detector = BuoyDetector(camera)
    planner = PathPlanner(USVController(), camera)
    
    assert camera is not None
    assert detector is not None
    assert planner is not None
    
def test_task_execution():
    """Test task execution and state management."""
    camera = MockZEDCamera()
    detector = BuoyDetector(camera)
    planner = PathPlanner(USVController(), camera)
    
    # Initialize camera
    camera.open()
    camera.enable_object_detection()
    
    # Get test frame
    frame = camera.get_frame()[0]  # Get RGB frame
    
    # Test buoy detection
    buoys = detector.detect_buoys(frame)
    assert len(buoys) > 0, "No buoys detected"
    
    # Test path planning
    path = planner.plan_path(buoys)
    assert path is not None, "No path generated"
    
def test_obstacle_avoidance():
    """Test obstacle avoidance with yellow buoys and vessels."""
    camera = MockZEDCamera()
    detector = BuoyDetector(camera)
    planner = PathPlanner(USVController(), camera)
    
    # Initialize camera and enable detection
    camera.open()
    camera.enable_object_detection()
    
    # Test yellow buoy detection
    frame = camera.get_frame()[0]  # Get RGB frame
    obstacles = detector.detect_obstacles(frame)
    assert len(obstacles['yellow_buoys']) > 0, "No yellow buoys detected"
    
    # Test vessel detection
    assert len(obstacles['stationary_vessels']) > 0, "No vessels detected"
    
    # Test path planning with obstacles
    buoys = detector.detect_buoys(frame)
    path = planner.plan_path(buoys, obstacles)
    assert path is not None, "No path generated with obstacles"
    
def test_buoy_specifications():
    """Test buoy detection with competition specifications."""
    camera = MockZEDCamera()
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
        assert SPEED_GATE_MIN <= distance <= SPEED_GATE_MAX, \
            f"Speed buoy at wrong distance: {distance}m"
            
    # Test path gate buoys (Polyform A-0)
    path_buoys = [b for b in buoys['red'] + buoys['green'] + buoys['yellow']
                  if b['type'] == 'path_gate' and b['distance'] >= PATH_GATE_MIN]
    assert len(path_buoys) == 3, "Should detect 3 path gate buoys"
    
    for buoy in path_buoys:
        height = buoy['dimensions'][1]
        diameter = max(buoy['dimensions'][0], buoy['dimensions'][2])
        distance = buoy['distance']
        
        assert 0.8 * 0.1524 <= height <= 1.2 * 0.1524, \
            f"Invalid path buoy height: {height}m"
        assert 0.8 * 0.203 <= diameter <= 1.2 * 0.203, \
            f"Invalid path buoy diameter: {diameter}m"
        assert distance >= PATH_GATE_MIN, \
            f"Path buoy too close: {distance}m"
