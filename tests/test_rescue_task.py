"""RoboBoat 2025 kurtarma görevi testleri."""
import pytest
from unittest.mock import patch, MagicMock
from fofana.tasks.rescue_task import RescueTask

def test_water_spray_control():
    """Su püskürtme kontrolü testi."""
    with patch('fofana.core.mavlink_controller.mavutil.mavlink_connection') as mock_mavlink:
        task = RescueTask(MagicMock(), MagicMock())
        
        # Test water spray
        success = task._spray_water({'position': (100, 100)})
        
        # Verify correct PWM signals were sent
        task.usv.set_servo.assert_any_call(7, 2000)  # Start spray
        task.usv.set_servo.assert_any_call(7, 1000)  # Stop spray
        assert success == True

def test_ball_throwing_control():
    """Top fırlatma kontrolü testi."""
    with patch('fofana.core.mavlink_controller.mavutil.mavlink_connection') as mock_mavlink:
        task = RescueTask(MagicMock(), MagicMock())
        
        # Test ball throwing
        success = task._throw_ball({'position': (200, 200)})
        
        # Verify correct PWM signals were sent
        task.usv.set_servo.assert_any_call(8, 2000)  # Throw ball
        task.usv.set_servo.assert_any_call(8, 1000)  # Reset mechanism
        assert success == True
