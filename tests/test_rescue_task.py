"""RoboBoat 2025 kurtarma görevi testleri."""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add mock modules to path
sys.path.insert(0, os.path.dirname(__file__))
import mock_zed_sdk
import mock_serial

sys.modules['pyzed.sl'] = mock_zed_sdk
sys.modules['serial'] = mock_serial

from fofana.tasks.rescue_task import RescueTask

def test_water_spray_control():
    """Su püskürtme kontrolü testi."""
    mock_usv = MagicMock()
    with patch('fofana.core.mavlink_controller.mavutil.mavlink_connection'):
        task = RescueTask(MagicMock(), MagicMock())
        task.usv = mock_usv  # Replace USV controller with mock
        
        # Test water spray
        success = task._spray_water({'position': (100, 100)})
        
        # Verify correct PWM signals were sent
        mock_usv.set_servo.assert_any_call(7, 2000)  # Start spray
        mock_usv.set_servo.assert_any_call(7, 1000)  # Stop spray
        assert success == True

def test_ball_throwing_control():
    """Top fırlatma kontrolü testi."""
    mock_usv = MagicMock()
    with patch('fofana.core.mavlink_controller.mavutil.mavlink_connection'):
        task = RescueTask(MagicMock(), MagicMock())
        task.usv = mock_usv  # Replace USV controller with mock
        
        # Test ball throwing
        success = task._throw_ball({'position': (200, 200)})
        
        # Verify correct PWM signals were sent
        mock_usv.set_servo.assert_any_call(8, 2000)  # Throw ball
        mock_usv.set_servo.assert_any_call(8, 1000)  # Reset mechanism
        assert success == True
