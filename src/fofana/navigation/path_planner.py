"""
Yol planlama ve navigasyon kontrol modülü.

Özellikler:
- Şamandıralar arası güvenli geçiş planlama
- Engel algılama ve kaçınma
- Hedef noktaya gitme kontrolü
- PWM tabanlı motor hız kontrolü
- Oransal kontrol ile dönüş hesaplama
"""
import numpy as np
from typing import Tuple, List, Optional
from ..core.motor_control import MotorController

class PathPlanner:
    def __init__(self, motor_controller: MotorController):
        """Initialize path planner.
        
        Args:
            motor_controller: Motor controller instance
        """
        self.motor_controller = motor_controller
        self.target_position = None
        self.min_distance = 1.0  # Minimum distance to maintain from buoys
        
    def set_target(self, x: float, y: float) -> None:
        """Set target position to navigate to.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
        """
        self.target_position = (x, y)
        
    def navigate_through_gates(self, red_buoys: List[Tuple[int, int, int]], 
                             green_buoys: List[Tuple[int, int, int]]) -> None:
        """Navigate through red-green buoy gates.
        
        Args:
            red_buoys: List of (x, y, radius) for red buoys
            green_buoys: List of (x, y, radius) for green buoys
        """
        if not red_buoys or not green_buoys:
            # No clear gate detected, stop
            self.motor_controller.set_motor_speeds(0, 0)
            return
            
        # Find closest gate
        gate = self._find_closest_gate(red_buoys, green_buoys)
        if not gate:
            return
            
        red_x, red_y, _ = gate[0]
        green_x, green_y, _ = gate[1]
        
        # Calculate gate center
        gate_center_x = (red_x + green_x) // 2
        gate_center_y = (red_y + green_y) // 2
        
        # Calculate heading error (assuming camera center is at frame_width/2)
        frame_center_x = 640  # Assuming 1280x720 resolution
        heading_error = gate_center_x - frame_center_x
        
        # Simple proportional control
        max_speed = 50  # Maximum PWM value
        turn_factor = 0.1  # Steering sensitivity
        
        base_speed = 30
        turn_adjustment = int(heading_error * turn_factor)
        
        # Calculate motor speeds
        left_speed = base_speed - turn_adjustment
        right_speed = base_speed + turn_adjustment
        
        # Limit speeds
        left_speed = max(-max_speed, min(max_speed, left_speed))
        right_speed = max(-max_speed, min(max_speed, right_speed))
        
        self.motor_controller.set_motor_speeds(left_speed, right_speed)
        
    def _find_closest_gate(self, red_buoys: List[Tuple[int, int, int]], 
                          green_buoys: List[Tuple[int, int, int]]) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Find the closest valid gate formed by red and green buoys.
        
        Args:
            red_buoys: List of red buoy positions
            green_buoys: List of green buoy positions
            
        Returns:
            Tuple of (red_buoy, green_buoy) forming the closest gate, or None
        """
        min_distance = float('inf')
        closest_gate = None
        
        for red in red_buoys:
            for green in green_buoys:
                # Check if buoys form a reasonable gate
                distance = np.sqrt((red[0] - green[0])**2 + (red[1] - green[1])**2)
                if 100 < distance < 500:  # Reasonable gate width in pixels
                    if distance < min_distance:
                        min_distance = distance
                        closest_gate = (red, green)
                        
        return closest_gate
