"""
Yol planlama ve navigasyon kontrol modülü.

Özellikler:
- 3B ortam haritalama ve engel algılama
- A* algoritması ile yol planlama
- Şamandıralar arası güvenli geçiş planlama:
  * İlk iki geçit arası: 25-100 feet (7.62-30.48m)
  * Tüm geçitler: 6-10 feet genişlik (1.83-3.05m)
- Costmap tabanlı engel kaçınma
- PWM tabanlı motor hız kontrolü
- Oransal kontrol ile dönüş hesaplama
"""
# Şamandıra geçit sabitleri
GATE_WIDTH_MIN = 1.83        # 6 feet (metre cinsinden)
GATE_WIDTH_MAX = 3.05        # 10 feet (metre cinsinden)
FIRST_GATE_SPACING_MIN = 7.62   # 25 feet (metre cinsinden)
FIRST_GATE_SPACING_MAX = 30.48  # 100 feet (metre cinsinden)

import numpy as np
from typing import Tuple, List, Optional, Dict
from ..core.motor_control import MotorController
from ..vision.camera import ZEDCamera
import heapq

class PathPlanner:
    def __init__(self, motor_controller: MotorController, camera: ZEDCamera):
        """Initialize path planner.
        
        Args:
            motor_controller: Motor controller instance
            camera: ZED camera instance for spatial mapping
        """
        self.motor_controller = motor_controller
        self.camera = camera
        self.target_position = None
        self.min_distance = 1.0  # Minimum distance to maintain from buoys
        self.gates_passed = 0    # Geçilen geçit sayısı
        self.last_gate_center = None  # Son geçilen geçitin merkezi
        
        # Costmap parameters
        self.resolution = 0.1  # 10cm per cell
        self.map_size = int(20.0 / self.resolution)  # 20m x 20m area
        self.costmap = None
        
        # Obstacle detection
        from ..navigation.buoy_detector import BuoyDetector
        self.buoy_detector = BuoyDetector(camera)
        
    def set_target(self, x: float, y: float) -> None:
        """Set target position to navigate to.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
        """
        self.target_position = (x, y)
        
    def update_costmap(self) -> None:
        """Update costmap with detected obstacles."""
        # Create empty costmap
        self.costmap = np.zeros((self.map_size, self.map_size))
        
        # Get obstacles from detector
        obstacles = self.detect_obstacles()
        if not obstacles:
            return
        
        # Add yellow buoys with larger padding (endangered species)
        for buoy in obstacles['yellow_buoys']:
            x, y = self._world_to_costmap(buoy['position'])
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                radius = int(1.0 / self.resolution)  # 1m safety radius
                self._add_circular_cost(x, y, radius, 0.8)
        
        # Add vessels with variable padding
        for vessel in obstacles['stationary_vessels']:
            x, y = self._world_to_costmap(vessel['position'])
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                dims = vessel['dimensions']
                radius = int(max(dims) / self.resolution) + 5  # 50cm extra
                self._add_circular_cost(x, y, radius, 1.0)
                
        # Add other obstacles with standard padding
        for obstacle in obstacles['other']:
            x, y = self._world_to_costmap(obstacle['position'])
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                radius = int(0.5 / self.resolution)  # 50cm padding
                self._add_circular_cost(x, y, radius, 0.5)
            
    def navigate_through_gates(self, red_buoys: List[Dict], green_buoys: List[Dict]) -> None:
        """Navigate through red-green buoy gates using spatial mapping.
        
        Args:
            red_buoys: List of buoy dicts with position and dimensions
            green_buoys: List of buoy dicts with position and dimensions
        """
        if not red_buoys or not green_buoys:
            self.motor_controller.set_motor_speeds(0, 0)
            return
            
        # Update environment map
        self.update_costmap()
        
        # Find closest gate
        gate = self._find_closest_gate(red_buoys, green_buoys)
        if not gate:
            return
            
        red_pos = gate[0]['position']
        green_pos = gate[1]['position']
        
        # Calculate gate center in 3D space
        gate_center = (
            (red_pos[0] + green_pos[0]) / 2,
            (red_pos[1] + green_pos[1]) / 2,
            (red_pos[2] + green_pos[2]) / 2
        )
        
        # Get current position
        current_pos = self.camera.get_position()
        if not current_pos:
            return
            
        # Plan path to gate
        start = (int((current_pos['x'] + 10.0) / self.resolution),
                int((current_pos['z'] + 10.0) / self.resolution))
        goal = (int((gate_center[0] + 10.0) / self.resolution),
               int((gate_center[2] + 10.0) / self.resolution))
               
        path = self._astar_search(start, goal)
        if not path:
            self.motor_controller.set_motor_speeds(0, 0)
            return
            
        # Follow path using proportional control
        next_point = path[1] if len(path) > 1 else path[0]
        dx = next_point[0] - start[0]
        dy = next_point[1] - start[1]
        
        # Calculate desired heading
        desired_heading = np.arctan2(dy, dx)
        current_heading = current_pos['yaw']
        
        # Calculate heading error
        heading_error = np.arctan2(np.sin(desired_heading - current_heading),
                                 np.cos(desired_heading - current_heading))
        
        # Proportional control
        max_speed = 50
        turn_factor = 30.0  # Increased for better responsiveness
        
        base_speed = 30
        turn_adjustment = int(heading_error * turn_factor)
        
        # Calculate motor speeds
        left_speed = base_speed - turn_adjustment
        right_speed = base_speed + turn_adjustment
        
        # Limit speeds
        left_speed = max(-max_speed, min(max_speed, left_speed))
        right_speed = max(-max_speed, min(max_speed, right_speed))
        
        self.motor_controller.set_motor_speeds(left_speed, right_speed)
        
    def _find_closest_gate(self, red_buoys: List[Dict], 
                          green_buoys: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
        """Find the closest valid gate formed by red and green buoys.
        
        Args:
            red_buoys: List of red buoy dicts with position and dimensions
            green_buoys: List of green buoy dicts with position and dimensions
            
        Returns:
            Tuple of (red_buoy, green_buoy) forming the closest gate, or None
        """
        min_distance = float('inf')
        closest_gate = None
        
        # Get current position
        current_pos = self.camera.get_position()
        if not current_pos:
            return None
            
        for red in red_buoys:
            for green in green_buoys:
                # Calculate gate width
                red_pos = red['position']
                green_pos = green['position']
                gate_width = np.sqrt((red_pos[0] - green_pos[0])**2 + 
                                   (red_pos[2] - green_pos[2])**2)
                
                # Geçit genişliğini kontrol et (6-10 feet / 1.83-3.05m)
                if not (GATE_WIDTH_MIN <= gate_width <= GATE_WIDTH_MAX):
                    continue
                    
                # Geçit merkezi hesapla
                gate_center = (
                    (red_pos[0] + green_pos[0]) / 2,
                    (red_pos[1] + green_pos[1]) / 2,
                    (red_pos[2] + green_pos[2]) / 2
                )
                
                # İlk iki geçit arası mesafe kontrolü (25-100 feet)
                if self.gates_passed == 1 and self.last_gate_center:
                    gate_spacing = np.sqrt(
                        (gate_center[0] - self.last_gate_center[0])**2 +
                        (gate_center[2] - self.last_gate_center[2])**2
                    )
                    if not (FIRST_GATE_SPACING_MIN <= gate_spacing <= 
                           FIRST_GATE_SPACING_MAX):
                        continue
                
                # En yakın geçidi seç
                distance = np.sqrt((gate_center[0] - current_pos['x'])**2 +
                                 (gate_center[2] - current_pos['z'])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_gate = (red, green)
                    
                    # Geçit geçildiğinde güncelle
                    if distance < 2.0:  # 2m yakınlığa gelince geçilmiş say
                        self.gates_passed += 1
                        self.last_gate_center = gate_center
                        # Geçit geçildiğinde sayacı artır
                        if distance < 2.0:  # 2m yakınlığa gelince geçilmiş say
                            self.gates_passed += 1
                        
        return closest_gate
        
    def _world_to_costmap(self, position: Tuple[float, float, float]) -> Tuple[int, int]:
        """Convert world coordinates to costmap coordinates.
        
        Args:
            position: (x, y, z) world coordinates in meters
            
        Returns:
            Tuple[int, int]: (x, y) costmap coordinates
        """
        # Ensure coordinates are within map bounds
        x = min(max(0, int((position[0] + self.map_size/2) / self.resolution)), self.map_size-1)
        y = min(max(0, int((position[2] + self.map_size/2) / self.resolution)), self.map_size-1)
        return x, y
        
    def detect_obstacles(self) -> Optional[Dict[str, List[Dict]]]:
        """Detect obstacles using buoy detector.
        
        Returns:
            Optional[Dict[str, List[Dict]]]: Detected obstacles by category
        """
        frame = self.camera.get_frame()[0]  # Get RGB frame
        if frame is None:
            return None
            
        return self.buoy_detector.detect_obstacles(frame)
        
    def _add_circular_cost(self, x: int, y: int, radius: int, cost: float) -> None:
        """Add circular cost region to costmap.
        
        Args:
            x: Center X coordinate in costmap
            y: Center Y coordinate in costmap
            radius: Radius in costmap cells
            cost: Cost value to add (0.0-1.0)
        """
        y_indices, x_indices = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x_indices**2 + y_indices**2 <= radius**2
        
        x_start = max(0, x - radius)
        x_end = min(self.map_size, x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(self.map_size, y + radius + 1)
        
        mask_start_y = max(0, radius - y)
        mask_end_y = mask_start_y + (y_end - y_start)
        mask_start_x = max(0, radius - x)
        mask_end_x = mask_start_x + (x_end - x_start)
        
        self.costmap[y_start:y_end, x_start:x_end] = np.maximum(
            self.costmap[y_start:y_end, x_start:x_end],
            cost * mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
        )
        
    def _astar_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* path planning algorithm.
        
        Args:
            start: Start position (x, y) in costmap coordinates
            goal: Goal position (x, y) in costmap coordinates
            
        Returns:
            List of (x, y) positions forming the path
        """
        if self.costmap is None:
            return []
            
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
            
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), 
                          (1,1), (1,-1), (-1,1), (-1,-1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if (0 <= new_pos[0] < self.map_size and 
                    0 <= new_pos[1] < self.map_size and
                    self.costmap[new_pos[1], new_pos[0]] < 0.5):
                    neighbors.append(new_pos)
            return neighbors
            
        # Initialize data structures
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
                    
        # Reconstruct path
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        return path
