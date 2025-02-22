"""
Yol planlama ve navigasyon kontrol modülü.

Özellikler:
- 3B ortam haritalama ve engel algılama
- A* algoritması ile yol planlama
- Şamandıralar arası güvenli geçiş planlama
- Costmap tabanlı engel kaçınma
- PWM tabanlı motor hız kontrolü
- Oransal kontrol ile dönüş hesaplama
"""
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
        
        # Costmap parameters
        self.resolution = 0.1  # 10cm per cell
        self.map_size = int(20.0 / self.resolution)  # 20m x 20m area
        self.costmap = None
        
    def set_target(self, x: float, y: float) -> None:
        """Set target position to navigate to.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
        """
        self.target_position = (x, y)
        
    def update_costmap(self) -> None:
        """Update costmap using ZED spatial mapping."""
        if not self.camera.mapping_enabled:
            return
            
        spatial_map = self.camera.get_spatial_map()
        if spatial_map is None:
            return
            
        # Create empty costmap
        self.costmap = np.zeros((self.map_size, self.map_size))
        
        # Convert spatial map vertices to costmap
        for vertex in spatial_map.vertices:
            x = int((vertex[0] + 10.0) / self.resolution)  # Offset by 10m to center map
            y = int((vertex[2] + 10.0) / self.resolution)  # Use Z as Y in top-down view
            
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                # Add cost based on height (obstacles)
                self.costmap[y, x] = 1.0
                
        # Add padding around obstacles
        kernel = np.ones((5, 5))  # 50cm padding
        self.costmap = np.minimum(1.0, np.maximum(0.0, 
            self.costmap + 0.5 * np.array(self.costmap > 0.5, dtype=float)))
            
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
                
                # Check if gate width is reasonable (1.8-2.4m)
                if 1.8 <= gate_width <= 2.4:
                    # Calculate distance to gate center
                    gate_center = (
                        (red_pos[0] + green_pos[0]) / 2,
                        (red_pos[1] + green_pos[1]) / 2,
                        (red_pos[2] + green_pos[2]) / 2
                    )
                    
                    distance = np.sqrt((gate_center[0] - current_pos['x'])**2 +
                                     (gate_center[2] - current_pos['z'])**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_gate = (red, green)
                        
        return closest_gate
        
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
