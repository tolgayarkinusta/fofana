"""Path planning module for RoboBoat 2025."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.mavlink_controller import USVController

class PathPlanner:
    def __init__(self, controller: USVController, camera):
        """Initialize path planner.
        
        Args:
            controller: USVController instance
            camera: ZEDCamera instance
        """
        self.controller = controller
        self.camera = camera
        
        # Planning parameters
        self.safety_margin = 1.0  # meters
        self.max_speed = 2.0  # m/s
        self.min_speed = 0.5  # m/s
        
    def plan_path(self, buoys: Dict[str, List[Dict]], obstacles: Optional[Dict[str, List[Dict]]] = None) -> List[Dict]:
        """Plan path through buoys avoiding obstacles.
        
        Args:
            buoys: Detected buoys by color and type
            obstacles: Optional detected obstacles
            
        Returns:
            List[Dict]: Waypoints with position and speed
        """
        waypoints = []
        
        # Get navigation gate buoys
        nav_buoys = [b for b in buoys['red'] + buoys['green']
                    if b['type'] == 'navigation_gate']
        
        # Get speed gate buoys
        speed_buoys = [b for b in buoys['red'] + buoys['green'] + buoys['black']
                      if b['type'] == 'speed_gate']
                      
        # Get path gate buoys
        path_buoys = [b for b in buoys['red'] + buoys['green'] + buoys['yellow']
                     if b['type'] == 'path_gate']
                     
        # Plan through navigation gates
        for i in range(0, len(nav_buoys), 2):
            if i + 1 < len(nav_buoys):
                gate_center = self._get_gate_center(nav_buoys[i], nav_buoys[i+1])
                waypoints.append({
                    'position': gate_center,
                    'speed': self.max_speed
                })
                
        # Plan through speed gates
        for i in range(0, len(speed_buoys), 3):
            if i + 2 < len(speed_buoys):
                gate_center = self._get_gate_center(speed_buoys[i], speed_buoys[i+1])
                waypoints.append({
                    'position': gate_center,
                    'speed': self.max_speed
                })
                
        # Plan through path gates avoiding obstacles
        if obstacles:
            yellow_buoys = obstacles.get('yellow_buoys', [])
            vessels = obstacles.get('stationary_vessels', [])
            
            # Add safety margins around obstacles
            obstacle_positions = []
            for buoy in yellow_buoys:
                obstacle_positions.append((
                    buoy['position'],
                    self.safety_margin + max(buoy['dimensions'][0], buoy['dimensions'][2])/2
                ))
            for vessel in vessels:
                obstacle_positions.append((
                    vessel['position'],
                    self.safety_margin + max(vessel['dimensions'][0], vessel['dimensions'][2])/2
                ))
                
            # Plan path avoiding obstacles
            for i in range(0, len(path_buoys), 2):
                if i + 1 < len(path_buoys):
                    gate_center = self._get_gate_center(path_buoys[i], path_buoys[i+1])
                    safe_point = self._find_safe_point(gate_center, obstacle_positions)
                    waypoints.append({
                        'position': safe_point,
                        'speed': self.min_speed
                    })
                    
        return waypoints
        
    def _get_gate_center(self, buoy1: Dict, buoy2: Dict) -> Tuple[float, float, float]:
        """Calculate center point between two buoys."""
        return (
            (buoy1['position'][0] + buoy2['position'][0])/2,
            (buoy1['position'][1] + buoy2['position'][1])/2,
            (buoy1['position'][2] + buoy2['position'][2])/2
        )
        
    def _find_safe_point(self, target: Tuple[float, float, float],
                        obstacles: List[Tuple[Tuple[float, float, float], float]]) -> Tuple[float, float, float]:
        """Find safe point near target avoiding obstacles.
        
        Args:
            target: Target position (x, y, z)
            obstacles: List of (position, radius) tuples
            
        Returns:
            Tuple[float, float, float]: Safe position
        """
        # Check if target is safe
        for obs_pos, obs_radius in obstacles:
            dist = np.sqrt((target[0] - obs_pos[0])**2 + (target[2] - obs_pos[2])**2)
            if dist < obs_radius:
                # Move away from obstacle
                dx = target[0] - obs_pos[0]
                dz = target[2] - obs_pos[2]
                norm = np.sqrt(dx**2 + dz**2)
                safe_dist = obs_radius + self.safety_margin
                return (
                    obs_pos[0] + dx/norm * safe_dist,
                    target[1],
                    obs_pos[2] + dz/norm * safe_dist
                )
        return target
