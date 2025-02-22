"""
Haritalama görevi modülü.

Bu modül şu işlevleri içerir:
- Sarı şamandıra tespiti ve sayımı
- Engel algılama ve kaçınma
- Gerçek zamanlı durum raporlama
"""
import multiprocessing as mp
from typing import Dict, List, Tuple
from ..core.mavlink_controller import USVController
from ..vision.camera import ZEDCamera
from ..navigation.buoy_detector import BuoyDetector
from ..navigation.path_planner import PathPlanner

class MappingTask:
    def __init__(self, control_queue: mp.Queue, status_queue: mp.Queue):
        """Initialize mapping task.
        
        Args:
            control_queue: Queue for task control commands
            status_queue: Queue for status updates
        """
        self.control_queue = control_queue
        self.status_queue = status_queue
        self.usv = USVController()
        self.camera = ZEDCamera()
        self.buoy_detector = BuoyDetector()
        self.path_planner = PathPlanner(self.usv)
        self.running = False
        self.yellow_buoy_count = 0
        self.mapped_area = set()  # Track explored areas
        
    def run(self) -> None:
        """Run mapping task in a loop."""
        self.running = True
        self.usv.arm_vehicle()
        self.camera.open()
        
        try:
            while self.running:
                # Check control queue
                if not self.control_queue.empty():
                    cmd = self.control_queue.get()
                    if cmd == "stop":
                        break
                
                # Get camera frame and point cloud
                frame, depth = self.camera.get_frame()
                point_cloud = self.camera.get_point_cloud()
                
                if frame is None:
                    continue
                    
                # Detect buoys
                buoys = self.buoy_detector.detect_buoys(frame)
                
                # Count yellow buoys
                new_yellow_buoys = self._process_yellow_buoys(
                    buoys['yellow'],
                    point_cloud
                )
                self.yellow_buoy_count += new_yellow_buoys
                
                # Send status update
                self._send_status_update()
                
        finally:
            self.cleanup()
            
    def _process_yellow_buoys(self, yellow_buoys: List[Tuple[int, int, int]], 
                            point_cloud) -> int:
        """Process detected yellow buoys.
        
        Args:
            yellow_buoys: List of detected yellow buoy positions
            point_cloud: 3D point cloud data
            
        Returns:
            int: Number of new yellow buoys detected
        """
        new_buoys = 0
        for x, y, radius in yellow_buoys:
            # Convert to 3D position
            pos = (int(x/10), int(y/10))  # Simplified grid position
            if pos not in self.mapped_area:
                self.mapped_area.add(pos)
                new_buoys += 1
        return new_buoys
        
    def _send_status_update(self) -> None:
        """Send status update through queue."""
        status = {
            "yellow_buoys": self.yellow_buoy_count,
            "mapped_area": len(self.mapped_area)
        }
        self.status_queue.put(status)
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.usv.stop_motors()
        self.usv.disarm_vehicle()
        self.camera.close()
        self.running = False
