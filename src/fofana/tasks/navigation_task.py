"""
Navigasyon görevi modülü.

Bu modül şu işlevleri içerir:
- Kırmızı-yeşil şamandıra geçişi
- Güvenli navigasyon kontrolü
- Çoklu işlem (multiprocessing) desteği
"""
import multiprocessing as mp
from typing import Optional
from ..core.mavlink_controller import USVController
from ..vision.camera import ZEDCamera
from ..navigation.buoy_detector import BuoyDetector
from ..navigation.path_planner import PathPlanner

class NavigationTask:
    def __init__(self, control_queue: mp.Queue, status_queue: mp.Queue, params: Optional[dict] = None):
        """Initialize navigation task.
        
        Args:
            control_queue: Queue for task control commands
            status_queue: Queue for task status updates
            params: Optional task parameters
        """
        self.control_queue = control_queue
        self.status_queue = status_queue
        self.usv = USVController()
        self.camera = ZEDCamera()
        self.buoy_detector = BuoyDetector(self.camera)
        self.path_planner = PathPlanner(self.usv, self.camera)
        self.running = False
        
    def run(self) -> None:
        """Run navigation task in a loop."""
        self.running = True
        self.usv.arm_vehicle()
        self.camera.open()
        
        try:
            while self.running:
                # Check control queue for commands
                if not self.control_queue.empty():
                    cmd = self.control_queue.get()
                    if cmd == "stop":
                        break
                
                # Get camera frame
                frame, depth = self.camera.get_frame()
                if frame is None:
                    continue
                    
                # Detect buoys
                buoys = self.buoy_detector.detect_buoys(frame)
                
                # Navigate through gates
                self.path_planner.navigate_through_gates(
                    buoys['red'],
                    buoys['green']
                )
                
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self.usv.stop_motors()
        self.usv.disarm_vehicle()
        self.camera.close()
        self.running = False
