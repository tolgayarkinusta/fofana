"""Task management module for RoboBoat competition tasks."""
from enum import Enum
from typing import Optional, Dict, Any
from ..core.mavlink_controller import USVController
from ..vision.camera import ZEDCamera
from ..navigation.buoy_detector import BuoyDetector
from ..navigation.path_planner import PathPlanner

class TaskState(Enum):
    """Task states for state machine."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskManager:
    def __init__(self):
        """Initialize task manager with required components."""
        self.usv_controller = USVController()
        self.camera = ZEDCamera()
        self.buoy_detector = BuoyDetector()
        self.path_planner = PathPlanner(self.usv_controller)
        self.current_task: Optional[str] = None
        self.task_state = TaskState.IDLE
        
    def start_task(self, task_name: str, params: Dict[str, Any] = None) -> bool:
        """Start a competition task.
        
        Args:
            task_name: Name of the task to start
            params: Optional parameters for the task
            
        Returns:
            bool: True if task started successfully
        """
        if self.task_state == TaskState.RUNNING:
            return False
            
        self.current_task = task_name
        self.task_state = TaskState.RUNNING
        
        # Initialize task-specific components
        if not self.camera.open():
            self.task_state = TaskState.FAILED
            return False
            
        try:
            self.usv_controller.arm_vehicle()
        except Exception as e:
            print(f"Failed to arm vehicle: {e}")
            self.task_state = TaskState.FAILED
            return False
            
        return True
        
    def stop_task(self) -> None:
        """Stop the current task safely."""
        if self.task_state == TaskState.RUNNING:
            self.usv_controller.stop_motors()
            self.usv_controller.disarm_vehicle()
            self.camera.close()
            self.task_state = TaskState.IDLE
            self.current_task = None
            
    def get_task_state(self) -> Dict[str, Any]:
        """Get current task state information.
        
        Returns:
            dict: Current task state and information
        """
        return {
            "task": self.current_task,
            "state": self.task_state.value,
            "armed": self.usv_controller is not None
        }
