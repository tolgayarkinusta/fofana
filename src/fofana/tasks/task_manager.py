"""
RoboBoat yarışması görev yönetim modülü.

Bu modül şu görevleri yönetir:
1. Navigasyon Kanalı: Kırmızı-yeşil şamandıralar arası geçiş
2. Göç Yolları: Sarı şamandıraları tespit ve sayma
3. Tehlikeli Sular: Doğru yanaşma yerine otonom yanaşma
4. Kirlilikle Yarış: Hızlı parkur tamamlama
5. Kurtarma Teslimatları: Su püskürtme ve top atma
6. Eve Dönüş: Başlangıç noktasına dönüş

Her görev için:
- Multiprocessing ile paralel çalışma
- İşlemler arası haberleşme (queue)
- Güvenli başlatma/durdurma
- Durum takibi ve hata yönetimi
"""
import multiprocessing as mp
from enum import Enum
from typing import Optional, Dict, Any
from ..core.mavlink_controller import USVController
from ..vision.camera import ZEDCamera
from ..navigation.buoy_detector import BuoyDetector
from ..navigation.path_planner import PathPlanner
from .navigation_task import NavigationTask
from .mapping_task import MappingTask
from .docking_task import DockingTask
from .rescue_task import RescueTask
from .return_home_task import ReturnHomeTask

class TaskState(Enum):
    """Task states for state machine."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskManager:
    def __init__(self, camera: Optional[ZEDCamera] = None, controller: Optional[USVController] = None):
        """Initialize task manager with required components.
        
        Args:
            camera: Optional ZEDCamera instance
            controller: Optional USVController instance
        """
        self.camera = camera or ZEDCamera()
        self.usv_controller = controller or USVController()
        self.buoy_detector = BuoyDetector(self.camera)
        self.path_planner = PathPlanner(self.usv_controller, self.camera)
        
        # Task registry
        self.tasks = {
            'navigation': NavigationTask,
            'mapping': MappingTask,
            'docking': DockingTask,
            'rescue': RescueTask,
            'return_home': ReturnHomeTask
        }
        
        # Active processes and queues
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        self.task_state = TaskState.IDLE
        
    def start_task(self, task_name: str, params: Dict[str, Any] = None) -> bool:
        """Start a competition task in a separate process.
        
        Args:
            task_name: Name of the task to start
            params: Optional parameters for the task
            
        Returns:
            bool: True if task started successfully
        """
        if task_name not in self.tasks:
            print(f"Unknown task: {task_name}")
            return False
            
        if task_name in self.active_processes:
            print(f"Task {task_name} is already running")
            return False
            
        # Create communication queues
        control_queue = mp.Queue()
        status_queue = mp.Queue()
        
        # Initialize camera and vehicle
        if not self.camera.open():
            print("Failed to open camera")
            return False
            
        try:
            self.usv_controller.arm_vehicle()
        except Exception as e:
            print(f"Failed to arm vehicle: {e}")
            self.camera.close()
            return False
            
        # Start task process
        process = mp.Process(
            target=self.tasks[task_name],
            args=(control_queue, status_queue),
            kwargs={'params': params} if params else {}
        )
        process.start()
        
        # Store process information
        self.active_processes[task_name] = {
            'process': process,
            'control_queue': control_queue,
            'status_queue': status_queue,
            'state': TaskState.RUNNING
        }
        
        return True
        
    def stop_task(self, task_name: str) -> None:
        """Stop a specific task safely.
        
        Args:
            task_name: Name of the task to stop
        """
        if task_name not in self.active_processes:
            return
            
        process_info = self.active_processes[task_name]
        
        # Send stop command to task
        process_info['control_queue'].put('stop')
        
        # Wait for process to finish (with timeout)
        process_info['process'].join(timeout=5)
        
        # Force terminate if still running
        if process_info['process'].is_alive():
            process_info['process'].terminate()
            process_info['process'].join()
            
        # Cleanup
        process_info['control_queue'].close()
        process_info['status_queue'].close()
        del self.active_processes[task_name]
        
        # Stop hardware if no more tasks
        if not self.active_processes:
            self.usv_controller.stop_motors()
            self.usv_controller.disarm_vehicle()
            self.camera.close()
            
    def get_task_state(self, task_name: str = None) -> Dict[str, Any]:
        """Get current task state information.
        
        Args:
            task_name: Optional name of task to get state for. If None, returns first active task.
            
        Returns:
            dict: Task state information including running state
        """
        if task_name and task_name in self.active_processes:
            process_info = self.active_processes[task_name]
            return {
                "task": task_name,
                "state": process_info['state'].value,
                "running": True,  # Task is running if it exists in active_processes
                "status": process_info['status_queue'].get() if not process_info['status_queue'].empty() else None
            }
            
        # Get first active task if no specific task requested
        active_tasks = list(self.active_processes.items())
        if not active_tasks:
            return {"state": "idle", "running": False}
        
        # Get latest status update
        status = None
        while not process_info['status_queue'].empty():
            status = process_info['status_queue'].get()
            
        return {
            "task": task_name,
            "state": process_info['state'].value,
            "status": status
        }
        
    def stop_all_tasks(self) -> None:
        """Stop all running tasks safely."""
        for task_name in list(self.active_processes.keys()):
            self.stop_task(task_name)
