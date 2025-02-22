"""
RoboBoat 2025 ana görev çalıştırıcı.

Bu modül tüm görevleri sırayla çalıştırır:
1. Navigasyon Kanalı
2. Göç Yolları (Sarı şamandıra tespiti)
3. Tehlikeli Sular (Yanaşma)
4. Kurtarma Teslimatları
5. Eve Dönüş
"""
import multiprocessing as mp
from typing import Dict, Any
import time
from .tasks import (
    TaskManager, NavigationTask, MappingTask,
    DockingTask, RescueTask, ReturnHomeTask
)

class RoboBoat2025Runner:
    def __init__(self):
        """Initialize RoboBoat runner."""
        self.task_manager = TaskManager()
        self.control_queues: Dict[str, mp.Queue] = {}
        self.status_queues: Dict[str, mp.Queue] = {}
        self.processes: Dict[str, mp.Process] = {}
        
    def run_all_tasks(self) -> None:
        """Run all RoboBoat tasks in sequence."""
        try:
            # 1. Navigasyon Kanalı
            if not self._run_navigation_task():
                return
                
            # 2. Göç Yolları
            if not self._run_mapping_task():
                return
                
            # 3. Tehlikeli Sular
            if not self._run_docking_task():
                return
                
            # 4. Kurtarma Teslimatları
            if not self._run_rescue_task():
                return
                
            # 5. Eve Dönüş
            if not self._run_return_home_task():
                return
                
            print("Tüm görevler başarıyla tamamlandı!")
            
        finally:
            self._cleanup()
            
    def _run_navigation_task(self) -> bool:
        """Run navigation task."""
        print("Navigasyon görevi başlatılıyor...")
        control_queue = mp.Queue()
        status_queue = mp.Queue()
        
        task = NavigationTask(control_queue)
        process = mp.Process(target=task.run)
        process.start()
        
        # Monitor task status
        while process.is_alive():
            if not status_queue.empty():
                status = status_queue.get()
                print(f"Navigasyon durumu: {status}")
            time.sleep(0.1)
            
        return process.exitcode == 0
        
    def _run_mapping_task(self) -> bool:
        """Run mapping task."""
        print("Haritalama görevi başlatılıyor...")
        control_queue = mp.Queue()
        status_queue = mp.Queue()
        
        task = MappingTask(control_queue, status_queue)
        process = mp.Process(target=task.run)
        process.start()
        
        yellow_buoys = 0
        while process.is_alive():
            if not status_queue.empty():
                status = status_queue.get()
                yellow_buoys = status.get('yellow_buoys', 0)
                print(f"Tespit edilen sarı şamandıra: {yellow_buoys}")
            time.sleep(0.1)
            
        return process.exitcode == 0
        
    def _run_docking_task(self) -> bool:
        """Run docking task."""
        print("Yanaşma görevi başlatılıyor...")
        control_queue = mp.Queue()
        status_queue = mp.Queue()
        
        task = DockingTask(control_queue, status_queue)
        process = mp.Process(target=task.run)
        process.start()
        
        while process.is_alive():
            if not status_queue.empty():
                status = status_queue.get()
                print(f"Yanaşma durumu: {status['status']}")
            time.sleep(0.1)
            
        return process.exitcode == 0
        
    def _run_rescue_task(self) -> bool:
        """Run rescue task."""
        print("Kurtarma görevi başlatılıyor...")
        control_queue = mp.Queue()
        status_queue = mp.Queue()
        
        task = RescueTask(control_queue, status_queue)
        process = mp.Process(target=task.run)
        process.start()
        
        while process.is_alive():
            if not status_queue.empty():
                status = status_queue.get()
                print(f"Turuncu hedefler: {status['orange_targets']}/3")
                print(f"Siyah hedefler: {status['black_targets']}/3")
            time.sleep(0.1)
            
        return process.exitcode == 0
        
    def _run_return_home_task(self) -> bool:
        """Run return home task."""
        print("Eve dönüş görevi başlatılıyor...")
        control_queue = mp.Queue()
        status_queue = mp.Queue()
        
        task = ReturnHomeTask(control_queue, status_queue, (0, 0))
        process = mp.Process(target=task.run)
        process.start()
        
        while process.is_alive():
            if not status_queue.empty():
                status = status_queue.get()
                print(f"Eve dönüş durumu: {status['status']}")
            time.sleep(0.1)
            
        return process.exitcode == 0
        
    def _cleanup(self) -> None:
        """Clean up all processes and resources."""
        for queue in self.control_queues.values():
            queue.put("stop")
        
        for process in self.processes.values():
            if process.is_alive():
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    
if __name__ == "__main__":
    runner = RoboBoat2025Runner()
    runner.run_all_tasks()
