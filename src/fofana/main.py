"""
RoboBoat 2025 ana görev çalıştırıcı.

Bu modül şu işlevleri sağlar:
1. Sistem başlatma ve doğrulama:
   - ZED2i kamera başlatma
   - SLAM ve haritalama
   - Nesne tespiti
   - Motor kontrolü

2. Görev sıralaması:
   - Navigasyon Kanalı: Kırmızı-yeşil şamandıralar arası geçiş
   - Göç Yolları: Sarı şamandıraları tespit ve sayma
   - Tehlikeli Sular: Doğru yanaşma yerine otonom yanaşma
   - Kirlilikle Yarış: Hızlı parkur tamamlama
   - Kurtarma Teslimatları: Su püskürtme ve top atma
   - Eve Dönüş: Başlangıç noktasına dönüş
"""
import time
from typing import Dict, Any, Optional
from .tasks import TaskManager
from .vision.camera import ZEDCamera

class RoboBoat2025Runner:
    def __init__(self):
        """Initialize RoboBoat runner."""
        self.camera = ZEDCamera()
        self.task_manager = TaskManager()
        self.start_position = None
        
    def initialize_systems(self) -> bool:
        """Initialize and verify all systems.
        
        Returns:
            bool: True if all systems initialized successfully
        """
        print("Sistemler başlatılıyor...")
        
        # Initialize camera
        if not self.camera.open():
            print("Kamera başlatılamadı!")
            return False
            
        # Enable positional tracking
        if not self.camera.enable_positional_tracking():
            print("Pozisyon takibi başlatılamadı!")
            self.camera.close()
            return False
            
        # Enable spatial mapping
        if not self.camera.enable_spatial_mapping():
            print("Spatial mapping başlatılamadı!")
            self.camera.close()
            return False
            
        # Enable object detection
        if not self.camera.enable_object_detection():
            print("Nesne tespiti başlatılamadı!")
            self.camera.close()
            return False
            
        # Store start position for return home
        pos = self.camera.get_position()
        if pos:
            self.start_position = (pos['x'], pos['z'])  # Store X,Z coordinates
            
        print("Tüm sistemler hazır!")
        return True
        
    def run_all_tasks(self) -> None:
        """Run all RoboBoat tasks in sequence."""
        if not self.initialize_systems():
            return
            
        try:
            tasks = [
                ('navigation', {}),
                ('mapping', {}),
                ('docking', {}),
                ('speed', {}),
                ('rescue', {}),
                ('return_home', {'start_position': self.start_position})
            ]
            
            for task_name, params in tasks:
                print(f"\n{task_name.upper()} görevi başlatılıyor...")
                
                if not self.task_manager.start_task(task_name, params):
                    print(f"{task_name} görevi başlatılamadı!")
                    return
                    
                # Monitor task status
                while True:
                    status = self.task_manager.get_task_state(task_name)
                    
                    if status.get('error'):
                        print(f"Hata: {status['error']}")
                        return
                        
                    if status['state'] == 'completed':
                        print(f"{task_name} görevi tamamlandı!")
                        break
                        
                    if status['state'] == 'failed':
                        print(f"{task_name} görevi başarısız!")
                        return
                        
                    # Print task-specific status
                    if status.get('status'):
                        self._print_task_status(task_name, status['status'])
                        
                    time.sleep(0.1)
                    
            print("\nTüm görevler başarıyla tamamlandı!")
            
        finally:
            self._cleanup()
            
    def _print_task_status(self, task_name: str, status: Dict[str, Any]) -> None:
        """Print task-specific status information.
        
        Args:
            task_name: Name of the task
            status: Task status information
        """
        if task_name == 'mapping':
            print(f"Tespit edilen sarı şamandıra: {status.get('yellow_buoys', 0)}")
            print(f"Tespit edilen siyah şamandıra: {status.get('black_buoys', 0)}")
            
        elif task_name == 'docking':
            print(f"Yanaşma durumu: {status.get('status', '')}")
            print(f"Hedef: {status.get('target_bay', '')}")
            
        elif task_name == 'speed':
            print(f"Geçen süre: {status.get('elapsed_time', 0):.1f}s")
            print(f"Hız: {status.get('speed', 0):.1f} m/s")
            
        elif task_name == 'rescue':
            print(f"Turuncu hedefler: {status.get('orange_targets', 0)}/3")
            print(f"Siyah hedefler: {status.get('black_targets', 0)}/3")
            
        elif task_name == 'return_home':
            print(f"Eve dönüş durumu: {status.get('status', '')}")
            if 'distance' in status:
                print(f"Kalan mesafe: {status['distance']:.1f}m")
        
    def _cleanup(self) -> None:
        """Clean up all tasks and resources."""
        print("\nSistem kapatılıyor...")
        self.task_manager.stop_all_tasks()
        self.camera.close()
        
if __name__ == "__main__":
    runner = RoboBoat2025Runner()
    runner.run_all_tasks()
