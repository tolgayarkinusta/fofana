"""Process manager for handling multiprocessing tasks."""
import multiprocessing as mp
from typing import Callable, Dict, List

class ProcessManager:
    def __init__(self):
        """Initialize process manager."""
        self.processes: Dict[str, mp.Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
        
    def start_process(self, name: str, target: Callable, args: tuple = ()) -> None:
        """Start a new process.
        
        Args:
            name: Unique name for the process
            target: Function to run in the process
            args: Arguments to pass to the target function
        """
        if name in self.processes:
            raise ValueError(f"Process {name} already exists")
            
        queue = mp.Queue()
        self.queues[name] = queue
        process = mp.Process(target=target, args=(queue,) + args)
        self.processes[name] = process
        process.start()
        
    def stop_process(self, name: str) -> None:
        """Stop a running process.
        
        Args:
            name: Name of the process to stop
        """
        if name in self.processes:
            self.processes[name].terminate()
            self.processes[name].join()
            del self.processes[name]
            del self.queues[name]
            
    def stop_all(self) -> None:
        """Stop all running processes."""
        process_names = list(self.processes.keys())
        for name in process_names:
            self.stop_process(name)
