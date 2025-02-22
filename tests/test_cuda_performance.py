"""Test CUDA performance with ZED camera."""
import time
import pytest
from fofana.vision.camera import ZEDCamera

def test_camera_fps():
    """Test camera frame rate with CUDA acceleration."""
    camera = ZEDCamera()
    
    # Initialize camera
    assert camera.open(), "Failed to open camera"
    assert camera.enable_positional_tracking(), "Failed to enable tracking"
    assert camera.enable_spatial_mapping(), "Failed to enable mapping"
    assert camera.enable_object_detection(), "Failed to enable detection"
    
    # Test frame rate
    frames = 0
    start_time = time.time()
    
    while frames < 100:  # Capture 100 frames
        frame, depth, pose = camera.get_frame()
        if frame is not None:
            frames += 1
            
        # Ensure frames are on GPU
        assert frame.is_cuda, "Frame not on CUDA device"
        assert depth.is_cuda, "Depth not on CUDA device"
            
    duration = time.time() - start_time
    fps = frames / duration
    
    print(f"\nPerformance Results:")
    print(f"Frames captured: {frames}")
    print(f"Duration: {duration:.2f}s")
    print(f"Frame rate: {fps:.1f} FPS")
    
    # Verify minimum performance
    assert fps >= 30.0, f"Frame rate {fps:.1f} FPS below minimum 30 FPS"
    
    camera.close()

def test_cuda_memory():
    """Test CUDA memory usage."""
    import torch
    
    camera = ZEDCamera()
    camera.open()
    
    # Get initial memory usage
    init_memory = torch.cuda.memory_allocated()
    
    # Capture frames and check memory
    for _ in range(10):
        frame, depth, pose = camera.get_frame()
        
        # Ensure tensors are properly freed
        del frame
        del depth
        torch.cuda.empty_cache()
        
    final_memory = torch.cuda.memory_allocated()
    
    # Verify no memory leaks
    assert final_memory <= init_memory * 1.1, "Significant CUDA memory leak detected"
    
    camera.close()
