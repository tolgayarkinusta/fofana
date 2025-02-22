"""
ZED2i kamera entegrasyon modülü (CUDA hızlandırma destekli).

Özellikler:
- HD720 çözünürlükte görüntü yakalama
- CUDA destekli derinlik algılama
- Stereo görüş ile 3B nokta bulutu oluşturma
- Gerçek zamanlı görüntü işleme için optimize edilmiş
"""
import pyzed.sl as sl
import numpy as np
import torch

class ZEDCamera:
    def __init__(self):
        """Initialize ZED2i camera with CUDA acceleration."""
        self.zed = sl.Camera()
        
        # Create camera configuration
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.sdk_cuda_ctx = True  # Enable CUDA context sharing
        
        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        
    def open(self) -> bool:
        """Open the camera connection.
        
        Returns:
            bool: True if camera opened successfully
        """
        status = self.zed.open(self.init_params)
        return status == sl.ERROR_CODE.SUCCESS
        
    def close(self) -> None:
        """Close the camera connection."""
        self.zed.close()
        
    def get_frame(self) -> tuple:
        """Capture and retrieve camera frame with depth.
        
        Returns:
            tuple: (image_np, depth_np) - RGB image and depth map as numpy arrays
        """
        image = sl.Mat()
        depth = sl.Mat()
        
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            
            return image.get_data(), depth.get_data()
        return None, None
        
    def get_point_cloud(self) -> np.ndarray:
        """Get 3D point cloud data.
        
        Returns:
            np.ndarray: Point cloud as numpy array
        """
        point_cloud = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            return point_cloud.get_data()
        return None
