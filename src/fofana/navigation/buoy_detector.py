"""Buoy detection and tracking module."""
import cv2
import numpy as np
from typing import List, Tuple, Optional

class BuoyDetector:
    def __init__(self):
        """Initialize buoy detector."""
        # Color ranges in HSV
        self.red_ranges = [
            ((0, 100, 100), (10, 255, 255)),    # Lower red
            ((160, 100, 100), (180, 255, 255))  # Upper red
        ]
        self.green_range = ((40, 100, 100), (80, 255, 255))
        self.yellow_range = ((20, 100, 100), (40, 255, 255))
        
    def detect_buoys(self, frame: np.ndarray) -> dict:
        """Detect buoys in the frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            dict: Detected buoys with their positions and colors
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        detected_buoys = {
            'red': self._detect_color(hsv, self.red_ranges, multiple_ranges=True),
            'green': self._detect_color(hsv, [self.green_range]),
            'yellow': self._detect_color(hsv, [self.yellow_range])
        }
        return detected_buoys
        
    def _detect_color(self, hsv: np.ndarray, 
                     color_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
                     multiple_ranges: bool = False) -> List[Tuple[int, int, int]]:
        """Detect objects of specific color.
        
        Args:
            hsv: HSV image
            color_ranges: List of HSV color ranges
            multiple_ranges: Whether to combine multiple ranges
            
        Returns:
            List of (x, y, radius) for detected objects
        """
        mask = None
        for lower, upper in color_ranges:
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if mask is None:
                mask = color_mask
            elif multiple_ranges:
                mask = cv2.bitwise_or(mask, color_mask)
                
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                (x, y), radius = cv2.minEnclosingCircle(contour)
                objects.append((int(x), int(y), int(radius)))
                
        return objects
