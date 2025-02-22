"""Shape detection for docking task."""
import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ShapeDetector:
    """Detects and classifies shapes for docking task."""
    
    def __init__(self):
        """Initialize shape detector."""
        # HSV color ranges
        self.colors = {
            'red': ([0, 50, 50], [10, 255, 255]),    # Red wraps around hue
            'green': ([35, 50, 50], [85, 255, 255]), # Green range
            'blue': ([100, 50, 50], [140, 255, 255]) # Blue range
        }
        
        # Shape detection parameters
        self.shape_params = {
            'circle': {
                'vertices': (8, 12),      # Expected vertex count range
                'area_ratio': (0.7, 1.0), # Area ratio range (filled/bounding)
                'aspect_ratio': (0.9, 1.1) # Width/height ratio range
            },
            'triangle': {
                'vertices': (3, 4),       # Allow some noise
                'area_ratio': (0.4, 0.6),
                'aspect_ratio': (0.8, 1.2)
            },
            'square': {
                'vertices': (4, 6),
                'area_ratio': (0.8, 1.0),
                'aspect_ratio': (0.9, 1.1)
            },
            'plus': {
                'vertices': (8, 16),      # Complex shape
                'area_ratio': (0.4, 0.9), # Less filled than circle/square
                'aspect_ratio': (0.9, 1.1)
            }
        }
        
    def detect_shapes(self, frame: torch.Tensor) -> List[Dict[str, any]]:
        """Detect shapes in frame."""
        # Convert tensor to numpy array
        if frame.is_cuda:
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame.numpy()
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV)
        
        detected = []
        
        # Detect each color
        for color_name, (lower, upper) in self.colors.items():
            # Create color mask
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Special case for red (wraps around hue)
            if color_name == 'red':
                lower2 = np.array([170, 50, 50])
                upper2 = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 100:  # Minimum area threshold
                    continue
                    
                # Get shape properties
                shape_info = self._classify_shape(contour)
                if not shape_info:
                    continue
                    
                # Calculate confidence based on shape metrics
                confidence = self._calculate_confidence(shape_info)
                
                # Get position and dimensions
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                # Calculate distance based on apparent size
                # Using inverse relationship: distance ∝ 1/size
                distance = 100 / np.sqrt(area)  # Arbitrary scale factor
                
                # Create detection info
                detection = {
                    'type': shape_info['type'],
                    'color': color_name,
                    'position': center,
                    'confidence': confidence,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'distance': distance,
                    'dimensions': {
                        'width': w/100,  # Convert to meters
                        'height': h/100
                    }
                }
                detected.append(detection)
                
        return detected
        
    def _classify_shape(self, contour: np.ndarray) -> Optional[Dict[str, any]]:
        """Classify shape type from contour."""
        # Get shape metrics
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        
        # Calculate area ratio
        area = cv2.contourArea(contour)
        bbox_area = w * h
        area_ratio = area / bbox_area if bbox_area > 0 else 0
        
        logger.debug(f"Contour vertices: {vertices}")
        
        # Check each shape type
        for shape_type, params in self.shape_params.items():
            # Check vertex count
            if not (params['vertices'][0] <= vertices <= params['vertices'][1]):
                continue
                
            # Check area ratio
            if not (params['area_ratio'][0] <= area_ratio <= params['area_ratio'][1]):
                logger.debug(f"{shape_type} shape rejected: area ratio {area_ratio:.2f} outside range {params['area_ratio']}")
                continue
                
            # Check aspect ratio
            if not (params['aspect_ratio'][0] <= aspect_ratio <= params['aspect_ratio'][1]):
                continue
                
            # Additional check for plus sign
            if shape_type == 'plus':
                if not self._verify_plus_shape(contour):
                    continue
                logger.debug("Plus shape detected")
                
            # Return shape info
            return {
                'type': shape_type,
                'vertices': vertices,
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio
            }
            
        return None
        
    def _verify_plus_shape(self, contour: np.ndarray) -> bool:
        """Additional verification for plus sign shape."""
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create mask of contour
        mask = np.zeros((h, w), dtype=np.uint8)
        contour_shifted = contour - [x, y]
        cv2.drawContours(mask, [contour_shifted], 0, 255, -1)
        
        # Check for horizontal and vertical bars
        h_center = h // 2
        v_center = w // 2
        
        # Get horizontal and vertical profiles
        h_profile = mask[h_center, :]
        v_profile = mask[:, v_center]
        
        # Calculate metrics
        aspect_ratio = float(w)/h
        area_ratio = cv2.countNonZero(mask) / (w * h)
        center_distance = abs(v_center - h_center) / max(w, h)
        
        # Log metrics
        logger.debug("Plus shape detected - metrics:")
        logger.debug(f"- Aspect ratio: {aspect_ratio:.2f}")
        logger.debug(f"- Area ratio: {area_ratio:.2f}")
        logger.debug(f"- Vertex count: {len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True))}")
        logger.debug(f"- Center distance: {center_distance:.2f}")
        
        # Verify metrics
        return (0.8 < aspect_ratio < 1.2 and  # Nearly square
                0.3 < area_ratio < 0.7 and    # Cross shape fills ~50%
                center_distance < 0.2)         # Centered cross
                
    def _calculate_confidence(self, shape_info: Dict[str, any]) -> float:
        """Calculate confidence score for shape detection."""
        confidence = 100.0  # Start with maximum confidence
        
        # Penalize based on area ratio deviation from ideal
        ideal_area_ratio = (self.shape_params[shape_info['type']]['area_ratio'][0] + 
                          self.shape_params[shape_info['type']]['area_ratio'][1]) / 2
        area_ratio_dev = abs(shape_info['area_ratio'] - ideal_area_ratio)
        confidence -= area_ratio_dev * 50  # Up to 50% penalty
        
        # Penalize based on aspect ratio deviation from 1.0
        aspect_ratio_dev = abs(1.0 - shape_info['aspect_ratio'])
        confidence -= aspect_ratio_dev * 30  # Up to 30% penalty
        
        # Ensure confidence is in [0, 100]
        return float(np.clip(confidence, 0, 100))
