"""Buoy tracking module for continuous detection and state estimation."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .buoy_detector import BuoyDetector

@dataclass
class TrackedBuoy:
    """Tracked buoy state information."""
    position: Tuple[int, int]  # x, y in pixels
    radius: int
    color: str
    velocity: Tuple[float, float]  # dx, dy in pixels/frame
    last_seen: int  # frame counter
    confidence: float

class BuoyTracker:
    def __init__(self, max_distance: int = 50):
        """Initialize buoy tracker.
        
        Args:
            max_distance: Maximum pixel distance for track association
        """
        self.detector = BuoyDetector()
        self.tracked_buoys: List[TrackedBuoy] = []
        self.max_distance = max_distance
        self.frame_counter = 0
        
    def update(self, frame: np.ndarray) -> List[TrackedBuoy]:
        """Update tracked buoys with new frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            List of currently tracked buoys
        """
        self.frame_counter += 1
        detected_buoys = self.detector.detect_buoys(frame)
        
        # Update existing tracks
        updated_tracks = []
        for color, detections in detected_buoys.items():
            for x, y, radius in detections:
                matched = False
                # Try to match with existing tracks
                for track in self.tracked_buoys:
                    if track.color == color:
                        dist = np.sqrt((track.position[0] - x)**2 + 
                                     (track.position[1] - y)**2)
                        if dist < self.max_distance:
                            # Update track
                            dx = x - track.position[0]
                            dy = y - track.position[1]
                            track.velocity = (dx, dy)
                            track.position = (x, y)
                            track.radius = radius
                            track.last_seen = self.frame_counter
                            track.confidence = min(1.0, track.confidence + 0.1)
                            updated_tracks.append(track)
                            matched = True
                            break
                            
                if not matched:
                    # Create new track
                    new_track = TrackedBuoy(
                        position=(x, y),
                        radius=radius,
                        color=color,
                        velocity=(0, 0),
                        last_seen=self.frame_counter,
                        confidence=0.5
                    )
                    updated_tracks.append(new_track)
        
        # Remove old tracks
        self.tracked_buoys = [track for track in updated_tracks 
                            if self.frame_counter - track.last_seen < 30]
        
        return self.tracked_buoys
        
    def predict_buoy_positions(self, frames_ahead: int = 5) -> List[TrackedBuoy]:
        """Predict future buoy positions based on current tracks.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            List of predicted buoy positions
        """
        predictions = []
        for track in self.tracked_buoys:
            if track.confidence > 0.7:  # Only predict high confidence tracks
                pred_x = track.position[0] + track.velocity[0] * frames_ahead
                pred_y = track.position[1] + track.velocity[1] * frames_ahead
                predictions.append(TrackedBuoy(
                    position=(int(pred_x), int(pred_y)),
                    radius=track.radius,
                    color=track.color,
                    velocity=track.velocity,
                    last_seen=track.last_seen,
                    confidence=track.confidence * 0.9  # Reduce confidence for predictions
                ))
        return predictions
