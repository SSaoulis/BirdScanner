"""Video file writing utilities for saving bird detection clips."""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import cv2
import numpy as np


class VideoWriter:
    """Handles writing video frames to MP4 files with optional overlays."""
    
    def __init__(
        self,
        output_path: str,
        frame_width: int,
        frame_height: int,
        fps: int = 30,
        codec: str = "mp4v",
    ):
        """Initialize the video writer.
        
        Args:
            output_path: Path where the video file will be saved.
            frame_width: Width of output video frames.
            frame_height: Height of output video frames.
            fps: Frames per second for the output video.
            codec: FourCC codec string (default 'mp4v' for H.264 in MP4).
        """
        self.output_path = output_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.codec = codec
        
        # Create parent directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the cv2.VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video file.
        
        Args:
            frame: Frame as numpy array. Should match the initialized dimensions.
                  Expected to be BGR format (as used by OpenCV).
        """
        # Ensure frame has correct dimensions
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        self.writer.write(frame)
    
    def release(self) -> None:
        """Close the video file and release resources."""
        if self.writer and self.writer.isOpened():
            self.writer.release()
    
    def __del__(self):
        """Ensure video writer is released when object is destroyed."""
        self.release()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def save_detection_video(
    output_dir: str,
    species: str,
    frames: list[np.ndarray],
    fps: int = 30,
    include_labeled: bool = True,
) -> tuple[str, Optional[str]]:
    """Save buffered video frames to files.
    
    Creates timestamped video files for detections. Can save both raw and
    labeled (with bounding boxes) versions.
    
    Args:
        output_dir: Directory where videos will be saved.
        species: Bird species name for organizing output.
        frames: List of frames to write (as BGR numpy arrays).
        fps: Frames per second for video.
        include_labeled: Whether to save a labeled version with boxes/text.
            (This will be used when frames have drawn boxes on them.)
        
    Returns:
        Tuple of (raw_video_path, labeled_video_path).
        labeled_video_path is None if include_labeled=False.
    """
    if not frames:
        return None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    species_dir = os.path.join(output_dir, species)
    os.makedirs(species_dir, exist_ok=True)
    
    # Get frame dimensions from first frame
    h, w = frames[0].shape[:2]
    
    # Save raw video
    raw_video_path = os.path.join(species_dir, f"{timestamp}_raw.mp4")
    with VideoWriter(raw_video_path, w, h, fps=fps) as writer:
        for frame in frames:
            writer.write_frame(frame)
    
    return raw_video_path, None


def add_boxes_to_frames(
    frames: list[np.ndarray],
    detections: list[dict],
) -> list[np.ndarray]:
    """Add detection boxes and labels to frames.
    
    Args:
        frames: List of frames (BGR).
        detections: List of detection dicts with keys: box, label, confidence, species, species_confidence.
        
    Returns:
        List of frames with boxes drawn on them.
    """
    labeled_frames = []
    
    for frame in frames:
        labeled_frame = frame.copy()
        
        # Draw all detections
        for det in detections:
            x, y, w, h = det.get('box', (0, 0, 0, 0))
            
            # Draw box
            cv2.rectangle(labeled_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Build label text
            label = f"{det.get('label', 'Unknown')} ({det.get('confidence', 0):.2f})"
            if det.get('species'):
                label += f" - {det['species']} ({det.get('species_confidence', 0):.2f})"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x + 5
            text_y = y + 15
            
            overlay = labeled_frame.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_height),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.30, labeled_frame, 0.70, 0, labeled_frame)
            
            # Draw label text
            cv2.putText(
                labeled_frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        
        labeled_frames.append(labeled_frame)
    
    return labeled_frames
