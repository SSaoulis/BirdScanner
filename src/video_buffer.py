"""Circular video frame buffer for storing recent frames from camera stream."""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import numpy as np


@dataclass
class BufferedFrame:
    """A single frame with timestamp information."""
    
    frame: np.ndarray  # The actual frame data
    timestamp: float   # Timestamp in seconds (relative to buffer start)
    datetime: datetime # Absolute datetime when frame was captured


class VideoBuffer:
    """Circular buffer for storing recent video frames from the camera stream.
    
    Maintains a fixed-size buffer of frames with their timestamps. Allows
    retrieval of frame sequences around a specific time point.
    """
    
    def __init__(self, max_buffer_seconds: float, fps: int = 30):
        """Initialize the video buffer.
        
        Args:
            max_buffer_seconds: Maximum number of seconds of video to keep in buffer.
            fps: Expected frames per second from camera.
        """
        self.fps = fps
        self.max_buffer_seconds = max_buffer_seconds
        self.max_frames = int(max_buffer_seconds * fps)
        
        self._buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._frame_count = 0
    
    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer.
        
        Args:
            frame: Frame as numpy array (typically RGB or BGR).
        """
        with self._lock:
            if self._start_time is None:
                self._start_time = 0.0
                self._frame_count = 0
            
            relative_time = self._frame_count / self.fps
            buffered_frame = BufferedFrame(
                frame=frame.copy(),  # Store a copy to avoid external modifications
                timestamp=relative_time,
                datetime=datetime.now()
            )
            self._buffer.append(buffered_frame)
            self._frame_count += 1
    
    def get_frames_around_time(
        self, 
        relative_timestamp: float,
        seconds_before: float = 3.0,
        seconds_after: float = 3.0,
    ) -> list[BufferedFrame]:
        """Get frames from a time window around a specific timestamp.
        
        Args:
            relative_timestamp: Timestamp (in seconds) of the detection/event.
            seconds_before: Seconds of video to include before the event.
            seconds_after: Seconds of video to include after the event.
            
        Returns:
            List of BufferedFrame objects in the requested time window.
            May contain fewer frames if the buffer doesn't have the full range.
        """
        with self._lock:
            time_start = relative_timestamp - seconds_before
            time_end = relative_timestamp + seconds_after
            
            matching_frames = [
                bf for bf in self._buffer
                if time_start <= bf.timestamp <= time_end
            ]
            return matching_frames
    
    def get_all_frames(self) -> list[BufferedFrame]:
        """Get all frames currently in the buffer.
        
        Returns:
            List of all BufferedFrame objects.
        """
        with self._lock:
            return list(self._buffer)
    
    def get_buffer_size_seconds(self) -> float:
        """Get the current amount of video buffered (in seconds).
        
        Returns:
            Duration of video currently in buffer.
        """
        with self._lock:
            if not self._buffer:
                return 0.0
            first_ts = self._buffer[0].timestamp
            last_ts = self._buffer[-1].timestamp
            return last_ts - first_ts
    
    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._start_time = None
            self._frame_count = 0
