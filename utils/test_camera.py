#!/usr/bin/env python3
"""
Script to take a photo using picamera2 and save it locally.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2 is not installed. Install it with: pip install picamera2")
    exit(1)


def take_photo(output_dir: str = ".", filename: str = None) -> str:
    """
    Take a photo using picamera2 and save it to the specified directory.
    
    Args:
        output_dir: Directory to save the photo (default: current directory)
        filename: Optional custom filename. If not provided, uses timestamp.
    
    Returns:
        Path to the saved photo
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
    
    filepath = os.path.join(output_dir, filename)
    
    # Initialize camera
    print("Initializing camera...")
    camera = Picamera2()
    
    # Configure camera
    config = camera.create_preview_configuration()
    camera.configure(config)
    
    # Start camera
    camera.start()
    
    # Give sensor time to stabilize
    import time
    time.sleep(2)
    
    # Capture photo
    print(f"Taking photo and saving to {filepath}...")
    camera.capture_file(filepath)
    
    # Stop camera
    camera.stop()
    camera.close()
    
    print(f"Photo saved successfully: {filepath}")
    return filepath


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Take a photo using picamera2")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/stefan/Pictures/camera_view/.",
        help="Directory to save the photo (default: current directory)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Custom filename for the photo (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    take_photo(output_dir=args.output_dir, filename=args.filename)


if __name__ == "__main__":
    main()
