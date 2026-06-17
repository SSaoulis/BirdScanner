#!/usr/bin/env python3
"""
End-to-end script to:
1. SSH into Raspberry Pi
2. Run test_camera.py to capture a photo
3. Copy the camera_view directory back to local machine
4. Extract and display the most recent image
"""

import tarfile
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
from PIL import Image
import sys
import paramiko
from io import BytesIO

# Configuration
PI_HOST = "192.168.1.31"
PI_USER = "stefan"
PI_PASSWORD = "123"
PI_REPO_PATH = "/home/stefan/git/BirdScanner"
PI_PICTURES_PATH = "/home/stefan/Pictures"
LOCAL_EXTRACT_DIR = "./camera_view"
TAR_FILE = "camera_view.tar"


def run_ssh_command(
    host: str, user: str, password: str, command: str
) -> tuple[bool, str]:
    """
    Execute a command on the Pi via SSH using paramiko.

    Args:
        host: IP address or hostname
        user: SSH username
        password: SSH password
        command: Command to execute

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=user, password=password, timeout=30)

        print(f"Executing: {command}")
        stdin, stdout, stderr = client.exec_command(command, timeout=60)
        output = stdout.read().decode()
        error = stderr.read().decode()

        client.close()

        if output:
            print(output)

        if error:
            # libcamera logs go to stderr, but that's not necessarily an error
            # Only fail if there's an actual error message
            print(f"stderr output: {error}")
            if "error" in error.lower() or "failed" in error.lower():
                return False, error

        return True, output
    except Exception as e:
        print(f"SSH Error: {e}")
        return False, str(e)


def take_photo_on_pi(host: str, user: str, password: str) -> bool:
    """
    SSH into Pi and run test_camera.py
    """
    print("\n=== Taking photo on Raspberry Pi ===")

    command = f"cd {PI_REPO_PATH} && source venv/bin/activate && cd utils && python test_camera.py"

    success, output = run_ssh_command(host, user, password, command)
    return success


def copy_camera_view_from_pi(host: str, user: str, password: str) -> bool:
    """
    Copy camera_view directory from Pi using SFTP
    """
    print("\n=== Copying camera_view directory from Pi ===")

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=user, password=password, timeout=10)

        # Use SSH to create tar stream
        stdin, stdout, stderr = client.exec_command(
            f"tar -C {PI_PICTURES_PATH} -cf - camera_view"
        )

        # Read tar data from stdout
        tar_data = stdout.read()
        client.close()

        # Write to local file
        with open(TAR_FILE, "wb") as f:
            f.write(tar_data)

        print(f"Successfully saved tar file: {TAR_FILE}")
        return True
    except Exception as e:
        print(f"Error copying directory: {e}")
        return False


def extract_tar_file(tar_path: str, extract_dir: str = ".") -> bool:
    """
    Extract tar file locally
    """
    print(f"\n=== Extracting {tar_path} ===")

    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)
        print(f"Successfully extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting tar file: {e}")
        return False


def find_most_recent_image(directory: str) -> Optional[str]:
    """
    Find the most recently modified image file in the directory
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                filepath = os.path.join(root, file)
                image_files.append((filepath, os.path.getmtime(filepath)))

    if not image_files:
        print(f"No images found in {directory}")
        return None

    # Sort by modification time, most recent first
    image_files.sort(key=lambda x: x[1], reverse=True)
    most_recent = image_files[0][0]

    print(f"Most recent image: {most_recent}")
    return most_recent


def display_image(image_path: str):
    """
    Display image using matplotlib
    """
    print(f"\n=== Displaying image ===")

    try:
        img = Image.open(image_path)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Most Recent Capture: {Path(image_path).name}")
        plt.tight_layout()
        plt.show()

        print("Image displayed successfully")
    except Exception as e:
        print(f"Error displaying image: {e}")


def main():
    """Main entry point"""
    print("=== BirdFinder End-to-End Camera Test ===\n")

    # Step 1: Take photo on Pi
    if not take_photo_on_pi(PI_HOST, PI_USER, PI_PASSWORD):
        print("Failed to take photo on Pi")
        sys.exit(1)

    # Step 2: Copy directory from Pi
    if not copy_camera_view_from_pi(PI_HOST, PI_USER, PI_PASSWORD):
        print("Failed to copy directory from Pi")
        sys.exit(1)

    # Step 3: Extract tar file
    if not extract_tar_file(TAR_FILE):
        print("Failed to extract tar file")
        sys.exit(1)

    # Step 4: Find and display most recent image
    image_path = find_most_recent_image(LOCAL_EXTRACT_DIR)
    if image_path:
        display_image(image_path)
    else:
        print("No images found to display")
        sys.exit(1)

    print("\n=== End-to-end test completed successfully ===")


if __name__ == "__main__":
    main()
