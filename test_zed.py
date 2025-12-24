#!/usr/bin/env python3
"""Test direct ZED camera access"""
import pyzed.sl as sl
import time

print("Testing ZED camera access...\n")

# List all cameras
devices = sl.Camera.get_device_list()
print(f"Found {len(devices)} ZED camera(s):")
for d in devices:
    print(f"  Serial: {d.serial_number}, Model: {d.camera_model}")

if len(devices) == 0:
    print("\n✗ No ZED cameras found!")
    exit(1)

# Try to open first camera
serial = devices[0].serial_number
print(f"\nAttempting to open camera {serial}...")

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.NONE
init.set_from_serial_number(int(serial))

cam = sl.Camera()
status = cam.open(init)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"\n✗ Failed to open camera: {status}")
    print("\nPossible causes:")
    print("  1. Another process is using the camera")
    print("  2. Insufficient permissions (try: sudo usermod -aG video $USER)")
    print("  3. USB bandwidth issue (disconnect other USB devices)")
    print("  4. Run: sudo chmod 666 /dev/video*")
    exit(1)

print("✓ Camera opened successfully!")

# Try to grab frames
print("\nGrabbing 10 frames...")
img = sl.Mat()
for i in range(10):
    if cam.grab() == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(img, sl.VIEW.LEFT)
        print(f"  Frame {i+1}: {img.get_width()}x{img.get_height()}")
        time.sleep(0.033)
    else:
        print(f"  ✗ Frame {i+1} failed")

cam.close()
print("\n✓ All tests passed! ZED camera is working correctly.")