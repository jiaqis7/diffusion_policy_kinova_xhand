#!/usr/bin/env python3
"""Test robot Redis connection and state availability"""
import redis
import time
import sys

REDIS_HOST = "192.168.1.15"
REDIS_PORT = 6379

print(f"Testing robot connection to {REDIS_HOST}:{REDIS_PORT}...\n")

# Try to connect
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password="iprl", socket_connect_timeout=5)
    r.ping()
    print("✓ Redis connection successful\n")
except Exception as e:
    print(f"✗ Cannot connect to Redis: {e}")
    sys.exit(1)

# Expected robot state keys (from your kinova_bimanual.py)
robot_keys = {
    "Bot 1 Position": "kinova::bot1::ee_pos",
    "Bot 1 Quaternion": "kinova::bot1::ee_quat_wxyz",
    "Bot 1 Gripper": "kinova::bot1::gripper_position",
    "Bot 1 Joints": "kinova::bot1::q",
    "Bot 2 Position": "kinova::bot2::ee_pos",
    "Bot 2 Quaternion": "kinova::bot2::ee_quat_wxyz",
    "Bot 2 Gripper": "kinova::bot2::gripper_position",
    "Bot 2 Joints": "kinova::bot2::q",
}

print("Checking robot state keys:")
print("-" * 60)
missing_keys = []
for name, key in robot_keys.items():
    val = r.get(key)
    if val:
        # Truncate long values
        val_str = val.decode('utf-8', errors='ignore') if isinstance(val, bytes) else str(val)
        if len(val_str) > 40:
            val_str = val_str[:40] + "..."
        print(f"  ✓ {name:20s}: {val_str}")
    else:
        print(f"  ✗ {name:20s}: NOT FOUND")
        missing_keys.append(key)

if missing_keys:
    print(f"\n⚠ Missing {len(missing_keys)} keys!")
    print("\nThis means the robot controller is NOT publishing state to Redis.")
    print("Please ensure the robot controller is running on the NUC.")
    sys.exit(1)

print("\n✓ All robot state keys found!")

# Monitor for updates
print("\n" + "="*60)
print("Monitoring state updates for 5 seconds...")
print("(Values should change if robot is moving/active)")
print("="*60)

test_keys = [
    "kinova::bot1::ee_pos",
    "kinova::bot2::ee_pos",
]

initial_values = {k: r.get(k) for k in test_keys}
time.sleep(1)

for i in range(5):
    print(f"\n[{i+1}s]")
    for key in test_keys:
        val = r.get(key)
        if val:
            val_str = val.decode('utf-8', errors='ignore') if isinstance(val, bytes) else str(val)
            changed = "CHANGED" if val != initial_values[key] else "same"
            print(f"  {key:30s}: {val_str[:50]} [{changed}]")
        else:
            print(f"  {key:30s}: None")
    time.sleep(1)

print("\n" + "="*60)
print("✓ Robot connection test complete!")
print("\nIf values are NOT changing, the robot controller may be:")
print("  1. Not running")
print("  2. Not connected to the robot")
print("  3. In an error state")