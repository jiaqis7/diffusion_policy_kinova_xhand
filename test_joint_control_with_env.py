#!/usr/bin/env python3
"""
Test script for joint control mode using the full environment pipeline.
This simulates how the policy sends actions through the environment.

Usage:
    python test_joint_control_with_env.py --robot_ip 192.168.1.15
"""

import sys
sys.path.insert(0, '/home/hshadow/wild_human')
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import click
from diffusion_policy.real_world.utils.kinova_bimanual_with_xhand import KinovaBimanualWithXhand


@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.15', help='Robot IP address')
@click.option('--password', '-p', default='iprl', help='Redis password')
@click.option('--frequency', '-f', default=30, type=int, help='Control frequency (Hz)')
@click.option('--delay', '-d', default=2.0, type=float, help='Delay between actions (seconds)')
def main(robot_ip, password, frequency, delay):
    """Test joint control using the environment pipeline."""

    print("=" * 60)
    print("Joint Control Test with Environment Pipeline")
    print("=" * 60)
    print(f"Robot IP: {robot_ip}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Delay between actions: {delay}s")
    print()

    # Define 5 hardcoded actions (19D each: 7 arm joints + 12 xhand joints)
    # All angles are in RADIANS
    actions = [
        # Action 1: Home position for robot2
        np.array([
            -1.5708, 0.8901, 2.1817, -1.9199, 0.8727, -1.0472, 0.2793,  # Arm
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Xhand
        ], dtype=np.float64),

        # Action 2: Slightly move arm, close hand
        np.array([
            -1.5708, 0.9, 2.2, -1.92, 0.88, -1.05, 0.28,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        ], dtype=np.float64),

        # Action 3: Move arm more
        np.array([
            -1.57, 0.95, 2.25, -1.93, 0.90, -1.04, 0.30,
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
        ], dtype=np.float64),

        # Action 4: Different configuration
        np.array([
            -1.55, 0.88, 2.15, -1.91, 0.85, -1.06, 0.27,
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3
        ], dtype=np.float64),

        # Action 5: Return to home
        np.array([
            -1.5708, 0.8901, 2.1817, -1.9199, 0.8727, -1.0472, 0.2793,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float64),
    ]

    # Robot1 stays at home (19D)
    robot1_home = np.array([
        2.1118, 0.9076, -2.9671, -1.9897, -1.0123, -1.0821, -3.1241,  # Arm
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Xhand
    ], dtype=np.float64)

    print("Hardcoded actions loaded:")
    print(f"  Number of actions: {len(actions)}")
    print(f"  Action dimension: {actions[0].shape[0]}D (19D per robot)")
    print()

    # Create robot controller with shared memory
    with SharedMemoryManager() as shm_manager:
        print("Creating robot controller...")
        robot = KinovaBimanualWithXhand(
            shm_manager=shm_manager,
            frequency=frequency,
            launch_timeout=30,
            verbose=True,
            get_max_k=128,
            host=robot_ip,
            password=password,
            control_mode='joint'  # Use joint control mode
        )

        print("Starting robot controller...")
        robot.start(wait=True)
        print("✓ Robot controller ready!")
        print()

        time.sleep(1.0)  # Let it stabilize

        try:
            # Send actions one by one
            for i, action in enumerate(actions, 1):
                print(f"Action {i}/{len(actions)}:")
                print(f"  Arm joints (rad):   {action[:7]}")
                print(f"  Arm joints (deg):   {np.rad2deg(action[:7])}")
                print(f"  Xhand joints (rad): {action[7:19]}")

                # Combine robot1 home + robot2 action = 38D
                full_action = np.concatenate([robot1_home, action])
                assert full_action.shape[0] == 38, f"Expected 38D, got {full_action.shape[0]}D"

                # Schedule waypoint with timestamp
                target_time = time.time() + 0.1  # Execute 100ms in the future
                print(f"  → Scheduling waypoint (target time: {target_time:.3f})")
                robot.schedule_waypoint(
                    pose=full_action,
                    target_time=target_time
                )
                print(f"  ✓ Scheduled!")

                # Read current state
                state = robot.get_state()
                if state is not None:
                    print(f"  Current state:")
                    if 'robot2_joint_qpos' in state:
                        print(f"    Robot2 arm (rad): {state['robot2_joint_qpos']}")
                        print(f"    Robot2 arm (deg): {np.rad2deg(state['robot2_joint_qpos'])}")
                    if 'robot2_xhand_qpos' in state:
                        print(f"    Robot2 xhand:     {state['robot2_xhand_qpos']}")

                # Wait before next action
                if i < len(actions):
                    print(f"  Waiting {delay}s before next action...")
                    time.sleep(delay)
                print()

            print("=" * 60)
            print("✓ All actions sent successfully!")
            print("=" * 60)
            print()
            print("Waiting 2 seconds before stopping...")
            time.sleep(2.0)

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\nStopping robot controller...")
            robot.stop(wait=True)
            print("✓ Stopped!")


if __name__ == '__main__':
    main()
