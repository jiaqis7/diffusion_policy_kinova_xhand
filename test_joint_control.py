#!/usr/bin/env python3
"""
Test script for joint control mode.
Sends hardcoded actions to the robot to verify the control pipeline.

Usage:
    python test_joint_control.py --robot_ip 192.168.1.15
"""

import sys
sys.path.insert(0, '/home/hshadow/wild_human')
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

import time
import numpy as np
import click
from diffusion_policy.real_world.utils.kinova_bimanual_xhand import KinovaBimanualXhand


@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.15', help='Robot IP address')
@click.option('--password', '-p', default='iprl', help='Redis password')
@click.option('--delay', '-d', default=2.0, type=float, help='Delay between actions (seconds)')
@click.option('--interpolation', '-i', default=0, type=int, help='Number of interpolation points between actions (0 = no interpolation)')
def main(robot_ip, password, delay, interpolation):
    """Test joint control by sending hardcoded actions."""

    print("=" * 60)
    print("Joint Control Test Script")
    print("=" * 60)
    print(f"Robot IP: {robot_ip}")
    print(f"Delay between actions: {delay}s")
    print(f"Interpolation points: {interpolation}")
    print()

    # Initialize robot connection
    print("Connecting to robot...")
    kinova = KinovaBimanualXhand(host=robot_ip, password=password)
    print("✓ Connected!")
    print()

    # Define 5 hardcoded actions (19D each: 7 arm joints + 12 xhand joints)
    # All angles are in RADIANS
    actions = [
        # Action 1: Home position for robot2
        np.array([
            # Arm joints (7D) - radians, will be converted to degrees
            -0.89, 1.5, 2.17, -1.61, -0.1, 0.73, 2.01,
            # Xhand joints (12D) - radians, sent as-is
            1.8, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float64),

        # Action 2
        np.array([
            -0.92, 1.56, 2.14, -1.56, -0.08, 0.76, 2.01,
            1.8, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float64),

        # Action 3
        np.array([
            -0.97, 1.63, 2.14, -1.55, -0.05, 0.79, 2.01,
            1.8, 0.0, 1.33, 0.0, 0.65, 1.9,
            0.2, 0.94, 1.07, 0.83, 1.13, 1.46
        ], dtype=np.float64),

        # Action 4
        np.array([
            -0.96, 1.89, 1.69, -1.6, 0.3, 0.66, 2.01,
            1.8, 0.0, 1.33, 0.0, 0.65, 1.9,
            0.2, 0.94, 1.07, 0.83, 1.13, 1.46
        ], dtype=np.float64),

        # Action 5
        np.array([
            -0.97, 1.9, 1.62, -1.62, 0.4, 0.59, 2.01,
            1.8, 0.0, 1.33, 0.0, 0.65, 1.9,
            0.2, 0.94, 1.07, 0.83, 1.13, 1.46
        ], dtype=np.float64),
    ]

    # Apply interpolation if requested
    if interpolation > 0:
        print(f"Applying linear interpolation with {interpolation} points between actions...")
        interpolated_actions = []
        for i in range(len(actions) - 1):
            # Add current action
            interpolated_actions.append(actions[i])
            # Add interpolated points between current and next action
            for j in range(1, interpolation + 1):
                alpha = j / (interpolation + 1)  # interpolation weight
                interp_action = (1 - alpha) * actions[i] + alpha * actions[i + 1]
                interpolated_actions.append(interp_action)
        # Add the last action
        interpolated_actions.append(actions[-1])
        actions = interpolated_actions
        print(f"  Total actions after interpolation: {len(actions)}")
        print()

    print("Hardcoded actions loaded:")
    print(f"  Number of actions: {len(actions)}")
    print(f"  Action dimension: {actions[0].shape[0]}D")
    print()

    # Get robot1's current position and keep it there
    print("Getting robot1's current position to keep it stationary...")
    q1_current, xhand1_current, q2_current, xhand2_current = kinova.get_joint_xhand()
    print(f"  Robot1 current arm joints: {q1_current}")
    print(f"  Robot1 current xhand joints: {xhand1_current}")
    print()

    # Send actions one by one
    try:
        for i, action in enumerate(actions, 1):
            print(f"Action {i}/{len(actions)}:")
            print(f"  Arm joints (rad):   {action[:7]}")
            print(f"  Arm joints (deg):   {np.rad2deg(action[:7])}")
            print(f"  Xhand joints (rad): {action[7:19]}")

            # Split action into arm and xhand
            q_des2 = action[:7]      # Robot2 arm (radians)
            xhand_des2 = action[7:19]  # Robot2 xhand (radians)

            # Send to robot2 only (keep robot1 at current position)
            print(f"  → Sending to robot2...")
            kinova.goto_joint_xhand(
                q_des1=q1_current,
                xhand_des1=xhand1_current,
                q_des2=q_des2,
                xhand_des2=xhand_des2
            )
            print(f"  ✓ Sent!")

            # Wait before next action
            if i < len(actions):
                print(f"  Waiting {delay}s before next action...")
                time.sleep(delay)
            else:
                # Wait 5 seconds after the last action
                print(f"  Waiting 5s after last action...")
                time.sleep(5.0)
            print()

        # Return to first action position
        print("=" * 60)
        print("Returning to first action position...")
        print("=" * 60)
        first_action = actions[0]
        q_des2 = first_action[:7]
        xhand_des2 = first_action[7:19]

        print(f"  Arm joints (rad):   {q_des2}")
        print(f"  Arm joints (deg):   {np.rad2deg(q_des2)}")
        print(f"  Xhand joints (rad): {xhand_des2}")
        print(f"  → Sending to robot2...")

        kinova.goto_joint_xhand(
            q_des1=q1_current,
            xhand_des1=xhand1_current,
            q_des2=q_des2,
            xhand_des2=xhand_des2
        )
        print(f"  ✓ Sent!")
        print()

        print("=" * 60)
        print("✓ All actions sent successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTest completed.")


if __name__ == '__main__':
    main()
