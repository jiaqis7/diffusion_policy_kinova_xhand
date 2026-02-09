# sim2real_adapter.py
"""
Adapters for translating between simulation policy format and real robot format.

Simulation Policy (DDDC Benchmark):
    - Observations: head_cam [T,3,256,256], agent_pos [T,19]
    - Actions: [H, 19] (single arm joint positions)

Real Robot (Kinova Bimanual + XHand):
    - Observations: agentview_image [T,H,W,3], robot2_joint_qpos [T,7], robot2_xhand_qpos [T,12]
    - Actions: [H, 19] or [H, 38]
      - 19D: single arm, env auto-pads robot1 with home position
      - 38D: bimanual (robot1[19] + robot2[19])

Note: Sim2RealActionAdapter is optional since RealXhandEnv.exec_actions()
      now accepts 19D actions directly and auto-pads robot1.
"""

import numpy as np
from typing import Dict, Optional


class Sim2RealObservationAdapter:
    """
    Adapts real robot observations to match simulation policy expectations.

    Real Robot Observations (joint mode):
        - agentview_image: [T, H, W, 3] RGB uint8
        - robot2_joint_qpos: [T, 7] arm joint positions (radians)
        - robot2_xhand_qpos: [T, 12] hand joint positions (radians)
        - timestamp: [T]

    Sim Policy Expectations:
        - head_cam: [T, H, W, 3] RGB (will be transformed by get_real_obs_dict)
        - agent_pos: [T, 19] concatenated [arm_joints(7), hand_joints(12)]
    """

    def __init__(self,
                 sim_image_key: str = "head_cam",
                 sim_proprio_key: str = "agent_pos",
                 real_image_key: str = "agentview_image",
                 real_arm_key: str = "robot2_joint_qpos",
                 real_hand_key: str = "robot2_xhand_qpos"):
        """
        Args:
            sim_image_key: Key name expected by sim policy for image observation
            sim_proprio_key: Key name expected by sim policy for proprioception
            real_image_key: Key name from real robot environment for image
            real_arm_key: Key name from real robot environment for arm joints
            real_hand_key: Key name from real robot environment for hand joints
        """
        self.sim_image_key = sim_image_key
        self.sim_proprio_key = sim_proprio_key
        self.real_image_key = real_image_key
        self.real_arm_key = real_arm_key
        self.real_hand_key = real_hand_key

    def adapt_observation(self, real_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform real robot observations to sim policy format.

        Args:
            real_obs: Dictionary from env.get_obs() with real robot keys

        Returns:
            Dictionary with sim policy keys (head_cam, agent_pos)
        """
        adapted = {}

        # Map image: agentview_image -> head_cam
        # Keep as THWC format - get_real_obs_dict will handle THWC->TCHW conversion
        if self.real_image_key in real_obs:
            adapted[self.sim_image_key] = real_obs[self.real_image_key]
        else:
            raise KeyError(f"Real observation missing image key: {self.real_image_key}")

        # Concatenate proprioception: robot2_joint_qpos + robot2_xhand_qpos -> agent_pos
        if self.real_arm_key in real_obs and self.real_hand_key in real_obs:
            arm_joints = real_obs[self.real_arm_key]  # [T, 7]
            hand_joints = real_obs[self.real_hand_key]  # [T, 12]
            adapted[self.sim_proprio_key] = np.concatenate(
                [arm_joints, hand_joints], axis=-1
            )  # [T, 19]
        else:
            missing = []
            if self.real_arm_key not in real_obs:
                missing.append(self.real_arm_key)
            if self.real_hand_key not in real_obs:
                missing.append(self.real_hand_key)
            raise KeyError(f"Real observation missing keys: {missing}")

        # Pass through timestamp if present
        if 'timestamp' in real_obs:
            adapted['timestamp'] = real_obs['timestamp']

        return adapted


class Sim2RealActionAdapter:
    """
    Adapts sim policy actions to real robot action format.

    Sim Policy Output:
        - action: [H, 19] single arm joint positions (7 arm + 12 hand)

    Real Robot Expects:
        - action: [H, 38] bimanual: [robot1(19), robot2(19)]

    Robot1 is kept stationary at home position while robot2 executes policy actions.
    """

    def __init__(self,
                 robot1_home_joints: Optional[np.ndarray] = None,
                 robot1_home_xhand: Optional[np.ndarray] = None,
                 joint_limits_min: Optional[np.ndarray] = None,
                 joint_limits_max: Optional[np.ndarray] = None,
                 xhand_limits_min: float = 0.0,
                 xhand_limits_max: float = 1.0):
        """
        Args:
            robot1_home_joints: [7] home joint positions for robot1 arm (radians)
            robot1_home_xhand: [12] home positions for robot1 hand
            joint_limits_min: [7] minimum joint angles for safety (radians)
            joint_limits_max: [7] maximum joint angles for safety (radians)
            xhand_limits_min: Minimum xhand position value
            xhand_limits_max: Maximum xhand position value
        """
        # Default robot1 home position (radians)
        # Corresponds to degrees: {121.0, 52.0, -170.0, -114.0, -58.0, -62.0, -179.0}
        if robot1_home_joints is None:
            robot1_home_joints = np.array([
                2.1118, 0.9076, -2.9671, -1.9897, -1.0123, -1.0821, -3.1241
            ], dtype=np.float64)

        if robot1_home_xhand is None:
            robot1_home_xhand = np.zeros(12, dtype=np.float64)

        self.robot1_home = np.concatenate([robot1_home_joints, robot1_home_xhand])

        # Kinova Gen3 joint limits (radians)
        if joint_limits_min is None:
            joint_limits_min = np.array(
                [-3.14, -2.41, -3.14, -2.66, -3.14, -2.23, -3.14],
                dtype=np.float64
            )
        if joint_limits_max is None:
            joint_limits_max = np.array(
                [3.14, 2.41, 3.14, 2.66, 3.14, 2.23, 3.14],
                dtype=np.float64
            )

        self.joint_limits_min = joint_limits_min
        self.joint_limits_max = joint_limits_max
        self.xhand_limits_min = xhand_limits_min
        self.xhand_limits_max = xhand_limits_max

    def adapt_action(self, sim_action: np.ndarray) -> np.ndarray:
        """
        Transform sim policy action to real robot format.

        Args:
            sim_action: [H, 19] single arm action from policy
                        Format: [arm_joints(7), hand_joints(12)]

        Returns:
            [H, 38] bimanual action for real robot
            Format: [robot1_home(19), robot2_action(19)]
        """
        if sim_action.ndim == 1:
            sim_action = sim_action[np.newaxis, :]

        H = sim_action.shape[0]
        assert sim_action.shape[-1] == 19, \
            f"Expected 19D sim action, got {sim_action.shape[-1]}D"

        # Extract and clip arm joint angles to safety limits
        robot2_arm = sim_action[:, :7].copy()
        robot2_arm = np.clip(robot2_arm, self.joint_limits_min, self.joint_limits_max)

        # Extract and clip hand positions
        robot2_hand = sim_action[:, 7:19].copy()
        robot2_hand = np.clip(robot2_hand, self.xhand_limits_min, self.xhand_limits_max)

        # Combine clipped robot2 action
        robot2_action = np.concatenate([robot2_arm, robot2_hand], axis=-1)  # [H, 19]

        # Create robot1 action (stationary at home position)
        robot1_action = np.tile(self.robot1_home, (H, 1))  # [H, 19]

        # Concatenate: [robot1(19), robot2(19)] = 38D
        full_action = np.concatenate([robot1_action, robot2_action], axis=-1)

        return full_action

    def get_robot1_home(self) -> np.ndarray:
        """Returns the robot1 home position [19D]."""
        return self.robot1_home.copy()

    def get_joint_limits(self):
        """Returns (min, max) joint limits for arm."""
        return self.joint_limits_min.copy(), self.joint_limits_max.copy()


def get_sim2real_shape_meta() -> dict:
    """
    Returns shape_meta matching the simulation policy expectations.
    Use this to override checkpoint's shape_meta for observation transformation.
    """
    return {
        'obs': {
            'head_cam': {
                'shape': [3, 256, 256],
                'type': 'rgb'
            },
            'agent_pos': {
                'shape': [19],
                'type': 'low_dim'
            }
        },
        'action': {
            'shape': [19]
        }
    }
