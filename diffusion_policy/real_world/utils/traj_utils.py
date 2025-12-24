#traj_utils

import ruckig
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation, Slerp

"""
Functions for computing linear and angular trajectories
"""


def interpolate_rotations(rot_matrix1: np.ndarray, rot_matrix2: np.ndarray, factor: float) -> np.ndarray:
    rot1 = Rotation.from_matrix(rot_matrix1)
    rot2 = Rotation.from_matrix(rot_matrix2)
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    slerp = Slerp([0,1], Rotation.from_quat([quat1, quat2]))
    intermediate_rotation = slerp([factor])[0]
    intermediate_rotation_matrix = intermediate_rotation.as_matrix()
    return intermediate_rotation_matrix

def saturate_velocity(velocity, max_velocity):
    """
    Saturate the velocity vector to ensure its magnitude does not exceed max_velocity.

    Parameters:
    velocity (numpy array): The velocity vector to be saturated.
    max_velocity (float): The maximum allowable magnitude for the velocity vector.

    Returns:
    numpy array: The saturated velocity vector.
    """
    magnitude = np.linalg.norm(velocity)
    if magnitude > max_velocity:
        return velocity * (max_velocity / magnitude)
    return velocity


def get_traj(curr_pos, target_pos, max_vel=0.05, max_accel=1.0, max_jerk=3.0):
    """Get linear trajectory"""

    num_dof = len(curr_pos)
    otg = ruckig.Ruckig(num_dof)
    inp = ruckig.InputParameter(num_dof)
    out = ruckig.OutputParameter(num_dof)

    inp.current_position = curr_pos
    inp.current_velocity = np.zeros(num_dof)
    inp.current_acceleration = np.zeros(num_dof)

    inp.target_position = target_pos
    inp.target_velocity = np.zeros(num_dof)
    inp.target_acceleration = np.zeros(num_dof)

    inp.max_velocity = np.ones(num_dof) * max_vel
    inp.max_acceleration = np.ones(num_dof) * max_accel
    inp.max_jerk = np.ones(num_dof) * max_jerk

    trajectory = ruckig.Trajectory(num_dof)
    result = otg.calculate(inp, trajectory)
    if result == ruckig.Result.ErrorInvalidInput:
        raise Exception("Invalid Input")

    return trajectory


def get_angular_traj(curr_angle, target_angle, max_vel=0.1):
    """
    Get angular trajectory (quaterions)

    curr_angle, target_angle: XYZW
    """

    # Create pyQuaternion objects
    curr_angle = Quaternion(
        x=curr_angle[0],
        y=curr_angle[1],
        z=curr_angle[2],
        w=curr_angle[3],
    )
    target_angle = Quaternion(
        x=target_angle[0],
        y=target_angle[1],
        z=target_angle[2],
        w=target_angle[3],
    )

    return AngularTraj(curr_angle, target_angle, max_vel)


class AngularTraj:
    def __init__(self, initial_angle, target_angle, max_vel):
        distance_pos = Quaternion.distance(initial_angle, target_angle)
        distance_neg = Quaternion.distance(-initial_angle, target_angle)
        # Choose shorter distance - this determines the trajectory duration
        if distance_pos < distance_neg:
            distance = distance_pos
        else:
            distance = distance_neg
        delta_t = 0.001
        num_steps = max(int(np.ceil(distance / (max_vel * delta_t))), 1)
        quat_sequence = []
        for q in Quaternion.intermediates(
            initial_angle, target_angle, num_steps, include_endpoints=True
        ):
            quat_sequence.append(q)

        self.traj_dict = {}
        time = 0
        for idx in range(num_steps):
            self.traj_dict[time] = quat_sequence[idx]
            time += delta_t

        self.duration = time - delta_t

    def at_time(self, t, quat_order="xyzw"):
        quat = (
            self.traj_dict.get(t)
            or self.traj_dict[min(self.traj_dict.keys(), key=lambda key: abs(key - t))]
        )
        if quat_order == "wxyz":
            return quat.q
        else:
            xyzw_quat = [quat.q[1], quat.q[2], quat.q[3], quat.q[0]]
            return xyzw_quat