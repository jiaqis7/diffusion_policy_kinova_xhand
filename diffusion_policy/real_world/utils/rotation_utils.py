import numpy as np
from scipy.spatial.transform import Rotation

def convert_rotation_matrix_to_quat_xyzw(rot_matrix: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(rot_matrix)
    quat_xyzw = rot.as_quat()
    return quat_xyzw

def convert_quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(quat_xyzw)
    return rot.as_matrix()

def get_quat_error(quat_cur, quat_des):
    """
    Find the quaternion difference between desired and current quaternions (x,y,z,w)
    """
    Rcur = Rotation.from_quat(quat_cur)
    Rdes = Rotation.from_quat(quat_des)

    delta = Rcur * Rdes.inv()

    return delta.as_quat()


def get_ori_from_6d(r6d):
    """Get xyz euler orientation from 6d basis vector representation (used in AO-Grasp)"""

    def normalize(x):
        length = max(np.linalg.norm(x), 1e-8)
        return x / length

    r6d = r6d.reshape(2, 3)
    x, y = r6d
    x = normalize(x)
    y -= np.dot(x, y) * x
    y = normalize(y)
    z = np.cross(x, y, axis=-1)
    R = Rotation.from_matrix(np.stack([x, y, z], axis=-1))
    ori = R.as_euler("xyz")
    return ori


def get_H(R, p):
    """Construct homogenous transformation matrix H"""
    H = np.zeros((4, 4))
    H[0:3, 0:3] = R
    H[0:3, 3] = p
    H[3, 3] = 1
    return H


def get_H_inv(H):
    """Get inverse of homogenous transformation matrix H"""

    H_inv = np.zeros(H.shape)
    H_inv[3, 3] = 1
    R = H[:3, :3]
    P = H[:3, 3]

    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ P

    return H_inv


def rotate_quat(q, R_transform):
    """Rotate quaternion q by R_transform"""
    Rq = Rotation.from_quat(q)
    Rt = Rotation.from_matrix(R_transform)
    R_new = Rt * Rq
    return R_new.as_quat()


def rotate_xyz_ori(ori, R_transform):
    """Rotate xyz euler angle by R_transform"""
    Rq = Rotation.from_euler("xyz", ori)
    Rt = Rotation.from_matrix(R_transform)
    R_new = Rt * Rq
    return R_new.as_euler("xyz")


def transform_pt(pt, H):
    """Transform pt (3, ) by H"""

    pt_new = (H @ np.append([pt], 1))[:3]
    return pt_new