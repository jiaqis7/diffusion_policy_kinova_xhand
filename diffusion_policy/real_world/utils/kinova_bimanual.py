# kinova_bimanual.py

import pdb 
import redis
import time
import numpy as np

from wild_human.configs.redis_keys import *
from wild_human.utils.redis_utils import *
import wild_human.utils.traj_utils as t_utils

class KinovaBimanual:
    def __init__(self, host, password):
        _redis = redis.Redis(
            host=host, port=6379, password=password, socket_timeout=1.0, socket_connect_timeout=1.0
        )

        # LEFT
        self.key_kinova_q_left = f"kinova::bot1::q"
        self.key_kinova_q_des_left = f"kinova::bot1::q_des"
        self.key_kinova_ee_pos_left = f"kinova::bot1::ee_pos"
        self.key_kinova_ee_quat_wxyz_left = f"kinova::bot1::ee_quat_wxyz"
        self.key_kinova_ee_pos_des_left = f"kinova::bot1::ee_pos_des"
        self.key_kinova_ee_quat_wxyz_des_left = f"kinova::bot1::ee_quat_wxyz_des"
        self.key_robotiq_sensor_pos_left = f"kinova::bot1::gripper_position"
        self.key_robotiq_control_pos_des_left = f"kinova::bot1::gripper_position_des"

        # RIGHT
        self.key_kinova_q_right = f"kinova::bot2::q"
        self.key_kinova_q_des_right = f"kinova::bot2::q_des"
        self.key_kinova_ee_pos_right = f"kinova::bot2::ee_pos"
        self.key_kinova_ee_quat_wxyz_right = f"kinova::bot2::ee_quat_wxyz"
        self.key_kinova_ee_pos_des_right = f"kinova::bot2::ee_pos_des"
        self.key_kinova_ee_quat_wxyz_des_right = f"kinova::bot2::ee_quat_wxyz_des"
        self.key_robotiq_sensor_pos_right = f"kinova::bot2::gripper_position"
        self.key_robotiq_control_pos_des_right = f"kinova::bot2::gripper_position_des"

        self.redis_pipe = _redis.pipeline()
        self.ee_home_pos1 = np.array([0.03, -0.57, -0.08])
        self.ee_home_quat_wxyz1 = np.array([0.067996, 0.918431, 0.0257873, 0.38884])
        self.ee_home_pos2 = np.array([-0.0350776, 0.368761, 0.0298257])
        self.ee_home_quat_wxyz2 = np.array([0.382063, -0.055829, 0.922089, 0.0257496,])

        self.gripper_home_pos1 = np.array([0])
        self.gripper_home_pos2 = np.array([0])

    def go_noisy_home(self, noise_level=0.03):
        ee_noisy_home_pos1 = np.random.uniform(self.ee_home_pos1 - noise_level, self.ee_home_pos1 + noise_level)
        ee_noisy_home_pos2 = np.random.uniform(self.ee_home_pos2 - noise_level, self.ee_home_pos2 + noise_level)

        eef_pos_cur1, eef_quat_cur1, _, _, eef_pos_cur2, eef_quat_cur2, _, _ = self.get_pose()
        linear_traj1 = t_utils.get_traj(eef_pos_cur1, ee_noisy_home_pos1, max_vel=0.07)
        angular_traj1 = t_utils.get_angular_traj(eef_quat_cur1, self.ee_home_quat_wxyz1)
        linear_traj2 = t_utils.get_traj(eef_pos_cur2, ee_noisy_home_pos2, max_vel=0.07)
        angular_traj2 = t_utils.get_angular_traj(eef_quat_cur2, self.ee_home_quat_wxyz2)
        duration = max(linear_traj1.duration, angular_traj1.duration, linear_traj2.duration, angular_traj2.duration)
        start_time = time.time()
        while (time.time() - start_time) < duration:
            curr_time = time.time() - start_time
            pos_command1 = np.array(linear_traj1.at_time(curr_time)[0])
            ori_command1 = np.array(angular_traj1.at_time(curr_time))
            pos_command2 = np.array(linear_traj2.at_time(curr_time)[0])
            ori_command2 = np.array(angular_traj2.at_time(curr_time))
            self.goto_pose(pos_des1=pos_command1, quat_wxyz_des1=ori_command1, gripper_des1=self.gripper_home_pos1, pos_des2=pos_command2, quat_wxyz_des2=ori_command2, gripper_des2=self.gripper_home_pos2)


    def go_home(self):
        eef_pos_cur1, eef_quat_cur1, _, _, eef_pos_cur2, eef_quat_cur2, _, _ = self.get_pose()
        linear_traj1 = t_utils.get_traj(eef_pos_cur1, self.ee_home_pos1, max_vel=0.07)
        angular_traj1 = t_utils.get_angular_traj(eef_quat_cur1, self.ee_home_quat_wxyz1)
        linear_traj2 = t_utils.get_traj(eef_pos_cur2, self.ee_home_pos2, max_vel=0.07)
        angular_traj2 = t_utils.get_angular_traj(eef_quat_cur2, self.ee_home_quat_wxyz2)
        duration = max(linear_traj1.duration, angular_traj1.duration, linear_traj2.duration, angular_traj2.duration)
        start_time = time.time()
        while (time.time() - start_time) < duration:
            curr_time = time.time() - start_time
            pos_command1 = np.array(linear_traj1.at_time(curr_time)[0])
            ori_command1 = np.array(angular_traj1.at_time(curr_time))
            pos_command2 = np.array(linear_traj2.at_time(curr_time)[0])
            ori_command2 = np.array(angular_traj2.at_time(curr_time))
            self.goto_pose(pos_des1=pos_command1, quat_wxyz_des1=ori_command1, gripper_des1=self.gripper_home_pos1, pos_des2=pos_command2, quat_wxyz_des2=ori_command2, gripper_des2=self.gripper_home_pos2)


    def goto_pose(self, pos_des1, quat_wxyz_des1, gripper_des1, pos_des2, quat_wxyz_des2, gripper_des2):
        # print("GOTOPOSE", pos_des)
        self.redis_pipe.set(self.key_kinova_ee_pos_des_left, encode_matlab(pos_des1))
        self.redis_pipe.set(self.key_kinova_ee_quat_wxyz_des_left, encode_matlab(quat_wxyz_des1))
        self.redis_pipe.set(self.key_robotiq_control_pos_des_left, encode_matlab(np.array([gripper_des1])))
        self.redis_pipe.set(self.key_kinova_ee_pos_des_right, encode_matlab(pos_des2))
        self.redis_pipe.set(self.key_kinova_ee_quat_wxyz_des_right, encode_matlab(quat_wxyz_des2))
        self.redis_pipe.set(self.key_robotiq_control_pos_des_right, encode_matlab(np.array([gripper_des2])))
        self.redis_pipe.execute()

    def close_gripper(self):
        self.redis_pipe.set(self.key_robotiq_control_pos_des_left, "1")
        self.redis_pipe.set(self.key_robotiq_control_pos_des_right, "1")
        self.redis_pipe.execute()

    def open_gripper(self):
        self.redis_pipe.set(self.key_robotiq_control_pos_des_left, "0")
        self.redis_pipe.set(self.key_robotiq_control_pos_des_right, "0")
        self.redis_pipe.execute()
        

    def get_pose(self):
        """Gets the pose of the end-effector.

        Returns:
            (XYZ position, XYZW quaternion) 2-tuple.
        """
        self.redis_pipe.get(self.key_kinova_ee_pos_left)
        self.redis_pipe.get(self.key_kinova_ee_quat_wxyz_left)
        self.redis_pipe.get(self.key_robotiq_sensor_pos_left)
        self.redis_pipe.get(self.key_kinova_ee_pos_right)
        self.redis_pipe.get(self.key_kinova_ee_quat_wxyz_right)
        self.redis_pipe.get(self.key_robotiq_sensor_pos_right)
        self.redis_pipe.get(self.key_kinova_q_left)
        self.redis_pipe.get(self.key_kinova_q_right)
        b_pos1, b_quat_wxyz1, b_gripper1, b_pos2, b_quat_wxyz2, b_gripper2, b_q1, b_q2 = self.redis_pipe.execute()
        pos1 = decode_matlab(b_pos1)
        quat_wxyz1 = decode_matlab(b_quat_wxyz1)
        gripper1 = decode_matlab(b_gripper1)
        q1 = decode_matlab(b_q1)
        pos2 = decode_matlab(b_pos2)
        quat_wxyz2 = decode_matlab(b_quat_wxyz2)
        gripper2 = decode_matlab(b_gripper2)
        q2 = decode_matlab(b_q2)
        return pos1, quat_wxyz1, gripper1, q1, pos2, quat_wxyz2, gripper2, q2


if __name__ == "__main__":
    kinova = KinovaBimanual(host="192.168.1.15", password="iprl")
    kinova.go_home()
    # pos1, quat1, gripper1, q1, pos2, quat2, gripper2, q2 = kinova.get_pose()
    # print("Pos1: ", pos1, ", Quat1: ", quat1, ", Gripper1: ", gripper1, ", Q1: ", q1)
    # print("Pos2: ", pos2, ", Quat2: ", quat2, ", Gripper2: ", gripper2, ", Q2: ", q2)
    # pos1[2] = pos1[2] - 0.1
    # pos2[2] = pos2[2] - 0.1
    # gripper1 = 0
    # gripper2 = 0
    # kinova.goto_pose(pos_des1=pos1, quat_wxyz_des1=quat1, gripper_des1=gripper1, pos_des2=pos2, quat_wxyz_des2=quat2, gripper_des2=gripper2)
    
    # print("Done")