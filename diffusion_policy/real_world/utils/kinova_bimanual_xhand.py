import pdb 
import redis
import time
import numpy as np

from wild_human.configs.redis_keys import *
from wild_human.utils.redis_utils import *
import wild_human.utils.traj_utils as t_utils

class KinovaBimanualXhand:
    def __init__(self, host, password):
        _redis = redis.Redis(
            host=host, port=6379, password=password,
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

        self.key_xhand_pos_left = f"kinova::bot1::xhand_position"
        self.key_xhand_pos_des_left = f"kinova::bot1::xhand_position_des"

        # RIGHT
        self.key_kinova_q_right = f"kinova::bot2::q"
        self.key_kinova_q_des_right = f"kinova::bot2::q_des"
        self.key_kinova_ee_pos_right = f"kinova::bot2::ee_pos"
        self.key_kinova_ee_quat_wxyz_right = f"kinova::bot2::ee_quat_wxyz"
        self.key_kinova_ee_pos_des_right = f"kinova::bot2::ee_pos_des"
        self.key_kinova_ee_quat_wxyz_des_right = f"kinova::bot2::ee_quat_wxyz_des"
        self.key_robotiq_sensor_pos_right = f"kinova::bot2::gripper_position"
        self.key_robotiq_control_pos_des_right = f"kinova::bot2::gripper_position_des"

        self.key_xhand_pos_right = f"kinova::bot2::xhand_position"
        self.key_xhand_pos_des_right = f"kinova::bot2::xhand_position_des"

        # Joint control keys
        self.key_kinova_q_des_direct_left = f"kinova::bot1::q_des_direct"
        self.key_kinova_q_des_direct_right = f"kinova::bot2::q_des_direct"

        self.redis_pipe = _redis.pipeline()
        self.ee_home_pos1 = np.array([-0.02, -0.65, 0.18])
        self.ee_home_quat_wxyz1 = np.array([0.7071, 0.7071, 0.0, 0.0])
        self.ee_home_pos2 = np.array([0.08, 0.65, 0.08])
        self.ee_home_quat_wxyz2 = np.array([-0.1943, 0.1943, 0.6801, 0.6801])

        self.gripper_home_pos1 = np.array([0])
        self.gripper_home_pos2 = np.array([0])


        self.xhand_home_pos1 = np.array([0.0, 0.0, 0.0, 
                                        0.0, 0.0, 0.0, 
                                        0.0, 0.0, 
                                        0.0, 0.0, 
                                        0.0, 0.0     ])
        

        self.xhand_home_pos2 = np.array([0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0,
                                        0.0, 0.0,
                                        0.0, 0.0     ])

        # Joint angle home positions (stored in radians, sent as degrees to robot)
        # Robot1 degrees: {121.0, 52.0, -170.0, -114.0, -58.0, -62.0, -179.0}
        self.joint_home_q1 = np.array([2.1118, 0.9076, -2.9671, -1.9897, -1.0123, -1.0821, -3.1241])
        # Robot2 degrees: {-90.0, 51.0, 125.0, -110.0, 50.0, -60.0, 16.0}
        self.joint_home_q2 = np.array([-1.5708, 0.8901, 2.1817, -1.9199, 0.8727, -1.0472, 0.2793])

######################################################
####################### Gripper ########################
######################################################
        



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
    

######################################################
####################### Xhand ########################
######################################################
    

    def go_noisy_home_xhand(self, noise_level=0.03):
        ee_noisy_home_pos1 = np.random.uniform(self.ee_home_pos1 - noise_level, self.ee_home_pos1 + noise_level)
        ee_noisy_home_pos2 = np.random.uniform(self.ee_home_pos2 - noise_level, self.ee_home_pos2 + noise_level)

        eef_pos_cur1, eef_quat_cur1, _, _, eef_pos_cur2, eef_quat_cur2, _, _ = self.get_pose_xhand()
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
            self.goto_pose_xhand(pos_des1=pos_command1, quat_wxyz_des1=ori_command1, xhand_des1=self.xhand_home_pos1, 
                                 pos_des2=pos_command2, quat_wxyz_des2=ori_command2, xhand_des2=self.xhand_home_pos2)



    def go_home_xhand(self):
        eef_pos_cur1, eef_quat_cur1, _, _, eef_pos_cur2, eef_quat_cur2, _, _ = self.get_pose_xhand()
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
            self.goto_pose_xhand(pos_des1=pos_command1, quat_wxyz_des1=ori_command1, xhand_des1=self.xhand_home_pos1, 
                                 pos_des2=pos_command2, quat_wxyz_des2=ori_command2, xhand_des2=self.xhand_home_pos2)
    


    def goto_pose_xhand(self, pos_des1, quat_wxyz_des1, xhand_des1, pos_des2, quat_wxyz_des2, xhand_des2):
        # print("GOTOPOSE", pos_des)
        self.redis_pipe.set(self.key_kinova_ee_pos_des_left, encode_matlab(pos_des1))
        self.redis_pipe.set(self.key_kinova_ee_quat_wxyz_des_left, encode_matlab(quat_wxyz_des1))
        self.redis_pipe.set(self.key_xhand_pos_des_left, encode_matlab(xhand_des1))
        self.redis_pipe.set(self.key_kinova_ee_pos_des_right, encode_matlab(pos_des2))
        self.redis_pipe.set(self.key_kinova_ee_quat_wxyz_des_right, encode_matlab(quat_wxyz_des2))
        self.redis_pipe.set(self.key_xhand_pos_des_right, encode_matlab(xhand_des2))
        self.redis_pipe.execute()
    

    def get_pose_xhand(self):
        """Gets the pose of the end-effector.

        Returns:
            (XYZ position, XYZW quaternion) 2-tuple.
        """
        self.redis_pipe.get(self.key_kinova_ee_pos_left)
        self.redis_pipe.get(self.key_kinova_ee_quat_wxyz_left)
        self.redis_pipe.get(self.key_xhand_pos_left)
        self.redis_pipe.get(self.key_kinova_ee_pos_right)
        self.redis_pipe.get(self.key_kinova_ee_quat_wxyz_right)
        self.redis_pipe.get(self.key_xhand_pos_right)
        self.redis_pipe.get(self.key_kinova_q_left)
        self.redis_pipe.get(self.key_kinova_q_right)
        b_pos1, b_quat_wxyz1, b_xhand1, b_pos2, b_quat_wxyz2, b_xhand2, b_q1, b_q2 = self.redis_pipe.execute()

        pos1 = decode_matlab(b_pos1)
        quat_wxyz1 = decode_matlab(b_quat_wxyz1)
        xhand1 = decode_matlab(b_xhand1)
        q1 = decode_matlab(b_q1)

        pos2 = decode_matlab(b_pos2)
        quat_wxyz2 = decode_matlab(b_quat_wxyz2)
        xhand2 = decode_matlab(b_xhand2)
        q2 = decode_matlab(b_q2)
        # print("&&&&&&&&&&&&&&&&&&&")
        # print(xhand2)

        return pos1, quat_wxyz1, xhand1, q1, pos2, quat_wxyz2, xhand2, q2


######################################################
##################### Joint Control ##################
######################################################

    def goto_joint_xhand(self, q_des1, xhand_des1, q_des2, xhand_des2):
        """
        Joint angle control for both arms with xhand.

        Args:
            q_des1: 7D joint angles for robot1 (in radians)
            xhand_des1: 12D xhand positions for robot1 (in radians)
            q_des2: 7D joint angles for robot2 (in radians)
            xhand_des2: 12D xhand positions for robot2 (in radians)

        Note:
            - Arm joint angles are converted from radians to degrees before sending
            - Xhand positions are sent as radians (no conversion)
        """
        # Convert arm joint angles: radians -> degrees
        q_des1_deg = np.rad2deg(q_des1)
        q_des2_deg = np.rad2deg(q_des2)

        # Send to Redis: arm in degrees, xhand in radians
        self.redis_pipe.set(self.key_kinova_q_des_direct_left, encode_matlab(q_des1_deg))
        self.redis_pipe.set(self.key_xhand_pos_des_left, encode_matlab(xhand_des1))  # radians
        self.redis_pipe.set(self.key_kinova_q_des_direct_right, encode_matlab(q_des2_deg))
        self.redis_pipe.set(self.key_xhand_pos_des_right, encode_matlab(xhand_des2))  # radians
        self.redis_pipe.execute()

    def get_joint_xhand(self):
        """
        Get joint angles and xhand positions for both arms.

        Returns:
            q1: 7D joint angles for robot1 (in radians)
            xhand1: 12D xhand positions for robot1 (in radians)
            q2: 7D joint angles for robot2 (in radians)
            xhand2: 12D xhand positions for robot2 (in radians)

        Note:
            - Arm joint angles are converted from degrees to radians
            - Xhand positions are already in radians (no conversion)
        """
        self.redis_pipe.get(self.key_kinova_q_left)
        self.redis_pipe.get(self.key_xhand_pos_left)
        self.redis_pipe.get(self.key_kinova_q_right)
        self.redis_pipe.get(self.key_xhand_pos_right)
        b_q1, b_xhand1, b_q2, b_xhand2 = self.redis_pipe.execute()

        # Decode from Redis
        q1_deg = decode_matlab(b_q1)
        xhand1 = decode_matlab(b_xhand1)  # already in radians
        q2_deg = decode_matlab(b_q2)
        xhand2 = decode_matlab(b_xhand2)  # already in radians

        # Convert arm joint angles: degrees -> radians
        q1 = np.deg2rad(q1_deg)
        q2 = np.deg2rad(q2_deg)

        return q1, xhand1, q2, xhand2


if __name__ == "__main__":
    kinova = KinovaBimanualXhand(host="192.168.1.15", password="iprl")
    kinova.go_home_xhand()
    