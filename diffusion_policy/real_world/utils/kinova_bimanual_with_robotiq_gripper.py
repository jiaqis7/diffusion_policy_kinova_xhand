# kinova_bimanual_with_robotiq_gripper.py

import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from scipy.spatial.transform import Rotation as R
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from .kinova_bimanual import KinovaBimanual

NUC_HOST = "192.168.1.15"
NUC_PWD = "iprl"

class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1

class KinovaBimanualWithGripper(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager, 
                 frequency=125,
                 launch_timeout=3,
                 verbose=False,
                 get_max_k=128,
                 receive_keys=None,
                 host="192.168.1.15",
                 password="iprl"
                 ):
        assert 0 < frequency <= 500

        super().__init__(name="KinovaBimanualController")

        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.host = host
        self.password = password

        # build input queue
        # Each waypoint contains poses for both arms: [pos1(3), quat1(4), gripper1(1), pos2(3), quat2(4), gripper2(1)] = 16D
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': np.zeros((16,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        if receive_keys is None:
            receive_keys = [
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
                "robot2_eef_pos",
                "robot2_eef_quat",
                "robot2_gripper_qpos",
            ]
        example = dict(
            robot1_eef_pos=np.zeros([3], dtype=np.float64),
            robot1_eef_quat=np.zeros([4], dtype=np.float64),
            robot1_gripper_qpos=np.zeros([1], dtype=np.float64),
            robot2_eef_pos=np.zeros([3], dtype=np.float64),
            robot2_eef_quat=np.zeros([4], dtype=np.float64),
            robot2_gripper_qpos=np.zeros([1], dtype=np.float64),
        )
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=False):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[KinovaBimanual] Controller process spawned at {self.pid}")

    def stop(self, wait=False):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (16,), f"Expected pose to have size 16 but got {pose.shape[0]}"

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def get_robot_pose(self, kinova):
        """Get poses from both arms and return as concatenated array"""
        pos1, quat1, gripper1, q1, pos2, quat2, gripper2, q2 = kinova.get_pose()
        
        # Concatenate: [pos1(3), quat1(4), gripper1(1), pos2(3), quat2(4), gripper2(1)]
        return np.concatenate([pos1, quat1, [gripper1], pos2, quat2, [gripper2]])

    def run(self):


        # kinova = KinovaBimanual(host=NUC_HOST, password=NUC_PWD)

        kinova = KinovaBimanual(host=self.host, password=self.password)

        try:
            if self.verbose:
                print(f"[KinovaBimanual] connect to robot")

            # home robot

            # kinova.go_noisy_home()
            kinova.go_home()
            print(f"[KinovaBimanual] robot reset")

            self.ready_event.set()

            dt = 1. / self.frequency

            current_target_pose = self.get_robot_pose(kinova)
            

            iter_idx = 0
            keep_running = True
            while keep_running:

                t_now = time.monotonic()
                # pose_command = pose_interp(t_now)
                pose_command = current_target_pose
                # Extract poses for both arms from interpolated command
                # Format: [pos1(3), quat1(4), gripper1(1), pos2(3), quat2(4), gripper2(1)]
                pos1 = pose_command[:3]
                quat1 = pose_command[3:7]
                gripper1 = pose_command[7]
                pos2 = pose_command[8:11]
                quat2 = pose_command[11:15]
                gripper2 = pose_command[15]
                

                # Send commands to both arms
                kinova.goto_pose(
                    pos_des1=pos1,
                    quat_wxyz_des1=quat1,
                    gripper_des1=gripper1,
                    pos_des2=pos2,
                    quat_wxyz_des2=quat2,
                    gripper_des2=gripper2
                )



                # update robot state
                state = dict()
                pos1, quat1, gripper1, q1, pos2, quat2, gripper2, q2 = kinova.get_pose()
                
                state['robot1_eef_pos'] = pos1
                state['robot1_eef_quat'] = quat1
                state['robot1_gripper_qpos'] = np.array([gripper1])
                state['robot2_eef_pos'] = pos2
                state['robot2_eef_quat'] = quat2
                state['robot2_gripper_qpos'] = np.array([gripper2])
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)



                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])

                except Empty:
                    n_cmd = 0





                for i in range(n_cmd):
                    
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # target pose is 16d: [pos1(3), quat1(4), gripper1(1), pos2(3), quat2(4), gripper2(1)]
                        action = command['target_pose']
                        assert len(action) == 16

                        # target_time = float(command['target_time'])
                        # # translate global time to monotonic time
                        # target_time = time.monotonic() - time.time() + target_time
                        # curr_time = t_now + dt
                        # pose_interp = pose_interp.schedule_waypoint(
                        #     pose=action,
                        #     time=target_time,
                        #     curr_time=curr_time,
                        #     last_waypoint_time=last_waypoint_time
                        # )
                        # last_waypoint_time = target_time

                        current_target_pose = action
                    else:
                        keep_running = False
                        break

                precise_wait(t_now + dt - time.monotonic())

                # first loop successful, ready to receive command
                # if iter_idx == 0:
                #     self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[KinovaBimanual] Actual frequency {1/(time.perf_counter() - t_now)}")
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[KinovaBimanual] Disconnected from robot")