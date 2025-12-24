# real_env.py


from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.utils.kinova_bimanual_with_xhand import KinovaBimanualWithXhand
# from diffusion_policy.real_world.multi_zed import MultiZed, SingleZed
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from diffusion_policy.real_world.zed_worker import ZedWorker

DEFAULT_OBS_KEY_MAP = {
    # robot
    "robot1_eef_pos": "robot1_eef_pos",
    "robot1_eef_quat": "robot1_eef_quat",
    "robot1_xhand_qpos": "robot1_xhand_qpos",
    "robot2_eef_pos": "robot2_eef_pos",
    "robot2_eef_quat": "robot2_eef_quat",
    "robot2_xhand_qpos": "robot2_xhand_qpos",

    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealXhandEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip=None,
            # env params
            frequency=30,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(640,480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            # tcp_offset=0.13,
            init_joints=False,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(1920, 1080),
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(720, 720),
            # shared memory
            shm_manager=None
            ):
        

        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        self.shm_manager = shm_manager
        self.obs_image_resolution = obs_image_resolution


        # if camera_serial_numbers is None:
        #     camera_serial_numbers = SingleZed.get_connected_devices_serial()

        video_capture_resolution_square = (
            min(video_capture_resolution[0], video_capture_resolution[1]),
            min(video_capture_resolution[0], video_capture_resolution[1])
        )
        print(f"[Real Env] creating image transform with input shape {video_capture_resolution_square} and output shape {obs_image_resolution}")

        # color_tf = get_image_transform(
        #     input_res=video_capture_resolution_square,
        #     output_res=obs_image_resolution, 
        #     # obs output rgb
        #     bgr_to_rgb=True)
        # color_transform = color_tf
        # if obs_float32:
        #     color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        if obs_float32:
            # old behavior: resize + RGB + /255
            color_tf = get_image_transform(
                input_res=video_capture_resolution_square,
                output_res=obs_image_resolution,
                bgr_to_rgb=True,
            )
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255
        else:
            # raw mode: don't touch
            def color_transform(x):
                return x


        def transform(data):
            data['color'] = color_transform(data['color'])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution_square,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transform = vis_transform
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transform = vis_transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)


        self.zed = ZedWorker(
            shm_manager=self.shm_manager,
            out_key="agentview_image",      
            square_size=self.obs_image_resolution[0], 
            resolution="HD720",              
            depth_mode="NONE",               
            fps=30,
            verbose=False
        )
        
        self.multi_cam_vis = None


        robot = KinovaBimanualWithXhand(
            shm_manager=shm_manager,
            frequency=30,
            launch_timeout=30,
            verbose=False,
            get_max_k=max_obs_buffer_size,
            host=robot_ip,
            password="iprl",
        )


        self.robot = robot
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_zed_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.zed.is_ready and self.robot.is_ready
    
    def start(self, wait=True):
        self.zed.start()
        self.robot.start()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop()
        self.zed.stop()
        if wait:
            self.stop_wait()


    def start_wait(self):
        import time
        t0 = time.time()
        TIMEOUT = 60.0
        print("[RealEnv] start_wait(): begin", flush=True)

        while not self.robot.is_ready:
            if time.time() - t0 > TIMEOUT:
                raise TimeoutError("[RealEnv] robot not ready within timeout")
            time.sleep(0.05)
        print("[RealEnv] robot ready ✓", flush=True)

        while not self.zed.is_ready:
            if time.time() - t0 > TIMEOUT:
                raise TimeoutError("[RealEnv] zed not ready within timeout")
            time.sleep(0.05)
        print("[RealEnv] zed ready ✓", flush=True)

        if self.multi_cam_vis is not None:
            while not self.multi_cam_vis.is_ready:
                if time.time() - t0 > TIMEOUT:
                    raise TimeoutError("[RealEnv] multi_cam_vis not ready within timeout")
                time.sleep(0.05)
            print("[RealEnv] multi_cam_vis ready ✓", flush=True)


    def stop_wait(self):
        self.robot.stop_wait()

        if hasattr(self.zed, "stop_wait"):
            self.zed.stop_wait()
        else:
            self.zed.join()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()


    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        assert self.is_ready

        k = self.n_obs_steps
        cam = self.zed.get(k=k)  

        last_robot_data = self.robot.get_all_state()


        dt = 1 / self.frequency
        last_timestamp = cam['timestamp'][-1]
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)


        this_timestamps = cam['timestamp']
        idxs = []
        for t in obs_align_timestamps:
            before = np.nonzero(this_timestamps < t)[0]
            idxs.append(before[-1] if len(before) > 0 else 0)

        camera_obs = {
            'agentview_image': cam['agentview_image'][idxs]
        }

        robot_timestamps = last_robot_data['robot_receive_timestamp']
        r_idxs = []
        for t in obs_align_timestamps:
            before = np.nonzero(robot_timestamps < t)[0]
            r_idxs.append(before[-1] if len(before) > 0 else 0)

        robot_obs_raw = {self.obs_key_map[k]: v for k, v in last_robot_data.items() if k in self.obs_key_map}
        robot_obs = {k: v[r_idxs] for k, v in robot_obs_raw.items()}

        if self.obs_accumulator is not None:
            self.obs_accumulator.put(robot_obs_raw, robot_timestamps)

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        # print(obs_data.keys())
        # print(obs_data['robot2_xhand_qpos'])
        # print("done!!")
        return obs_data

    

    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        
        print("[env.exec_actions] module file:", __file__, flush=True)

        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)


        print(f"[exec_actions] Received {len(actions)} actions with shape {actions.shape}")
        
        # Validate action dimension
        assert actions.shape[-1] == 38, \
            f"Expected 16D bimanual actions [pos1(3),quat1(4),gripper1(1),pos2(3),quat2(4),gripper2(1)], got {actions.shape[-1]}D"


        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
    
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
    
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        # this_video_dir = self.video_dir.joinpath(str(episode_id))
        # this_video_dir.mkdir(parents=True, exist_ok=True)
        this_video_dir = self.video_dir
        # n_cameras = self.zed.n_cameras
        n_cameras = 1
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # start recording on zed
        # self.zed.restart_put(start_time=start_time)
        # self.zed.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        # self.zed.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                # print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None


    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        # print(f'Episode {episode_id} dropped!')