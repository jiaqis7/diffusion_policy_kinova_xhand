from typing import Optional, Callable, Dict
import os
import re
import enum
import time
import json
import redis
import pickle
import numpy as np
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
import pyzed.sl as sl

from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder

from diffusion_policy.real_world.utils.zed_utils import init_zed, resize_img_to_square

REDIS_HOST = "192.168.1.15"
REDIS_PORT = 6379

class Command(enum.Enum):
    START_RECORDING = 0
    STOP_RECORDING = 1
    RESTART_PUT = 2

class SingleZed(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1920, 1080),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            get_max_k=30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = (min(resolution[0], resolution[1]), min(resolution[0], resolution[1]))
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.START_RECORDING.value,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='bgr24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    
    @staticmethod
    def get_connected_devices_serial():
        # r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password = 'iprl')
        # keys = r.keys('zed_left_image_*')
        # serials = []
        # for key in keys:
        #     try:
        #         key_str = key.decode('utf-8')  # Convert bytes to string
        #         match = re.match(r'zed_left_image_(\d+)', key_str)
        #         if match:
        #             serials.append(int(match.group(1)))
        #     except Exception as e:
        #         print(f"Error decoding key: {e}")
        # return serials

        devices = sl.Camera.get_device_list()
        return [d.serial_number for d in devices]

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)
    
    # ========= user API ===========
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
     
    # ========= interval API ===========
    # def run(self):
    #     # limit threads
    #     threadpool_limits(1)
    #     cv2.setNumThreads(1)

    #     w, h = self.resolution
    #     fps = self.capture_fps

    #     try:
    #         # one-time setup (intrinsics etc, ignore for now)
    #         if self.verbose:
    #             print(f'[SingleZed {self.serial_number}] Main loop started.')

    #         # put frequency regulation
    #         put_idx = None
    #         put_start_time = self.put_start_time
    #         if put_start_time is None:
    #             put_start_time = time.time()

    #         iter_idx = 0
    #         t_start = time.time()
    #         print(f"[SingleZed {self.serial_number}] Creating image buffer with resolution {self.resolution}")
    #         while not self.stop_event.is_set():
    #             # grab frame from redis
    #             if iter_idx == 0:
    #                 print(f"[SingleZed {self.serial_number}] Getting zed image from redis for the first time")
    #             key = f'zed_left_image_{self.serial_number}'
    #             encoded = self.r.get(key)
    #             if encoded is None:
    #                 return None
    #             try:
    #                 payload = pickle.loads(encoded)
    #                 img_left_bgr = payload["image"]
    #                 timestamp = payload["timestamp_ms"]
    #             except Exception as e:
    #                 print(f"Failed to decode image: {e}")
    #                 return None
    #             if iter_idx == 0:
    #                 print(f"[SingleZed {self.serial_number}] Got zed image from redis for the first time")

    #             assert img_left_bgr.shape[0] == self.resolution[1]
    #             assert img_left_bgr.shape[1] == self.resolution[0]
    #             assert img_left_bgr.shape[2] == 3

    #             img_left_bgr = resize_img_to_square(img_left_bgr)
    #             receive_time = time.time()

    #             # grab data
    #             data = dict()
    #             data['camera_receive_timestamp'] = receive_time
    #             data['camera_capture_timestamp'] = timestamp / 1000
    #             data['color'] = img_left_bgr
                
    #             # apply transform
    #             if self.transform is not None:
    #                 put_data = self.transform(data.copy())
    #             else:
    #                 put_data = data

    #             if self.put_downsample:                
    #                 # put frequency regulation
    #                 local_idxs, global_idxs, put_idx \
    #                     = get_accumulate_timestamp_idxs(
    #                         timestamps=[receive_time],
    #                         start_time=put_start_time,
    #                         dt=1/self.put_fps,
    #                         # this is non in first iteration
    #                         # and then replaced with a concrete number
    #                         next_global_idx=put_idx,
    #                         # continue to pump frames even if not started.
    #                         # start_time is simply used to align timestamps.
    #                         allow_negative=True
    #                     )

    #                 for step_idx in global_idxs:
    #                     put_data['step_idx'] = step_idx
    #                     put_data['timestamp'] = receive_time
    #                     print(step_idx, put_data['timestamp'])
    #                     self.ring_buffer.put(put_data, wait=False)
    #             else:
    #                 step_idx = int((receive_time - put_start_time) * self.put_fps)
    #                 put_data['step_idx'] = step_idx
    #                 put_data['timestamp'] = receive_time
    #                 self.ring_buffer.put(put_data, wait=False)

    #             # signal ready after having received some images (~0.5s)
    #             if iter_idx == 30:
    #                 self.ready_event.set()
                
    #             # put to vis
    #             vis_data = data
    #             if self.vis_transform == self.transform:
    #                 vis_data = data
    #             elif self.vis_transform is not None:
    #                 vis_data = self.vis_transform(dict(data))
    #             self.vis_ring_buffer.put(vis_data, wait=False)
                
    #             # record frame
    #             rec_data = data
    #             if self.recording_transform == self.transform:
    #                 rec_data = data
    #             elif self.recording_transform is not None:
    #                 rec_data = self.recording_transform(dict(data))

    #             if self.video_recorder.is_ready():
    #                 self.video_recorder.write_frame(rec_data['color'], 
    #                     frame_time=receive_time)

    #             # perf
    #             t_end = time.time()
    #             duration = t_end - t_start
    #             frequency = np.round(1 / duration, 1)
    #             t_start = t_end
    #             if self.verbose:
    #                 print(f'[SingleZed {self.serial_number}] FPS {frequency}')

    #             # fetch command from queue
    #             try:
    #                 commands = self.command_queue.get_all()
    #                 n_cmd = len(commands['cmd'])
    #             except Empty:
    #                 n_cmd = 0

    #             # execute commands
    #             for i in range(n_cmd):
    #                 command = dict()
    #                 for key, value in commands.items():
    #                     command[key] = value[i]
    #                 cmd = command['cmd']
    #                 if cmd == Command.START_RECORDING.value:
    #                     video_path = str(command['video_path'])
    #                     start_time = command['recording_start_time']
    #                     if start_time < 0:
    #                         start_time = None
    #                     self.video_recorder.start(video_path, start_time=start_time)
    #                 elif cmd == Command.STOP_RECORDING.value:
    #                     self.video_recorder.stop()
    #                     # stop need to flush all in-flight frames to disk, which might take longer than dt.
    #                     # soft-reset put to drop frames to prevent ring buffer overflow.
    #                     put_idx = None
    #                 elif cmd == Command.RESTART_PUT.value:
    #                     put_idx = None
    #                     put_start_time = command['put_start_time']
    #                     # self.ring_buffer.clear()

    #             iter_idx += 1
    #     finally:
    #         self.video_recorder.stop()
    #         self.ready_event.set()
        
    #     if self.verbose:
    #         print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')


    def run(self):
        # 1) Keep CPU thread usage predictable (important for real-time capture)
        threadpool_limits(1)
        cv2.setNumThreads(1)

        # 2) Build ZED init parameters from our config
        #    If you want to honor self.resolution exactly, map it to ZED enums.
        w, h = self.resolution
        fps = int(self.capture_fps)

        # Map (w,h) to ZED RESOLUTION enum (pick the closest)
        # You can also hardcode HD1080 or HD720 if you prefer.
        if (w, h) >= (2208, 1242):
            cam_res = sl.RESOLUTION.HD2K
        elif (w, h) >= (1920, 1080):
            cam_res = sl.RESOLUTION.HD1080
        elif (w, h) >= (1280, 720):
            cam_res = sl.RESOLUTION.HD720
        else:
            cam_res = sl.RESOLUTION.VGA

        init = sl.InitParameters(
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
            camera_resolution=cam_res,
            camera_fps=fps,
            depth_mode=sl.DEPTH_MODE.NONE,   # RGB-only for our pipeline
        )

        cam = sl.Camera()

        # If a specific serial_number was provided, request that device
        if getattr(self, "serial_number", None) is not None:
            init.set_from_serial_number(int(self.serial_number))

        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[SingleZed {self.serial_number}] ZED open failed: {status}")
            # Mark ready to avoid blocking, then exit gracefully
            self.ready_event.set()
            return

        # 3) Allocate an image buffer matching the actual opened camera
        cam_info = cam.get_camera_information()
        size = cam_info.camera_configuration.calibration_parameters.left_cam.image_size
        img_left = sl.Mat(size.width, size.height, sl.MAT_TYPE.U8_C4)

        # 4) Timing setup for downsampled put rate
        put_idx = None
        put_start_time = self.put_start_time if self.put_start_time is not None else time.time()
        iter_idx = 0

        print(f"[SingleZed {self.serial_number}] Capturing via ZED SDK @ {fps} FPS")
        try:
            while not self.stop_event.is_set():
                # Try to grab a frame
                if cam.grab() != sl.ERROR_CODE.SUCCESS:
                    # No frame yet: small sleep to avoid busy-wait
                    time.sleep(0.002)
                    continue

                # Retrieve left image into CPU memory
                cam.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU)
                bgr = img_left.get_data()[:, :, :3]
                # Convert BGR->RGB (DP expects RGB)
                rgb = bgr[..., ::-1].copy()

                # Timestamp bookkeeping
                receive_time = time.time()

                # Pack the data expected by downstream transforms / buffers
                base = {
                    'camera_receive_timestamp': receive_time,
                    'camera_capture_timestamp': receive_time,  # If needed, you can use cam.get_timestamp()
                    'color': rgb,
                }

                # Apply obs transform (resize / color-space / dtype) for the main ring buffer
                put_data = self.transform(dict(base)) if self.transform is not None else base

                # Regulate put frequency (downsample to put_fps if requested)
                if self.put_downsample:
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1.0 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = receive_time
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # Signal “ready” after ~1s of frames to let the rest of the system start
                if iter_idx == 30:
                    self.ready_event.set()

                # Also populate the visualization ring buffer (unscaled RGB frame)
                vis_data = self.vis_transform(dict(base)) if self.vis_transform is not None else base
                self.vis_ring_buffer.put(vis_data, wait=False)

                # If recording is enabled, write to the VideoRecorder
                rec_data = self.recording_transform(dict(base)) if self.recording_transform is not None else base
                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], frame_time=receive_time)

                # Handle commands from the control queue (start/stop recording, restart put, etc.)
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    cmd = int(commands['cmd'][i])
                    if cmd == 0:  # START_RECORDING
                        video_path = str(commands['video_path'][i])
                        start_time = float(commands['recording_start_time'][i])
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == 1:  # STOP_RECORDING
                        self.video_recorder.stop()
                        # Reset the put index so we don't overflow ring buffer after a stop
                        put_idx = None
                    elif cmd == 2:  # RESTART_PUT
                        put_idx = None
                        put_start_time = float(commands['put_start_time'][i])

                iter_idx += 1
        finally:
            # Always stop the recorder and mark ready (so callers don’t hang)
            self.video_recorder.stop()
            self.ready_event.set()
            cam.close()
