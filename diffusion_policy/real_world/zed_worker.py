# diffusion_policy/real_world/zed_worker.py
import os, time
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform
import multiprocessing as mp
import pyzed.sl as sl
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.cv2_util import get_image_transform
from wild_human.utils.zed_utils import init_zed 

class ZedWorker(mp.Process):
    def __init__(self, shm_manager, out_key="agentview_image",
                 square_size=256, resolution="HD720", depth_mode="NONE",
                 fps=30, launch_timeout=10.0, verbose=False):
        super().__init__(name="ZEDWorker")
        self.out_key = out_key
        self.square_size = square_size
        self.resolution = resolution
        self.depth_mode = depth_mode
        self.fps = int(fps)
        self.verbose = verbose
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        example = {
            out_key: np.zeros((square_size, square_size, 3), np.uint8),
            "timestamp": time.time(),
            "camera_capture_timestamp": 0.0,
            "camera_receive_timestamp": 0.0,
        }
        self.ring = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, examples=example,
            get_max_k=128, get_time_budget=0.2, put_desired_frequency=fps)

    @property
    def is_ready(self): return self.ready_event.is_set()

    def start_wait(self, timeout=30):
        self.ready_event.wait(timeout)

    def stop(self):
        self.stop_event.set()

    def get(self, k=None, out=None):
        return self.ring.get_last_k(k=k, out=out) if k else self.ring.get(out=out)


    def run(self):

        import os, time
        import numpy as np
        from diffusion_policy.common.cv2_util import get_image_transform


        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/zed/lib:/usr/local/cuda/lib64:" + ld

        import pyzed.sl as sl
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = int(self.fps)
        init.depth_mode = sl.DEPTH_MODE.NONE

        try:
            init.sdk_verbose = 0  
        except Exception:
            pass

        if getattr(self, "serial_number", None) is not None:
            try:
                init.set_from_serial_number(int(self.serial_number))
            except Exception:
                pass

        cam = sl.Camera()
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZEDWorker] open failed: {status}")

            # self.ready_event.set()
            return

  
        info = cam.get_camera_information()
        W = info.camera_configuration.calibration_parameters.left_cam.image_size.width
        H = info.camera_configuration.calibration_parameters.left_cam.image_size.height
        mat_left = sl.Mat(W, H, sl.MAT_TYPE.U8_C4)

        tf = get_image_transform(
            input_res=(W, H),                     
            output_res=(self.square_size, self.square_size),
            bgr_to_rgb=False                          
        )

        first_deadline = time.time() + 5.0 
        iter_idx = 0

        try:
            while not self.stop_event.is_set():
                if cam.grab() != sl.ERROR_CODE.SUCCESS:

                    if iter_idx == 0 and time.time() > first_deadline:
                        print("[ZEDWorker] first frame timeout (SDK)")
                        self.ready_event.set()
                    time.sleep(0.002)
                    continue

                cam.retrieve_image(mat_left, sl.VIEW.LEFT, sl.MEM.CPU)
                bgr = mat_left.get_data()[:, :, :3]       
                rgb = bgr[..., ::-1]                       # BGR → RGB
                rgb_sq = tf(rgb).astype(np.uint8)       

                t_recv = time.time()
                try:
                    t_cap = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds() / 1000.0
                except Exception:
                    t_cap = t_recv

                self.ring.put({
                    self.out_key: rgb_sq,              
                    "timestamp": t_recv,
                    "camera_capture_timestamp": t_cap,
                    "camera_receive_timestamp": t_recv
                }, wait=False)

                if iter_idx == 0 and not self.ready_event.is_set():
                    self.ready_event.set()
                    print("[ZEDWorker] first frame ✓; ready set (SDK)")

                iter_idx += 1

        finally:
            try:
                cam.close()
            except Exception:
                pass

            self.ready_event.set()
