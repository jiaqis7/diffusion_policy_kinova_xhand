import redis
import numpy as np
import pyzed.sl as sl
import time
import cv2
import pickle
import argparse

from diffusion_policy.real_world.utils.zed_utils import init_zed

def main(serial_number: int):
    redis_key = f"zed_left_image_{serial_number}"
    r = redis.Redis(host='localhost', port=6379)

    zed = init_zed(resolution="HD1080", depth_mode="NONE", serial_number=serial_number)
    img_left = sl.Mat(1920, 1080, sl.MAT_TYPE.U8_C4)

    print(f"[ZED Server] Pushing images to Redis key: {redis_key}")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU)
            timestamp_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            img_left_bgr = img_left.get_data()[:, :, :3]

            payload = {
                'image': img_left_bgr,
                'timestamp_ms': timestamp_ms
            }

            encoded = pickle.dumps(payload)
            r.set(redis_key, encoded)
            print(f"Last image timestamp: {timestamp_ms}", end="\r")
        else:
            time.sleep(0.005)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED Redis Server")
    parser.add_argument("--serial", type=int, default=27432424, help="ZED camera serial number")
    args = parser.parse_args()

    main(serial_number=args.serial)