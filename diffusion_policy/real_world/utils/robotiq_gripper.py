import numpy as np
import redis

from diffusion_policy.real_world.utils.redis_keys import *
from diffusion_policy.real_world.utils.redis_utils import decode_matlab, encode_matlab

class RobotiqGripper:
    def __init__(
        self,
        host: str = "192.168.1.15",
        port: int = 6379,
        password: str = "iprl",
        sim: bool = False,
    ):
        """Initializes the Robotiq85 Gripper Redis client.

        Args:
            host: Redis hostname (the NUC's ip address).
            port: Redis port.
            password: Redis password.
        """
        self._redis = redis.Redis(host=host, port=port, password=password)
        self._redis_pipe = self._redis.pipeline()
        self._sim = sim

    def goto_pose(self, pos: float)-> None:
        """
        Set the opening distance of the Robotiq85 Gripper. 
        Args
            pos: int between 0 (fully open) and 1 (fully closed)
        """
        robotiq_pos = (np.array([pos])*255).astype(np.uint8) 
        self._redis_pipe.set(KEY_BOT1_ROBOTIQ_CONTROL_POS_DES, encode_matlab(robotiq_pos))
        self._redis_pipe.execute()

    def get_pose(self)-> int:
        """
        Get the opening distance of the Robotiq85 Gripper. 
        Returns
            float between 0 (fully open) and 1 (fully closed)
            todo: this is probably wrong - fix
        """
        if self._sim:
            return 0
        self._redis_pipe.get(KEY_BOT1_ROBOTIQ_SENSOR_POS)
        b_pos = self._redis_pipe.execute()
        return decode_matlab(b_pos[0]).item()