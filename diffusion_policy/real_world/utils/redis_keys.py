LAPTOP_HOST = "192.168.1.15"
LAPTOP_PORT = 6379
LAPTOP_PWD = "iprl"

BOT1_NUC_HOST = "192.168.1.11"
BOT1_NUC_PORT = 6379
BOT1_NUC_PWD = "iprl"

BOT2_NUC_HOST = "192.168.1.13"
BOT2_NUC_PORT = 6379
BOT2_NUC_PWD = "iprl"

# Bot 1 Kinova keys for redis
KEY_BOT1_KINOVA_Q = "kinova::bot1::q_s"
KEY_BOT1_KINOVA_Q_DES = "kinova::bot1::q_des"
KEY_BOT1_KINOVA_EE_POS = "kinova::bot1::ee_pos"
KEY_BOT1_KINOVA_EE_QUAT_WXYZ = "kinova::bot1::ee_quat_wxyz"
KEY_BOT1_KINOVA_EE_POS_DES = "kinova::bot1::ee_pos_des"
KEY_BOT1_KINOVA_EE_QUAT_WXYZ_DES = "kinova::bot1::ee_quat_wxyz_des"

KEY_BOT1_ROBOTIQ_SENSOR_POS = "kinova::bot1::gripper_position"
KEY_BOT1_ROBOTIQ_CONTROL_POS_DES = "kinova::bot1::gripper_position_des"

KEY_BOT1_XHAND_POS = "kinova::bot1::xhand_position"
KEY_BOT1_XHAND_POS_DES = "kinova::bot1::xhand_position_des"

KEY_BOT1_STATUS = "kinova::bot1::status"


# Bot 2 Kinova keys for redis
KEY_BOT2_KINOVA_Q = "kinova::bot2::q_s"
KEY_BOT2_KINOVA_Q_DES = "kinova::bot2::q_des"
KEY_BOT2_KINOVA_EE_POS = "kinova::bot2::ee_pos"
KEY_BOT2_KINOVA_EE_QUAT_WXYZ = "kinova::bot2::ee_quat_wxyz"
KEY_BOT2_KINOVA_EE_POS_DES = "kinova::bot2::ee_pos_des"
KEY_BOT2_KINOVA_EE_QUAT_WXYZ_DES = "kinova::bot2::ee_quat_wxyz_des"

KEY_BOT2_ROBOTIQ_SENSOR_POS = "kinova::bot2::gripper_position"
KEY_BOT2_ROBOTIQ_CONTROL_POS_DES = "kinova::bot2::gripper_position_des"

KEY_BOT2_XHAND_POS = "kinova::bot2::xhand_position"
KEY_BOT2_XHAND_POS_DES = "kinova::bot2::xhand_position_des"

KEY_BOT2_STATUS = "kinova::bot2::status"


# Camera keys for redistarget_pose
APP_NAMESPACE = "rgbd"
LEFT_CAMERA_NAME = "left_camera"
KEY_LEFT_CAMERA_POS = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::pos"
KEY_LEFT_CAMERA_ORI = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::ori"
KEY_LEFT_CAMERA_INTRINSIC = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::intrinsic"
KEY_LEFT_CAMERA_DEPTH_MM = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::depth_mm"
KEY_LEFT_CAMERA_IMAGE_BIN = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::image_bin"
KEY_LEFT_CAMERA_TIMESTAMP = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::timestamp"


RIGHT_CAMERA_NAME = "right_camera"
KEY_RIGHT_CAMERA_POS = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::pos"
KEY_RIGHT_CAMERA_ORI = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::ori"
KEY_RIGHT_CAMERA_INTRINSIC = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::intrinsic"
KEY_RIGHT_CAMERA_DEPTH_MM = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::depth_mm"
KEY_RIGHT_CAMERA_IMAGE_BIN = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::image_bin"

KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN = f"{APP_NAMESPACE}::left_right_camera::image_bin"


KEY_CAMERA_DEPTH_BIN = f"{APP_NAMESPACE}::camera::depth_bin"
KEY_CAMERA_POINT_CLOUD_BIN = f"{APP_NAMESPACE}::camera::point_cloud_bin"


