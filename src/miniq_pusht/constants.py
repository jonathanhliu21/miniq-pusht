"""Names of MJCF sensors and sites referenced by the MiniQ environment."""

ROBOT_1_PREFIX = "r1_"
ROBOT_2_PREFIX = "r2_"

GAIT_FREQUENCY_HZ = 50.0

PUSH_T_FREEJOINT = "push_t_freejoint"
PUSH_T_SITES = ("push_t_bar_l", "push_t_bar_r", "push_t_stem")
TARGET_PUSH_T_SITES = (
    "target_push_t_bar_l",
    "target_push_t_bar_r",
    "target_push_t_stem",
)

GYRO_SENSOR = "gyro"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
POSITION_SENSOR = "position"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
SITE_POSITION_SENSOR_SUFFIX = "_pos"

FEET_SITES = ("fr_foot", "fl_foot", "br_foot", "bl_foot")
