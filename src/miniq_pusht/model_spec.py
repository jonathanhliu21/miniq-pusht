from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco

from .constants import (
    ACCELEROMETER_SENSOR,
    FEET_SITES,
    GLOBAL_ANGVEL_SENSOR,
    GLOBAL_LINVEL_SENSOR,
    GYRO_SENSOR,
    LOCAL_LINVEL_SENSOR,
    POSITION_SENSOR,
    PUSH_T_FREEJOINT,
    PUSH_T_SITES,
    ROBOT_1_PREFIX,
    ROBOT_2_PREFIX,
    SITE_POSITION_SENSOR_SUFFIX,
    TARGET_PUSH_T_SITES,
    UPVECTOR_SENSOR,
)


@dataclass(frozen=True)
class MiniQSpec:
    """Constants (joint limits), addresses of MiniQ robots"""

    # robot prefix for names
    miniq_prefix: str

    # IDs
    base_body_id: int
    imu_site_id: int
    feet_site_ids: jax.Array

    # structural
    nbody: int

    # extracted state
    base_mass: float
    init_q: jax.Array
    default_pose: jax.Array
    ctrlrange: jax.Array

    # derived joint limits
    soft_lowers: jax.Array
    soft_uppers: jax.Array

    # sensor addresses (IMU + feet only — actuated joints read qpos/qvel
    # directly via the addresses below)
    position_adr: jax.Array
    upvector_adr: jax.Array
    gyro_adr: jax.Array
    accelerometer_adr: jax.Array
    global_linvel_adr: jax.Array
    global_angvel_adr: jax.Array
    local_linvel_adr: jax.Array
    foot_linvel_adr: jax.Array

    # qpos / qvel addresses of the 8 actuated joints, in actuator order.
    # 0,2,4,6 drive the hips; 1,3,5,7 drive the knees
    actuator_qposadr: jax.Array
    actuator_dofadr: jax.Array

    @classmethod
    def from_mj_model(
        cls,
        miniq_prefix: str, 
        mj_model: mujoco.MjModel,
        config: config_dict.ConfigDict,
    ) -> "MiniQSpec":
        base_body_id = mj_model.body(f"{miniq_prefix}base_link").id
        imu_site_id = mj_model.site(f"{miniq_prefix}imu").id
        feet_site_ids = jp.array([mj_model.site(miniq_prefix + name).id for name in FEET_SITES])

        # get joint IDs of actuators for this robot
        actuator_ids = [
            actuator_id
            for actuator_id in range(mj_model.nu)
            if mj_model.actuator(actuator_id).name.startswith(miniq_prefix)
        ]
        actuated_jnt_ids = jp.array(
            [
                mj_model.actuator_trnid[actuator_id, 0]
                for actuator_id in actuator_ids
            ]
        )
        actuator_qposadr = jp.array(mj_model.jnt_qposadr[actuated_jnt_ids])
        actuator_dofadr = jp.array(mj_model.jnt_dofadr[actuated_jnt_ids])

        home_key_name = config.get("home_key", "home")
        home_key_id = (
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, miniq_prefix + home_key_name)
            if home_key_name
            else -1
        )
        if home_key_id >= 0:
            init_q = jp.array(mj_model.key_qpos[home_key_id])
            default_pose = jp.array(mj_model.key_ctrl[home_key_id][actuator_ids])
        else:
            init_q = jp.array(mj_model.qpos0)
            default_pose = jp.array(
                mj_model.qpos0[mj_model.jnt_qposadr[actuated_jnt_ids]]
            )

        ctrlrange = jp.array(mj_model.actuator_ctrlrange[actuator_ids])

        joint_range = jp.array(mj_model.jnt_range[actuated_jnt_ids]).T
        lowers = jp.maximum(joint_range[0], ctrlrange[:, 0])
        uppers = jp.minimum(joint_range[1], ctrlrange[:, 1])
        factor = config.soft_joint_pos_limit_factor

        def sensor_adr(name: str) -> list[int]:
            sensor_id = mj_model.sensor(name).id
            adr = mj_model.sensor_adr[sensor_id]
            dim = mj_model.sensor_dim[sensor_id]
            return list(range(adr, adr + dim))

        return cls(
            miniq_prefix=miniq_prefix,
            base_body_id=base_body_id,
            imu_site_id=imu_site_id,
            feet_site_ids=feet_site_ids,
            nbody=mj_model.nbody,
            base_mass=mj_model.body_subtreemass[base_body_id],
            init_q=init_q,
            default_pose=default_pose,
            ctrlrange=ctrlrange,
            soft_lowers=lowers * factor,
            soft_uppers=uppers * factor,
            position_adr=jp.array(sensor_adr(miniq_prefix + POSITION_SENSOR)),
            upvector_adr=jp.array(sensor_adr(miniq_prefix + UPVECTOR_SENSOR)),
            gyro_adr=jp.array(sensor_adr(miniq_prefix + GYRO_SENSOR)),
            accelerometer_adr=jp.array(sensor_adr(miniq_prefix + ACCELEROMETER_SENSOR)),
            global_linvel_adr=jp.array(sensor_adr(miniq_prefix + GLOBAL_LINVEL_SENSOR)),
            global_angvel_adr=jp.array(sensor_adr(miniq_prefix + GLOBAL_ANGVEL_SENSOR)),
            local_linvel_adr=jp.array(sensor_adr(miniq_prefix + LOCAL_LINVEL_SENSOR)),
            foot_linvel_adr=jp.array(
                [sensor_adr(f"{miniq_prefix}{site}_global_linvel") for site in FEET_SITES]
            ),
            actuator_qposadr=actuator_qposadr,
            actuator_dofadr=actuator_dofadr,
        )

@dataclass(frozen=True)
class ModelSpec:
    """Model spec - once extracted constants and addresses"""

    miniq1: MiniQSpec
    miniq2: MiniQSpec

    # Site position sensors are the stable way to read these points from
    # data.sensordata. Sites themselves do not own qpos entries.
    push_t_site_pos_adr: jax.Array
    target_push_t_site_pos_adr: jax.Array

    # qpos entries for the T body's freejoint: xyz + quaternion.
    push_t_qposadr: jax.Array

    @classmethod
    def from_mj_model(
        cls,
        mj_model: mujoco.MjModel,
        config: config_dict.ConfigDict,
    ) -> "ModelSpec":
        def sensor_adr(name: str) -> list[int]:
            sensor_id = mj_model.sensor(name).id
            adr = mj_model.sensor_adr[sensor_id]
            dim = mj_model.sensor_dim[sensor_id]
            return list(range(adr, adr + dim))

        push_t_joint_id = mj_model.joint(PUSH_T_FREEJOINT).id
        push_t_qpos_start = mj_model.jnt_qposadr[push_t_joint_id]

        return cls(
            miniq1=MiniQSpec.from_mj_model(ROBOT_1_PREFIX, mj_model, config),
            miniq2=MiniQSpec.from_mj_model(ROBOT_2_PREFIX, mj_model, config),
            push_t_site_pos_adr=jp.array(
                [
                    sensor_adr(site + SITE_POSITION_SENSOR_SUFFIX)
                    for site in PUSH_T_SITES
                ]
            ),
            target_push_t_site_pos_adr=jp.array(
                [
                    sensor_adr(site + SITE_POSITION_SENSOR_SUFFIX)
                    for site in TARGET_PUSH_T_SITES
                ]
            ),
            push_t_qposadr=jp.arange(push_t_qpos_start, push_t_qpos_start + 7),
        )
