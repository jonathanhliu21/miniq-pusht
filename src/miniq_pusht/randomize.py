from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from miniq_pusht.model_spec import MiniQSpec, ModelSpec


GAIT_PARAM_SIZE = 11
GAIT_STRIDE = slice(0, 4)
GAIT_FREQ = 4
GAIT_YAW = 5
GAIT_LIFT = 6
GAIT_PHASE = slice(7, 11)


def _section(cfg: config_dict.ConfigDict, name: str) -> Any:
    return cfg.get(name, config_dict.ConfigDict())


def _value(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    return cfg.get(name, default)


def _enabled(cfg: Any, default: bool = True) -> bool:
    if cfg is None:
        return default
    return bool(cfg.get("enable", default))


def _uniform(
    rng: jax.Array,
    bounds: tuple[float, float] | list[float],
    shape: tuple[int, ...] = (),
) -> jax.Array:
    return jax.random.uniform(
        rng,
        shape=shape,
        minval=float(bounds[0]),
        maxval=float(bounds[1]),
    )


def _yaw_quat(yaw: jax.Array) -> jax.Array:
    half = 0.5 * yaw
    return jp.array([jp.cos(half), 0.0, 0.0, jp.sin(half)])


def _geom_ids(model: mjx.Model, spec: ModelSpec, names: tuple[str, ...]) -> jax.Array:
    # MJX models do not carry names. The Push-T geoms are the only geoms on the
    # push_t body, so this helper currently exists to make call sites explicit.
    del model, names
    return spec.push_t_geom_ids


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    cfg: config_dict.ConfigDict,
    spec: ModelSpec,
) -> tuple[mjx.Model, Any]:
    """Returns a Brax/MJX-compatible randomized model and in_axes tree.

    The returned model has one leading batch dimension on randomized fields.
    MiniQ 1 and MiniQ 2 are randomized from different PRNG keys. Episode-state
    randomization such as starts, qpos/qvel noise, and gait parameters is handled
    by :func:`randomize_reset`.
    """
    dr = _section(cfg, "domain_randomization")
    if not _enabled(dr):
        return model, jax.tree_util.tree_map(lambda _: None, model)

    t_cfg = _section(dr, "push_t")
    robot_cfg = _section(dr, "miniq")
    contact_cfg = _section(dr, "contact")
    start_cfg = _section(dr, "starts")
    push_t_enabled = _enabled(t_cfg)
    miniq_enabled = _enabled(robot_cfg)
    contact_enabled = _enabled(contact_cfg)
    target_enabled = _enabled(start_cfg)

    floor_geom_id = int(_value(contact_cfg, "floor_geom_id", spec.floor_geom_id))
    t_geom_ids = _geom_ids(model, spec, ("push_t_stem", "push_t_bar"))
    robot_geom_ids = jp.concatenate([spec.miniq1.geom_ids, spec.miniq2.geom_ids])

    @jax.vmap
    def rand_model(rng: jax.Array) -> tuple[jax.Array, ...]:
        rng, floor_key, t_key, r1_key, r2_key, target_key = jax.random.split(rng, 6)
        del rng

        geom_friction = model.geom_friction
        geom_size = model.geom_size
        geom_pos = model.geom_pos
        body_mass = model.body_mass
        body_inertia = model.body_inertia
        body_ipos = model.body_ipos
        body_pos = model.body_pos
        body_quat = model.body_quat
        pair_friction = model.pair_friction
        dof_frictionloss = model.dof_frictionloss
        dof_damping = model.dof_damping
        dof_armature = model.dof_armature
        qpos0 = model.qpos0
        actuator_gainprm = model.actuator_gainprm
        actuator_biasprm = model.actuator_biasprm
        actuator_gear = model.actuator_gear

        floor_friction_key, floor_height_key, robot_friction_key = jax.random.split(
            floor_key, 3
        )
        if contact_enabled:
            floor_friction = _uniform(
                floor_friction_key, _value(contact_cfg, "floor_friction", (0.3, 0.4))
            )
            geom_friction = geom_friction.at[floor_geom_id, 0].set(floor_friction)
            floor_height = _uniform(
                floor_height_key,
                _value(contact_cfg, "floor_height_offset", (-0.002, 0.002)),
            )
            geom_pos = geom_pos.at[floor_geom_id, 2].add(floor_height)

        t_key, mass_key, inertia_key, com_key, size_key, t_floor_key, t_robot_key = (
            jax.random.split(t_key, 7)
        )
        if push_t_enabled:
            t_body = spec.push_t_body_id
            body_mass = body_mass.at[t_body].set(
                body_mass[t_body]
                * _uniform(mass_key, _value(t_cfg, "mass_scale", (0.7, 1.3)))
            )
            body_inertia = body_inertia.at[t_body].set(
                body_inertia[t_body]
                * _uniform(inertia_key, _value(t_cfg, "inertia_scale", (0.7, 1.5)))
            )
            t_com_xy = _uniform(
                com_key, _value(t_cfg, "com_xy_jitter", (-0.008, 0.008)), (2,)
            )
            body_ipos = body_ipos.at[t_body, :2].add(t_com_xy)
            t_size = _uniform(size_key, _value(t_cfg, "geom_size_scale", (0.95, 1.05)))
            geom_size = geom_size.at[t_geom_ids].set(geom_size[t_geom_ids] * t_size)

        if contact_enabled:
            t_floor_friction = _uniform(
                t_floor_key, _value(contact_cfg, "t_floor_friction", (0.25, 0.4))
            )
            geom_friction = geom_friction.at[t_geom_ids, 0].set(t_floor_friction)
            pair_friction = pair_friction.at[spec.t_floor_pair_ids, 0].set(
                jp.full((spec.t_floor_pair_ids.shape[0],), t_floor_friction)
            )

            robot_contact_friction = _uniform(
                robot_friction_key,
                _value(contact_cfg, "robot_contact_friction", (0.5, 1.2)),
            )
            geom_friction = geom_friction.at[robot_geom_ids, 0].set(
                robot_contact_friction
            )

        def rand_robot(
            rng: jax.Array,
            robot: MiniQSpec,
            geom_friction: jax.Array,
            body_mass: jax.Array,
            body_ipos: jax.Array,
            dof_frictionloss: jax.Array,
            dof_damping: jax.Array,
            dof_armature: jax.Array,
            qpos0: jax.Array,
            actuator_gainprm: jax.Array,
            actuator_biasprm: jax.Array,
            actuator_gear: jax.Array,
        ) -> tuple[jax.Array, ...]:
            keys = jax.random.split(rng, 9)
            base_body = robot.base_body_id
            body_mass = body_mass.at[base_body].set(
                body_mass[base_body]
                * _uniform(keys[0], _value(robot_cfg, "base_mass_scale", (0.9, 1.1)))
            )
            com_xy = _uniform(
                keys[1], _value(robot_cfg, "base_com_xy_jitter", (-0.004, 0.004)), (2,)
            )
            com_z = _uniform(
                keys[2], _value(robot_cfg, "base_com_z_jitter", (-0.003, 0.003))
            )
            body_ipos = body_ipos.at[base_body, :2].add(com_xy)
            body_ipos = body_ipos.at[base_body, 2].add(com_z)

            dofs = robot.actuator_dofadr
            actuators = robot.actuator_ids
            dof_frictionloss = dof_frictionloss.at[dofs].set(
                dof_frictionloss[dofs]
                * _uniform(
                    keys[3],
                    _value(robot_cfg, "joint_frictionloss_scale", (0.5, 2.0)),
                    (dofs.shape[0],),
                )
            )
            dof_armature = dof_armature.at[dofs].set(
                dof_armature[dofs]
                * _uniform(
                    keys[4],
                    _value(robot_cfg, "armature_scale", (0.7, 1.5)),
                    (dofs.shape[0],),
                )
            )
            actuator_gear = actuator_gear.at[actuators, 0].set(
                actuator_gear[actuators, 0]
                * _uniform(
                    keys[5],
                    _value(robot_cfg, "motor_strength_scale", (0.75, 1.25)),
                    (actuators.shape[0],),
                )
            )
            kp_scale = _uniform(
                keys[6],
                _value(robot_cfg, "kp_scale", (0.8, 1.2)),
                (actuators.shape[0],),
            )
            actuator_gainprm = actuator_gainprm.at[actuators, 0].set(
                actuator_gainprm[actuators, 0] * kp_scale
            )
            actuator_biasprm = actuator_biasprm.at[actuators, 1].set(
                actuator_biasprm[actuators, 1] * kp_scale
            )
            dof_damping = dof_damping.at[dofs].set(
                dof_damping[dofs]
                * _uniform(
                    keys[7],
                    _value(robot_cfg, "kd_scale", (0.5, 1.5)),
                    (dofs.shape[0],),
                )
            )
            qpos0 = qpos0.at[robot.actuator_qposadr].add(
                _uniform(
                    keys[8],
                    _value(robot_cfg, "default_joint_offset", (-0.08, 0.08)),
                    (robot.actuator_qposadr.shape[0],),
                )
            )
            geom_friction = geom_friction.at[robot.geom_ids, 0].set(
                _uniform(
                    keys[0],
                    _value(contact_cfg, "robot_contact_friction", (0.5, 1.2)),
                )
            )
            return (
                geom_friction,
                body_mass,
                body_ipos,
                dof_frictionloss,
                dof_damping,
                dof_armature,
                qpos0,
                actuator_gainprm,
                actuator_biasprm,
                actuator_gear,
            )

        if miniq_enabled:
            (
                geom_friction,
                body_mass,
                body_ipos,
                dof_frictionloss,
                dof_damping,
                dof_armature,
                qpos0,
                actuator_gainprm,
                actuator_biasprm,
                actuator_gear,
            ) = rand_robot(
                r1_key,
                spec.miniq1,
                geom_friction,
                body_mass,
                body_ipos,
                dof_frictionloss,
                dof_damping,
                dof_armature,
                qpos0,
                actuator_gainprm,
                actuator_biasprm,
                actuator_gear,
            )
            (
                geom_friction,
                body_mass,
                body_ipos,
                dof_frictionloss,
                dof_damping,
                dof_armature,
                qpos0,
                actuator_gainprm,
                actuator_biasprm,
                actuator_gear,
            ) = rand_robot(
                r2_key,
                spec.miniq2,
                geom_friction,
                body_mass,
                body_ipos,
                dof_frictionloss,
                dof_damping,
                dof_armature,
                qpos0,
                actuator_gainprm,
                actuator_biasprm,
                actuator_gear,
            )

        if target_enabled:
            target_key, xy_key, yaw_key = jax.random.split(target_key, 3)
            del target_key
            target_xy = _uniform(
                xy_key, _value(start_cfg, "target_xy_jitter", (-0.3, 0.3)), (2,)
            )
            target_yaw = _uniform(
                yaw_key, _value(start_cfg, "target_yaw", (-math.pi, math.pi))
            )
            target_body = spec.target_push_t_body_id
            body_pos = body_pos.at[target_body, :2].add(target_xy)
            body_quat = body_quat.at[target_body].set(_yaw_quat(target_yaw))

        return (
            geom_friction,
            geom_size,
            geom_pos,
            body_mass,
            body_inertia,
            body_ipos,
            body_pos,
            body_quat,
            pair_friction,
            dof_frictionloss,
            dof_damping,
            dof_armature,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
            actuator_gear,
        )

    (
        geom_friction,
        geom_size,
        geom_pos,
        body_mass,
        body_inertia,
        body_ipos,
        body_pos,
        body_quat,
        pair_friction,
        dof_frictionloss,
        dof_damping,
        dof_armature,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
        actuator_gear,
    ) = rand_model(rng)

    overrides = {
        "geom_friction": geom_friction,
        "geom_size": geom_size,
        "geom_pos": geom_pos,
        "body_mass": body_mass,
        "body_inertia": body_inertia,
        "body_ipos": body_ipos,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "pair_friction": pair_friction,
        "dof_frictionloss": dof_frictionloss,
        "dof_damping": dof_damping,
        "dof_armature": dof_armature,
        "qpos0": qpos0,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "actuator_gear": actuator_gear,
    }
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({name: 0 for name in overrides})
    return model.tree_replace(overrides), in_axes


def randomize_gait_params(
    rng: jax.Array,
    cfg: config_dict.ConfigDict,
) -> jax.Array:
    """Samples per-robot gait randomization arrays.

    Output shape is ``(2, GAIT_PARAM_SIZE)``. Rows are MiniQ 1 and MiniQ 2.
    """
    dr = _section(cfg, "domain_randomization")
    if not _enabled(dr):
        return jp.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jp.float32,
        )
    gait_cfg = _section(dr, "gait")
    if not _enabled(gait_cfg):
        return jp.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jp.float32,
        )

    @jax.vmap
    def rand_one(rng: jax.Array) -> jax.Array:
        keys = jax.random.split(rng, 8)
        right = _uniform(keys[0], _value(gait_cfg, "left_right_stride_scale", (0.85, 1.15)))
        left = _uniform(keys[1], _value(gait_cfg, "left_right_stride_scale", (0.85, 1.15)))
        front = _uniform(keys[2], _value(gait_cfg, "front_rear_stride_scale", (0.90, 1.10)))
        rear = _uniform(keys[3], _value(gait_cfg, "front_rear_stride_scale", (0.90, 1.10)))
        stride_scales = jp.array([right * front, left * front, right * rear, left * rear])
        freq_scale = _uniform(keys[4], _value(gait_cfg, "frequency_scale", (0.85, 1.15)))
        yaw_scale = _uniform(keys[5], _value(gait_cfg, "yaw_stride_gain_scale", (0.6, 1.1)))
        lift_scale = _uniform(keys[6], _value(gait_cfg, "lift_height_scale", (0.8, 1.2)))
        phase_offsets = _uniform(
            keys[7],
            _value(gait_cfg, "phase_offset_error", (-0.05, 0.05)),
            (4,),
        )
        return jp.concatenate(
            [
                stride_scales,
                jp.array([freq_scale, yaw_scale, lift_scale]),
                phase_offsets,
            ]
        )

    return rand_one(jax.random.split(rng, 2))


def randomize_reset(
    qpos: jax.Array,
    qvel: jax.Array,
    rng: jax.Array,
    cfg: config_dict.ConfigDict,
    spec: ModelSpec,
) -> tuple[jax.Array, jax.Array]:
    """Randomizes free-body starts and initial joint state for one episode."""
    dr = _section(cfg, "domain_randomization")
    if not _enabled(dr):
        return qpos, qvel
    start_cfg = _section(dr, "starts")
    if not _enabled(start_cfg):
        return qpos, qvel
    t_key, r1_key, r2_key, noise_key = jax.random.split(rng, 4)

    def set_free_pose(
        qpos: jax.Array,
        adr: jax.Array,
        xy_delta: jax.Array,
        yaw_delta: jax.Array,
    ) -> jax.Array:
        qpos = qpos.at[adr[:2]].add(xy_delta)
        yaw0 = 2.0 * jp.arctan2(qpos[adr[6]], qpos[adr[3]])
        qpos = qpos.at[adr[3:7]].set(_yaw_quat(yaw0 + yaw_delta))
        return qpos

    t_xy_key, t_yaw_key = jax.random.split(t_key)
    t_xy = _uniform(t_xy_key, _value(start_cfg, "push_t_xy_jitter", (-0.3, 0.3)), (2,))
    t_yaw = _uniform(t_yaw_key, _value(start_cfg, "push_t_yaw", (-math.pi, math.pi)))
    qpos = set_free_pose(qpos, spec.push_t_qposadr, t_xy, t_yaw)

    r1_xy_key, r1_yaw_key = jax.random.split(r1_key)
    r2_xy_key, r2_yaw_key = jax.random.split(r2_key)
    r1_xy = _uniform(r1_xy_key, _value(start_cfg, "robot_xy_jitter", (-0.1, 0.1)), (2,))
    r2_xy = _uniform(r2_xy_key, _value(start_cfg, "robot_xy_jitter", (-0.1, 0.1)), (2,))
    r1_nominal = qpos[spec.miniq1.base_qposadr[:2]] + r1_xy
    r2_nominal = qpos[spec.miniq2.base_qposadr[:2]] + r2_xy
    delta = r2_nominal - r1_nominal
    dist = jp.linalg.norm(delta)
    min_dist = float(_value(start_cfg, "min_robot_start_distance", 0.16))
    direction = delta / jp.maximum(dist, 1.0e-6)
    correction = 0.5 * jp.maximum(min_dist - dist, 0.0) * direction
    r1_xy = r1_xy - correction
    r2_xy = r2_xy + correction

    # Enforce minimum robot-to-T distance: the T may have jittered into a
    # robot's home region, so push each robot away from the T if needed.
    t_pos = qpos[spec.push_t_qposadr[:2]]
    min_robot_t_dist = float(_value(start_cfg, "min_robot_t_distance", 0.27))

    r1_pos = qpos[spec.miniq1.base_qposadr[:2]] + r1_xy
    delta_r1_t = r1_pos - t_pos
    dist_r1_t = jp.linalg.norm(delta_r1_t)
    dir_r1_t = delta_r1_t / jp.maximum(dist_r1_t, 1e-6)
    r1_xy = r1_xy + jp.maximum(min_robot_t_dist - dist_r1_t, 0.0) * dir_r1_t

    r2_pos = qpos[spec.miniq2.base_qposadr[:2]] + r2_xy
    delta_r2_t = r2_pos - t_pos
    dist_r2_t = jp.linalg.norm(delta_r2_t)
    dir_r2_t = delta_r2_t / jp.maximum(dist_r2_t, 1e-6)
    r2_xy = r2_xy + jp.maximum(min_robot_t_dist - dist_r2_t, 0.0) * dir_r2_t

    r1_yaw = _uniform(r1_yaw_key, _value(start_cfg, "robot_yaw", (-math.pi / 6, math.pi / 6)))
    r2_yaw = _uniform(r2_yaw_key, _value(start_cfg, "robot_yaw", (-math.pi / 6, math.pi / 6)))
    qpos = set_free_pose(qpos, spec.miniq1.base_qposadr, r1_xy, r1_yaw)
    qpos = set_free_pose(qpos, spec.miniq2.base_qposadr, r2_xy, r2_yaw)

    jq_key, qv_key = jax.random.split(noise_key)
    joint_noise = _uniform(
        jq_key,
        _value(start_cfg, "initial_qpos_noise", (-0.05, 0.05)),
        (spec.miniq1.actuator_qposadr.shape[0] + spec.miniq2.actuator_qposadr.shape[0],),
    )
    joint_adrs = jp.concatenate([spec.miniq1.actuator_qposadr, spec.miniq2.actuator_qposadr])
    qpos = qpos.at[joint_adrs].add(joint_noise)

    qvel_noise = _uniform(
        qv_key,
        _value(start_cfg, "initial_qvel_noise", (-0.02, 0.02)),
        (qvel.shape[0],),
    )
    return qpos, qvel + qvel_noise
