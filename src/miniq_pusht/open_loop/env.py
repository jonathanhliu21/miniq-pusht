from __future__ import annotations

from pathlib import Path
from typing import Any


def _patch_warp_mjx_compat() -> None:
    """Backfills Warp APIs the bundled mujoco_warp expects.

    1. `wp.types.warp_type_to_np_dtype` was dropped in Warp 1.13; restore
       it so mjx's io module can still build dtype lookups.
    2. `wp.math.sqrt` was moved to the top-level `wp.sqrt` in newer Warp
       releases; the bundled mujoco_warp solver still calls it via the old
       path. Alias it back.

    No-op if Warp isn't installed or the attributes are already present.
    """
    try:
        import warp as _wp
        import numpy as _np
    except ImportError:
        return
    if not hasattr(_wp.types, "warp_type_to_np_dtype"):
        _wp.types.warp_type_to_np_dtype = {
            _wp.float16: _np.float16,
            _wp.float32: _np.float32,
            _wp.float64: _np.float64,
            _wp.int8: _np.int8,
            _wp.int16: _np.int16,
            _wp.int32: _np.int32,
            _wp.int64: _np.int64,
            _wp.uint8: _np.uint8,
            _wp.uint16: _np.uint16,
            _wp.uint32: _np.uint32,
            _wp.uint64: _np.uint64,
            _wp.bool: _np.bool_,
        }
    if hasattr(_wp, "math") and not hasattr(_wp.math, "sqrt"):
        _wp.math.sqrt = _wp.sqrt

    try:
        from warp.jax_experimental.ffi import JaxCallableGraphMode as _GraphMode
        from mujoco.mjx import warp as _mjxw
    except ImportError:
        return
    if (
        hasattr(_mjxw, "types")
        and getattr(_mjxw.types, "GraphMode", None) is int
        and hasattr(_GraphMode, "WARP")
    ):
        _mjxw.types.GraphMode = _GraphMode


_patch_warp_mjx_compat()

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
import yaml

from miniq_pusht.model_spec import ModelSpec
from miniq_pusht.open_loop.observation import Observation
from miniq_pusht.open_loop.open_loop import GaitGenerator
from miniq_pusht.open_loop.reward import Reward
from miniq_pusht.randomize import (
    domain_randomize as randomize_model,
    randomize_gait_params,
    randomize_reset,
)


def append_yaml_to_config(
    file_name: str, config: config_dict.ConfigDict
) -> config_dict.ConfigDict:
    """Reads a YAML file and appends/merges its contents into an ml_collections ConfigDict.

    Args:
        file_name: The path to the YAML file.
        config: The existing ml_collections.ConfigDict instance.

    Returns:
        The updated ConfigDict instance.
    """
    try:
        with open(file_name, "r") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data:
            # ml_collections.ConfigDict.update() recursively merges nested dictionaries
            config.update(yaml_data)

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

    return config


class Env(mjx_env.MjxEnv):
    """Environment setup with open loop locomotion control"""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__(config, config_overrides)

        # set up appended config
        extra_configs = self._config.env.extra_configs
        self._config = append_yaml_to_config(extra_configs, self._config)

        # set up simulator
        self._xml_path = str(Path(self._config.mjcf.path))
        self._episode_length = self._config.episode_length
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

        # per-class gains
        kp_by_class = {
            "hip": self._config.Kp_hip,
            "knee": self._config.Kp_knee,
        }
        kd_by_class = {
            "hip": self._config.Kd_hip,
            "knee": self._config.Kd_knee,
        }

        def _classify_joint_name(name: str) -> str | None:
            if name.endswith("_joint0"):
                return "hip"
            if name.endswith("_knee_motor"):
                return "knee"
            return None  # bearings + freejoint — leave MJCF defaults alone

        for aid in range(self._mj_model.nu):
            jid = int(self._mj_model.actuator_trnid[aid, 0])
            cls = _classify_joint_name(self._mj_model.joint(jid).name)
            if cls is None:
                continue
            self._mj_model.actuator_gainprm[aid, 0] = kp_by_class[cls]
            self._mj_model.actuator_biasprm[aid, 1] = -kp_by_class[cls]

        for jid in range(self._mj_model.njnt):
            cls = _classify_joint_name(self._mj_model.joint(jid).name)
            if cls is None:
                continue
            dofadr = int(self._mj_model.jnt_dofadr[jid])
            self._mj_model.dof_damping[dofadr] = kd_by_class[cls]

        # set viewer dimensions
        self._mj_model.vis.global_.offwidth = 1280
        self._mj_model.vis.global_.offheight = 720

        # static model-derived constants (IDs, default pose, sensor addresses, ...)
        self._spec = ModelSpec.from_mj_model(self._mj_model, self._config)
        self._setup_inter_robot_collisions()

        # put model onto mjx
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._obs_builder = Observation(self._spec, self._config)
        self._reward = Reward(self._config, self._obs_builder, self._spec)
        self._gait1 = GaitGenerator(
            self._spec.miniq1.default_pose, self._spec.miniq1.ctrlrange
        )
        self._gait2 = GaitGenerator(
            self._spec.miniq2.default_pose, self._spec.miniq2.ctrlrange
        )

        # Latency config
        latency_cfg = self._config.get("latency", {})
        action_lat = latency_cfg.get("action_latency", {})
        obs_lat = latency_cfg.get("observation_latency", {})

        self._action_latency_enabled = bool(action_lat.get("enabled", False))
        self._action_history_size = int(action_lat.get("action_history_size", 16))
        self._action_delay_mean = float(action_lat.get("action_delay_mean", 3.0))
        self._action_delay_std = float(action_lat.get("action_delay_std", 3.0))

        self._obs_latency_enabled = bool(obs_lat.get("enabled", False))
        self._obs_history_size = int(obs_lat.get("obs_history_size", 16))
        self._obs_delay_mean = float(obs_lat.get("obs_delay_mean", 1.0))
        self._obs_delay_std = float(obs_lat.get("obs_delay_std", 0.5))

    def _setup_inter_robot_collisions(self) -> None:
        # make sure robot1 and robot2 collide with each other but r1 and r1 don't
        for gid in self._spec.miniq1.geom_ids:
            self._mj_model.geom_contype[int(gid)] = 2
            self._mj_model.geom_conaffinity[int(gid)] = 5
        for gid in self._spec.miniq2.geom_ids:
            self._mj_model.geom_contype[int(gid)] = 4
            self._mj_model.geom_conaffinity[int(gid)] = 3

    def domain_randomize(
        self,
        model: mjx.Model,
        rng: jax.Array,
    ) -> tuple[mjx.Model, Any]:
        """Brax/MJX domain randomization callback."""
        return randomize_model(model, rng, self._config, self._spec)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 4

    @property
    def observation_size(self) -> dict[str, int]:
        return self._obs_builder.observation_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, reset_rng, gait_rng = jax.random.split(rng, 3)
        qpos = self._spec.init_q
        qvel = jp.zeros(self.mjx_model.nv)
        randomization_cfg = self._config.get("domain_randomization", {})
        randomization_enabled = randomization_cfg.get("enable", True)
        starts_enabled = randomization_cfg.get("starts", {}).get("enable", True)
        gait_enabled = randomization_cfg.get("gait", {}).get("enable", True)
        if randomization_enabled and starts_enabled:
            qpos, qvel = randomize_reset(
                qpos, qvel, reset_rng, self._config, self._spec
            )

        # put data onto model
        data = mjx.make_data(
            self._mj_model,
            impl=self._config.impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        default_ctrl = jp.concatenate(
            [self._spec.miniq1.default_pose, self._spec.miniq2.default_pose]
        )
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=default_ctrl)
        data = mjx.forward(self.mjx_model, data)

        # build info
        info = {
            "rng": rng,
            "last_action": jp.zeros(4),
            "last_act": jp.zeros(4),
            "gait_phase": jp.zeros(2),
            "gait_params": (
                randomize_gait_params(gait_rng, self._config)
                if randomization_enabled and gait_enabled
                else jp.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=jp.float32,
                )
            ),
            "gait_frequency": jp.zeros(2),
            "t_vel_estimate": jp.zeros(3),
        }
        obs = self._obs_builder.get_obs(data, info)

        # latency buffers
        info["action_buffer"] = jp.zeros((self._action_history_size, self.action_size))
        info["obs_state_buffer"] = jp.tile(
            obs["state"][None], (self._obs_history_size, 1)
        )

        # termination counters
        info["success_counter"] = jp.zeros((), dtype=jp.int32)
        info["robot_robot_contact_steps"] = jp.zeros((), dtype=jp.int32)
        info["step_count"] = jp.zeros((), dtype=jp.int32)
        info["is_success"] = jp.zeros((), dtype=jp.bool_)

        # assign robot 2 to whichever push site is closer at episode start
        r2_xy = self._obs_builder.robot_xy(data, self._spec.miniq2)
        push_l_xy = data.sensordata[self._spec.push_t_r2_push_site_adr_l][:2]
        push_r_xy = data.sensordata[self._spec.push_t_r2_push_site_adr_r][:2]
        d_l = jp.linalg.norm(r2_xy - push_l_xy)
        d_r = jp.linalg.norm(r2_xy - push_r_xy)
        info["r2_push_site_adr"] = jp.where(
            d_l <= d_r,
            self._spec.push_t_r2_push_site_adr_l,
            self._spec.push_t_r2_push_site_adr_r,
        )

        # reward bookkeeping: previous-step values for progress/useful-push rewards
        info["prev_site_error"] = self._reward.site_mean_error(data)
        info["prev_abs_yaw_error"] = self._reward.abs_yaw_error(data)

        # build metrics
        metrics = {}
        reward, done = jp.zeros(()), jp.zeros(())
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.asarray(action).reshape((self.action_size,))
        action_scale = self._config.get("action_scale", 1.0)
        scaled_action = jp.clip(action * action_scale, -1.0, 1.0)

        info = dict(state.info)

        # apply action latency
        action_buffer = state.info["action_buffer"]
        new_action_buffer = jp.concatenate(
            [scaled_action[None], action_buffer[:-1]], axis=0
        )
        if self._action_latency_enabled:
            info["rng"], delay_rng = jax.random.split(info["rng"])
            raw_delay = (
                self._action_delay_mean
                + self._action_delay_std * jax.random.normal(delay_rng)
            )
            action_delay = jp.clip(
                jp.round(raw_delay).astype(jp.int32), 0, self._action_history_size - 1
            )
            effective_action = jax.lax.dynamic_slice_in_dim(
                new_action_buffer, action_delay, 1, axis=0
            )[0]
        else:
            action_delay = jp.array(0, dtype=jp.int32)
            effective_action = scaled_action

        if self._obs_latency_enabled:
            info["rng"], obs_delay_rng = jax.random.split(info["rng"])
            raw_delay = self._obs_delay_mean + self._obs_delay_std * jax.random.normal(
                obs_delay_rng
            )
            obs_delay = jp.clip(
                jp.round(raw_delay).astype(jp.int32), 0, self._obs_history_size - 1
            )
        else:
            obs_delay = jp.array(0, dtype=jp.int32)

        prev_t_xy = self._obs_builder.t_xy(state.data)
        prev_t_yaw = self._obs_builder.t_yaw(state.data)

        gait_steps_per_ctrl = self._config.get("gait_steps_per_ctrl", 1)
        sim_steps_per_gait = self.n_substeps // gait_steps_per_ctrl

        def gait_step(
            carry: tuple[mjx.Data, jax.Array],
            _: None,
        ) -> tuple[tuple[mjx.Data, jax.Array], None]:
            data, phase = carry
            gait_params = state.info["gait_params"]
            phase1, ctrl1 = self._gait1.step_randomized(
                phase[0], effective_action[:2], gait_params[0]
            )
            phase2, ctrl2 = self._gait2.step_randomized(
                phase[1], effective_action[2:], gait_params[1]
            )
            phase = jp.array([phase1, phase2])
            ctrl = jp.concatenate([ctrl1, ctrl2])
            data = mjx_env.step(self.mjx_model, data, ctrl, sim_steps_per_gait)
            return (data, phase), None

        (data, phase), _ = jax.lax.scan(
            gait_step,
            (state.data, state.info["gait_phase"]),
            None,
            length=gait_steps_per_ctrl,
        )

        t_xy = self._obs_builder.t_xy(data)
        t_yaw = self._obs_builder.t_yaw(data)
        yaw_delta = jp.arctan2(jp.sin(t_yaw - prev_t_yaw), jp.cos(t_yaw - prev_t_yaw))
        t_vel_estimate = jp.array(
            [
                (t_xy[0] - prev_t_xy[0]) / self.dt,
                (t_xy[1] - prev_t_xy[1]) / self.dt,
                yaw_delta / self.dt,
            ]
        )

        info.update(
            {
                "last_action": effective_action,
                "last_act": effective_action,
                "gait_phase": phase,
                "gait_frequency": jp.array(
                    [
                        self._gait1.frequency_with_scale(
                            effective_action[:2], state.info["gait_params"][0, 4]
                        ),
                        self._gait2.frequency_with_scale(
                            effective_action[2:], state.info["gait_params"][1, 4]
                        ),
                    ]
                ),
                "t_vel_estimate": t_vel_estimate,
                "action_buffer": new_action_buffer,
                "action_delay": action_delay,
                "obs_delay": obs_delay,
                "step_count": state.info["step_count"] + 1,
            }
        )
        obs = self._obs_builder.get_obs(data, info)

        # observation latency
        obs_state_buffer = state.info["obs_state_buffer"]
        new_obs_state_buffer = jp.concatenate(
            [obs["state"][None], obs_state_buffer[:-1]], axis=0
        )
        if self._obs_latency_enabled:
            delayed_state = jax.lax.dynamic_slice_in_dim(
                new_obs_state_buffer, obs_delay, 1, axis=0
            )[0]
            obs = {**obs, "state": delayed_state}
        info["obs_state_buffer"] = new_obs_state_buffer

        done = self._get_termination(data, info)  # also sets info["is_success"]
        components = self._reward.get_reward_components(data, effective_action, info, done)
        reward = self._reward.get_reward_value(components, self.dt)

        # update prev_ values for the next step's progress/useful-push rewards
        info["prev_site_error"] = self._reward.site_mean_error(data)
        info["prev_abs_yaw_error"] = self._reward.abs_yaw_error(data)

        return mjx_env.State(data, obs, reward, done, state.metrics, info)

    # ------------------------------------------------------------------
    # Termination helpers
    # ------------------------------------------------------------------

    def _check_success(self, data: mjx.Data, info: dict) -> jax.Array:
        site_err = self._obs_builder.t_site_error(data)  # (3, 3)
        xy_err = jp.linalg.norm(site_err[:, :2], axis=1)  # (3,)
        all_close = jp.all(xy_err < self._config.termination.success_site_error_m)
        new_count = jp.where(
            all_close, info["success_counter"] + 1, jp.zeros((), jp.int32)
        )
        info["success_counter"] = new_count
        is_success = new_count >= self._config.termination.success_consecutive_steps
        info["is_success"] = is_success
        return is_success

    def _check_robot_fallen(self, data: mjx.Data) -> jax.Array:
        min_h = self._config.termination.min_base_height_m
        min_up = self._config.termination.min_upvector_z

        def fallen(robot) -> jax.Array:
            base_z = data.qpos[robot.base_qposadr][2]
            up_z = data.sensordata[robot.upvector_adr][2]
            return (base_z < min_h) | (up_z < min_up)

        return fallen(self._spec.miniq1) | fallen(self._spec.miniq2)

    def _check_robot_robot_collision(self, data: mjx.Data, info: dict) -> jax.Array:
        summary = self._obs_builder.contact_summary(
            data, self._spec.miniq1.geom_ids, self._spec.miniq2.geom_ids
        )
        in_contact = summary[0] > 0.5
        new_steps = jp.where(
            in_contact,
            info["robot_robot_contact_steps"] + 1,
            jp.zeros((), jp.int32),
        )
        info["robot_robot_contact_steps"] = new_steps
        return (summary[1] > self._config.termination.robot_robot_force_n) | (
            new_steps >= self._config.termination.robot_robot_contact_steps
        )

    def _check_robot_on_t(self, data: mjx.Data) -> jax.Array:
        threshold = self._config.termination.robot_on_t_height_m
        z1 = data.qpos[self._spec.miniq1.base_qposadr][2]
        z2 = data.qpos[self._spec.miniq2.base_qposadr][2]
        return (z1 > threshold) | (z2 > threshold)

    def _check_out_of_bounds(self, data: mjx.Data) -> jax.Array:
        half_w = self._config.termination.arena_half_width_m

        def oob(xy: jax.Array) -> jax.Array:
            return jp.any(jp.abs(xy) > half_w)

        r1_xy = data.qpos[self._spec.miniq1.base_qposadr[:2]]
        r2_xy = data.qpos[self._spec.miniq2.base_qposadr[:2]]
        t_xy = data.qpos[self._spec.push_t_qposadr[:2]]
        return oob(r1_xy) | oob(r2_xy) | oob(t_xy)

    def _check_t_tilt(self, data: mjx.Data) -> jax.Array:
        # qpos order: x y z qw qx qy qz  →  up-vector z = 1 - 2(qx²+qy²)
        t_q = data.qpos[self._spec.push_t_qposadr]
        up_z = 1.0 - 2.0 * (t_q[4] * t_q[4] + t_q[5] * t_q[5])
        return up_z < self._config.termination.t_tilt_cos_min

    def _get_termination(self, data: mjx.Data, info: dict) -> jax.Array:
        term = (
            self._check_success(data, info)
            | self._check_robot_fallen(data)
            | self._check_robot_robot_collision(data, info)
            | self._check_robot_on_t(data)
            | self._check_out_of_bounds(data)
            | self._check_t_tilt(data)
            | (info["step_count"] >= self._config.episode_length)
        )
        return term.astype(jp.float32)
