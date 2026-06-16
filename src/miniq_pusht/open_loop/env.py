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


_patch_warp_mjx_compat()

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

from miniq_pusht.model_spec import ModelSpec
from miniq_pusht.open_loop.observation import Observation
from miniq_pusht.open_loop.open_loop import GaitGenerator

class Env(mjx_env.MjxEnv):
    """Environment setup with open loop locomotion control"""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__(config, config_overrides)

        # set up simulator
        self._xml_path = str(Path(self._config.mjcf.path))
        self._episode_length = self._config.episode_length
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

        # per-class gains
        kp_by_class = {
            "hip":  self._config.Kp_hip,
            "knee": self._config.Kp_knee,
        }
        kd_by_class = {
            "hip":  self._config.Kd_hip,
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

        # put model onto mjx
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # static model-derived constants (IDs, default pose, sensor addresses, ...)
        self._spec = ModelSpec.from_mj_model(self._mj_model, self._config)
        self._obs_builder = Observation(self._spec, self._config)
        self._gait1 = GaitGenerator(self._spec.miniq1.default_pose, self._spec.miniq1.ctrlrange)
        self._gait2 = GaitGenerator(self._spec.miniq2.default_pose, self._spec.miniq2.ctrlrange)

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
        qpos = self._spec.init_q
        qvel = jp.zeros(self.mjx_model.nv)

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
            "gait_frequency": jp.zeros(2),
            "t_vel_estimate": jp.zeros(3),
        }
        obs = self._obs_builder.get_obs(data, info)

        # build metrics
        metrics = {}
        reward, done = jp.zeros(()), jp.zeros(())
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.asarray(action).reshape((self.action_size,))
        action_scale = self._config.get("action_scale", 1.0)
        scaled_action = jp.clip(action * action_scale, -1.0, 1.0)

        prev_t_xy = self._obs_builder.t_xy(state.data)
        prev_t_yaw = self._obs_builder.t_yaw(state.data)

        gait_steps_per_ctrl = self._config.get("gait_steps_per_ctrl", 1)
        sim_steps_per_gait = self.n_substeps // gait_steps_per_ctrl

        def gait_step(
            carry: tuple[mjx.Data, jax.Array],
            _: None,
        ) -> tuple[tuple[mjx.Data, jax.Array], None]:
            data, phase = carry
            phase1, ctrl1 = self._gait1.step(phase[0], scaled_action[:2])
            phase2, ctrl2 = self._gait2.step(phase[1], scaled_action[2:])
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

        info = dict(state.info)
        info.update(
            {
                "last_action": scaled_action,
                "last_act": scaled_action,
                "gait_phase": phase,
                "gait_frequency": jp.array(
                    [
                        self._gait1.frequency(scaled_action[:2]),
                        self._gait2.frequency(scaled_action[2:]),
                    ]
                ),
                "t_vel_estimate": t_vel_estimate,
            }
        )
        obs = self._obs_builder.get_obs(data, info)

        reward, done = jp.zeros(()), jp.zeros(())
        return mjx_env.State(data, obs, reward, done, state.metrics, info)
