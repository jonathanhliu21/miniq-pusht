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

