from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from miniq_pusht.model_spec import ModelSpec
from miniq_pusht.open_loop.observation import Observation


class Reward:
    def __init__(
        self,
        config: config_dict.ConfigDict,
        obs: Observation,
        spec: ModelSpec,
    ):
        self._config = config
        self._obs = obs
        self._spec = spec

    def get_reward_components(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        scales = self._config.reward_config.scales
        rewards = {
            "endpoint_error": self._reward_endpoint_error(data),
            "progress":       self._reward_progress(data, info),
            "useful_push":    self._reward_useful_push(data, info),
            "r1_push_site":   self._reward_push_site(data, self._spec.miniq1, self._spec.push_t_r1_push_site_adr),
            "r2_push_site":   self._reward_push_site(data, self._spec.miniq2, info["r2_push_site_adr"]),
            "action_rate":    jp.sum(jp.square(action - info["last_act"])),
            "action_penalty": jp.sum(jp.square(action)),
            "success":        info["is_success"].astype(jp.float32),
        }
        return {k: v * scales[k] for k, v in rewards.items()}

    def get_reward_value(
        self,
        reward_components: dict[str, jax.Array],
        dt: float,
    ) -> jax.Array:
        total = sum(reward_components.values()) * dt
        if self._config.reward_config.only_positive_rewards:
            return jp.clip(total, 0.0, 10000.0)
        return jp.array(total)

    # ------------------------------------------------------------------
    # Per-component helpers
    # ------------------------------------------------------------------

    def _reward_endpoint_error(self, data: mjx.Data) -> jax.Array:
        return jp.exp(-self.site_mean_error(data) / self._config.reward_config.endpoint_sigma)

    def _reward_progress(self, data: mjx.Data, info: dict) -> jax.Array:
        return info["prev_site_error"] - self.site_mean_error(data)

    def _reward_useful_push(self, data: mjx.Data, info: dict) -> jax.Array:
        # Translation: dot product of T velocity with direction toward target
        target_xy   = self._obs.target_xy(data)
        t_xy        = self._obs.t_xy(data)
        desired_dir = target_xy - t_xy
        desired_dir = desired_dir / (jp.linalg.norm(desired_dir) + 1e-6)
        r_translation = jp.dot(info["t_vel_estimate"][:2], desired_dir)

        # Rotation: reduction in absolute yaw error this step
        r_rotation = info["prev_abs_yaw_error"] - self.abs_yaw_error(data)

        # Only reward useful pushes when a robot is in contact with the T
        contact_r1 = self._obs.contact_summary(
            data, self._spec.miniq1.geom_ids, self._spec.push_t_geom_ids
        )[0]
        contact_r2 = self._obs.contact_summary(
            data, self._spec.miniq2.geom_ids, self._spec.push_t_geom_ids
        )[0]
        contact = jp.maximum(contact_r1, contact_r2)

        return contact * (0.5 * r_translation + 0.5 * r_rotation)

    def _reward_push_site(
        self,
        data: mjx.Data,
        robot,
        push_site_adr: jax.Array,
    ) -> jax.Array:
        robot_xy     = self._obs.robot_xy(data, robot)
        push_site_xy = data.sensordata[push_site_adr][:2]
        dist = jp.linalg.norm(robot_xy - push_site_xy)
        return jp.exp(-dist / self._config.reward_config.push_site_sigma)

    # ------------------------------------------------------------------
    # Shared scalar helpers used by both reward and env info bookkeeping
    # ------------------------------------------------------------------

    def site_mean_error(self, data: mjx.Data) -> jax.Array:
        site_err = self._obs.t_site_error(data)          # (3, 3)
        return jp.mean(jp.linalg.norm(site_err[:, :2], axis=1))

    def abs_yaw_error(self, data: mjx.Data) -> jax.Array:
        err = self._obs.t_yaw(data) - self._obs.target_yaw(data)
        return jp.abs(jp.arctan2(jp.sin(err), jp.cos(err)))
