from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from miniq_pusht.model_spec import MiniQSpec, ModelSpec


class Observation:
    """Builds policy and critic observations for the MiniQ Push-T task."""

    def __init__(
        self,
        spec: ModelSpec,
        config: config_dict.ConfigDict,
    ) -> None:
        self._spec = spec
        self._config = config

    def get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        r1_rel_xy = self.robot_xy(data, self._spec.miniq1) - self.t_xy(data)
        r2_rel_xy = self.robot_xy(data, self._spec.miniq2) - self.t_xy(data)
        r1_rel_yaw = self.robot_yaw(data, self._spec.miniq1) - self.t_yaw(data)
        r2_rel_yaw = self.robot_yaw(data, self._spec.miniq2) - self.t_yaw(data)
        t_rel_target_xy = self.t_xy(data) - self.target_xy(data)
        t_yaw_error = self.t_yaw(data) - self.target_yaw(data)
        t_site_xy_error = jp.linalg.norm(self.t_site_error(data)[:, :2], axis=1)
        t_vel_estimate = info.get("t_vel_estimate", self.t_planar_vel(data))
        last_action = info.get("last_act", jp.zeros(4))

        actor_state = jp.hstack(
            [
                r1_rel_xy,
                r2_rel_xy,
                self.sin_cos(r1_rel_yaw),
                self.sin_cos(r2_rel_yaw),
                t_rel_target_xy,
                self.sin_cos(t_yaw_error),
                t_site_xy_error,
                t_vel_estimate,
                last_action,
            ]
        )
        state = self._add_noise(info, actor_state, self._noise_scale("state", 1.0))

        privileged_state = jp.hstack(
            [
                state,
                self.t_qpos(data),
                self.t_linear_vel(data),
                self.t_angular_vel(data),
                self.t_site_error(data).ravel(),
                self.robot_base_pose(data, self._spec.miniq1),
                self.robot_base_linear_vel(data, self._spec.miniq1),
                self.robot_base_angular_vel(data, self._spec.miniq1),
                self.robot_base_pose(data, self._spec.miniq2),
                self.robot_base_linear_vel(data, self._spec.miniq2),
                self.robot_base_angular_vel(data, self._spec.miniq2),
                self.robot_joint_qpos(data, self._spec.miniq1),
                self.robot_joint_qvel(data, self._spec.miniq1),
                self.robot_joint_qpos(data, self._spec.miniq2),
                self.robot_joint_qvel(data, self._spec.miniq2),
                self.robot_gravity(data, self._spec.miniq1),
                self.robot_upvector(data, self._spec.miniq1),
                self.robot_gravity(data, self._spec.miniq2),
                self.robot_upvector(data, self._spec.miniq2),
                data.actuator_force,
                data.xfrc_applied[self._spec.miniq1.body_ids].ravel(),
                data.xfrc_applied[self._spec.miniq2.body_ids].ravel(),
                self.contact_summary(data, self._spec.miniq1.geom_ids, self._spec.push_t_geom_ids),
                self.contact_summary(data, self._spec.miniq2.geom_ids, self._spec.push_t_geom_ids),
                self.contact_summary(data, self._spec.miniq1.geom_ids, self._spec.miniq2.geom_ids),
            ]
        )
        return {"state": state, "privileged_state": privileged_state}

    def _noise_scale(self, name: str, default: float) -> float:
        noise_config = getattr(self._config, "noise_config", None)
        scales = getattr(noise_config, "scales", None)
        if scales is None:
            return default
        return scales.get(name, default)

    def _add_noise(
        self, info: dict[str, Any], value: jax.Array, scale: float
    ) -> jax.Array:
        noise_config = getattr(self._config, "noise_config", None)
        level = getattr(noise_config, "level", 0.0)
        if level == 0.0 or scale == 0.0:
            return value
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noise = 2 * jax.random.uniform(noise_rng, shape=value.shape) - 1
        return value + noise * level * scale

    @staticmethod
    def sin_cos(angle: jax.Array) -> jax.Array:
        return jp.array([jp.sin(angle), jp.cos(angle)])

    @staticmethod
    def yaw_from_quat(quat: jax.Array) -> jax.Array:
        w, x, y, z = quat
        return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def t_qpos(self, data: mjx.Data) -> jax.Array:
        return data.qpos[self._spec.push_t_qposadr]

    def t_xy(self, data: mjx.Data) -> jax.Array:
        return self.t_qpos(data)[:2]

    def t_yaw(self, data: mjx.Data) -> jax.Array:
        return self.yaw_from_quat(self.t_qpos(data)[3:7])

    def t_linear_vel(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._spec.push_t_dofadr[:3]]

    def t_angular_vel(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._spec.push_t_dofadr[3:6]]

    def t_planar_vel(self, data: mjx.Data) -> jax.Array:
        return jp.array([self.t_linear_vel(data)[0], self.t_linear_vel(data)[1], self.t_angular_vel(data)[2]])

    def t_sites(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[self._spec.push_t_site_pos_adr]

    def target_sites(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[self._spec.target_push_t_site_pos_adr]

    def t_site_error(self, data: mjx.Data) -> jax.Array:
        return self.t_sites(data) - self.target_sites(data)

    def target_xy(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._spec.target_push_t_body_id][:2]

    def target_yaw(self, data: mjx.Data) -> jax.Array:
        return self.yaw_from_quat(data.xquat[self._spec.target_push_t_body_id])

    def robot_base_pose(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.qpos[robot.base_qposadr]

    def robot_xy(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return self.robot_base_pose(data, robot)[:2]

    def robot_yaw(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return self.yaw_from_quat(self.robot_base_pose(data, robot)[3:7])

    def robot_base_linear_vel(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.qvel[robot.base_dofadr[:3]]

    def robot_base_angular_vel(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.qvel[robot.base_dofadr[3:6]]

    def robot_joint_qpos(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.qpos[robot.actuator_qposadr]

    def robot_joint_qvel(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.qvel[robot.actuator_dofadr]

    def robot_upvector(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.sensordata[robot.upvector_adr]

    def robot_gravity(self, data: mjx.Data, robot: MiniQSpec) -> jax.Array:
        return data.site_xmat[robot.imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def contact_summary(
        self,
        data: mjx.Data,
        geom_ids_a: jax.Array,
        geom_ids_b: jax.Array,
    ) -> jax.Array:
        contact = data._impl.contact
        geom = contact.geom
        geom1 = geom[:, 0]
        geom2 = geom[:, 1]

        geom1_in_a = (geom1[:, None] == geom_ids_a[None, :]).any(axis=1)
        geom1_in_b = (geom1[:, None] == geom_ids_b[None, :]).any(axis=1)
        geom2_in_a = (geom2[:, None] == geom_ids_a[None, :]).any(axis=1)
        geom2_in_b = (geom2[:, None] == geom_ids_b[None, :]).any(axis=1)
        pair_match = (geom1_in_a & geom2_in_b) | (geom1_in_b & geom2_in_a)
        active = pair_match & (geom1 >= 0) & (geom2 >= 0) & (contact.dist <= 0.0)

        efc_address = jp.maximum(contact.efc_address, 0)
        normal_force = jp.abs(data._impl.efc_force[efc_address])
        force_sum = jp.sum(jp.where(active, normal_force, 0.0))
        flag = jp.any(active).astype(jp.float32)
        return jp.array([flag, force_sum])

    @property
    def observation_size(self) -> dict[str, int]:
        state_size = 22
        robot_body_force_size = (
            self._spec.miniq1.body_ids.size + self._spec.miniq2.body_ids.size
        ) * 6
        privileged_state_size = (
            state_size
            + 7  # true T qpos
            + 3  # true T linear velocity
            + 3  # true T angular velocity
            + 9  # T endpoint/site error
            + 2 * (7 + 3 + 3)  # robot base pose, linear velocity, angular velocity
            + 2 * (8 + 8)  # robot joint qpos and qvel
            + 2 * (3 + 3)  # gravity and upvector
            + 16  # actuator forces
            + robot_body_force_size
            + 3 * 2  # three contact summaries: flag + force
        )
        return {
            "state": state_size,
            "privileged_state": privileged_state_size,
        }
