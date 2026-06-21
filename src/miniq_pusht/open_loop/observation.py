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
        robot1_xy = self.robot_xy(data, self._spec.miniq1)
        robot1_xy_noisy = self._add_noise(info, robot1_xy, self._config.noise_config.scales.robot1_xy) 
        robot1_yaw = self.robot_yaw(data, self._spec.miniq1) 
        robot1_yaw_noisy = self._add_noise(info, robot1_yaw, self._config.noise_config.scales.robot1_yaw)

        robot2_xy = self.robot_xy(data, self._spec.miniq2)
        robot2_xy_noisy = self._add_noise(info, robot2_xy, self._config.noise_config.scales.robot2_xy) 
        robot2_yaw = self.robot_yaw(data, self._spec.miniq2)
        robot2_yaw_noisy = self._add_noise(info, robot2_yaw, self._config.noise_config.scales.robot2_yaw)

        t_xy = self.t_xy(data)
        t_xy_noisy = self._add_noise(info, t_xy, self._config.noise_config.scales.t_xy)
        t_yaw = self.t_yaw(data)
        t_yaw_noisy = self._add_noise(info, t_yaw, self._config.noise_config.scales.t_yaw)

        r1_rel_xy = robot1_xy - t_xy
        r2_rel_xy = robot2_xy - t_xy
        r1_rel_yaw = robot1_yaw - t_yaw
        r2_rel_yaw = robot2_yaw - t_yaw
        t_rel_target_xy = t_xy - self.target_xy(data)
        t_yaw_error = t_yaw - self.target_yaw(data)
        t_site_xy_error = jp.linalg.norm(self.t_site_error(data)[:, :2], axis=1)
        t_vel_estimate = info.get("t_vel_estimate", self.t_planar_vel(data))
        last_action = info.get("last_act", jp.zeros(4))

        r1_rel_xy_noisy = robot1_xy_noisy - t_xy_noisy
        r2_rel_xy_noisy = robot2_xy_noisy - t_xy_noisy
        r1_rel_yaw_noisy = robot1_yaw_noisy - t_yaw_noisy
        r2_rel_yaw_noisy = robot2_yaw_noisy - t_yaw_noisy
        t_rel_target_xy_noisy = t_xy_noisy - self.target_xy(data)
        t_yaw_error_noisy = t_yaw_noisy - self.target_yaw(data)
        t_vel_estimate_noisy = self._add_noise(info, t_vel_estimate, self._config.noise_config.scales.t_vel_estimate)

        state = jp.hstack([
                r1_rel_xy_noisy,
                r2_rel_xy_noisy,
                self.sin_cos(r1_rel_yaw_noisy),
                self.sin_cos(r2_rel_yaw_noisy),
                t_rel_target_xy_noisy,
                self.sin_cos(t_yaw_error_noisy),
                t_vel_estimate_noisy,
                last_action,
        ])

        unnoisy_state = jp.hstack(
            [
                r1_rel_xy,
                r2_rel_xy,
                self.sin_cos(r1_rel_yaw),
                self.sin_cos(r2_rel_yaw),
                t_rel_target_xy,
                self.sin_cos(t_yaw_error),
                t_site_xy_error,
                t_vel_estimate,
            ]
        )

        privileged_state = jp.hstack(
            [
                state,
                unnoisy_state,
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
                info["action_delay"].astype(jp.float32)[None],
                info["obs_delay"].astype(jp.float32)[None],
            ]
        )
        return {"state": state, "privileged_state": privileged_state}

    def _add_noise(
        self, info: dict[str, Any], value: jax.Array, scale: float
    ) -> jax.Array:
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noise = 2 * jax.random.uniform(noise_rng, shape=value.shape) - 1
        return value + noise * self._config.noise_config.level * scale

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
        return jp.mean(self.t_sites(data)[:, :2], axis=0)

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
        if hasattr(data._impl, "contact"):
            # JAX backend: contact data is a nested Contact struct.
            contact = data._impl.contact
            geom = contact.geom          # (naconmax, 2)
            dist = contact.dist          # (naconmax,)
            efc_address = contact.efc_address  # (naconmax,)
            efc_force = data._impl.efc_force   # (nefc,)
            active_mask = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        else:
            # Warp backend: contact fields are flat with double-underscore names.
            geom = data._impl.contact__geom    # (naconmax, 2)
            dist = data._impl.contact__dist    # (naconmax,)
            raw_adr = data._impl.contact__efc_address  # (naconmax, dim) 2D in warp
            efc_address = raw_adr[:, 0] if raw_adr.ndim == 2 else raw_adr
            efc_force = data._impl.efc__force  # (njmax,)
            # Inactive slots are not sentinel-marked; mask by nacon.
            nacon = data._impl.nacon
            active_mask = jp.arange(geom.shape[0]) < nacon

        geom1 = geom[:, 0]
        geom2 = geom[:, 1]

        geom1_in_a = (geom1[:, None] == geom_ids_a[None, :]).any(axis=1)
        geom1_in_b = (geom1[:, None] == geom_ids_b[None, :]).any(axis=1)
        geom2_in_a = (geom2[:, None] == geom_ids_a[None, :]).any(axis=1)
        geom2_in_b = (geom2[:, None] == geom_ids_b[None, :]).any(axis=1)
        pair_match = (geom1_in_a & geom2_in_b) | (geom1_in_b & geom2_in_a)
        active = pair_match & active_mask & (dist <= 0.0)

        efc_address_safe = jp.maximum(efc_address, 0)
        # make sure force is not NaN
        safe_efc_force = jp.nan_to_num(efc_force, nan=0.0, posinf=0.0, neginf=0.0)
        normal_force = jp.abs(safe_efc_force[efc_address_safe])
        force_sum = jp.sum(jp.where(active, normal_force, 0.0))
        flag = jp.any(active).astype(jp.float32)
        return jp.array([flag, force_sum])

    @property
    def observation_size(self) -> dict[str, int]:
        # state: r1_xy(2) r2_xy(2) r1_sc(2) r2_sc(2) t_xy(2) t_yaw_sc(2) t_vel(3) action(4)
        state_size = 19
        # unnoisy_state: r1_xy(2) r2_xy(2) r1_sc(2) r2_sc(2) t_xy(2) t_yaw_sc(2) t_site_err(3) t_vel(3)
        unnoisy_state_size = 18
        robot_body_force_size = (
            self._spec.miniq1.body_ids.size + self._spec.miniq2.body_ids.size
        ) * 6
        privileged_state_size = (
            state_size
            + unnoisy_state_size
            + 7   # T qpos
            + 3   # T linear vel
            + 3   # T angular vel
            + 9   # T site error (3 sites × 3 coords)
            + 2 * (7 + 3 + 3)  # robot base pose, linvel, angvel
            + 2 * (8 + 8)      # robot joint qpos, qvel
            + 2 * (3 + 3)      # gravity, upvector
            + 16               # actuator forces
            + robot_body_force_size
            + 3 * 2            # 3 contact summaries (flag + force each)
            + 2                # action_delay, obs_delay
        )
        return {
            "state": state_size,
            "privileged_state": privileged_state_size,
        }
