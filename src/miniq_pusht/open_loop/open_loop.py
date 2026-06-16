from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp

from miniq_pusht.constants import GAIT_FREQUENCY_HZ
from miniq_pusht.randomize import GAIT_FREQ, GAIT_LIFT, GAIT_PHASE, GAIT_STRIDE, GAIT_YAW


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GaitGenerator:
    """Continuous open-loop gait generator for one MiniQ.

    The generated target order matches the MJCF actuator order for one robot:
    FR hip, FR knee, FL hip, FL knee, BR hip, BR knee, BL hip, BL knee.

    Inputs are normalized commands `[forward, yaw]`.  This intentionally
    does not enforce a calibrated m/s mapping; larger absolute forward/yaw
    commands produce larger foot stride and higher stepping frequency.
    """

    default_pose: jax.Array
    ctrlrange: jax.Array
    base_frequency: float = 1.2
    max_frequency: float = 4.0
    x0_mm: float = 0.0
    y0_mm: float = 40.0
    max_stride_mm: float = 40.0
    max_lift_mm: float = 20.0
    stance_drop_mm: float = 0.0
    swing_count: float = 20.0
    stance_count: float = 40.0
    link_1_mm: float = 30.75
    link_2_mm: float = 30.75
    yaw_stride_gain: float = 0.75
    update_rate_hz: float = GAIT_FREQUENCY_HZ

    def frequency(self, command: jax.Array) -> jax.Array:
        """Returns oscillator frequency in Hz for `[forward, yaw]`."""
        return self.frequency_with_scale(command, 1.0)

    def frequency_with_scale(
        self, command: jax.Array, frequency_scale: jax.Array
    ) -> jax.Array:
        """Returns oscillator frequency with a multiplicative randomization."""
        forward, yaw = self._clip_command(command)
        command_mag = jp.maximum(jp.abs(forward), jp.abs(yaw))
        speed_scale = jp.clip(command_mag, 0.0, 1.0)
        frequency = self.base_frequency + (self.max_frequency - self.base_frequency) * speed_scale
        return frequency * frequency_scale

    def step_phase(self, phase: jax.Array, command: jax.Array) -> jax.Array:
        """Advances a normalized phase in cycles, wrapping to `[0, 1)`."""
        return self.step_phase_with_scale(phase, command, 1.0)

    def step_phase_with_scale(
        self,
        phase: jax.Array,
        command: jax.Array,
        frequency_scale: jax.Array,
    ) -> jax.Array:
        """Advances a normalized phase with a randomized frequency scale."""
        dt = 1.0 / self.update_rate_hz
        return jp.mod(phase + self.frequency_with_scale(command, frequency_scale) * dt, 1.0)

    def step(self, phase: jax.Array, command: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advances phase at 50 Hz and returns `(next_phase, targets)`.

        The generator remains stateless so this method can be jitted.  Store the
        returned phase in the caller's env state/info and pass it back on the next
        control tick.
        """
        next_phase = self.step_phase(phase, command)
        return next_phase, self.targets(next_phase, command)

    def step_randomized(
        self,
        phase: jax.Array,
        command: jax.Array,
        params: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Advances phase and returns targets using per-episode gait params."""
        next_phase = self.step_phase_with_scale(phase, command, params[GAIT_FREQ])
        return next_phase, self.targets_randomized(next_phase, command, params)

    def targets(self, phase: jax.Array, command: jax.Array) -> jax.Array:
        """Maps normalized gait phase and command to 8 actuator targets."""
        default_params = jp.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            dtype=jp.float32,
        )
        return self.targets_randomized(phase, command, default_params)

    def targets_randomized(
        self,
        phase: jax.Array,
        command: jax.Array,
        params: jax.Array,
    ) -> jax.Array:
        """Maps gait phase and command to targets with per-leg randomization."""
        forward, yaw = self._clip_command(command)
        command_mag = jp.maximum(jp.abs(forward), jp.abs(yaw))
        movement_scale = jp.clip(command_mag, 0.0, 1.0)

        yaw_stride_gain = self.yaw_stride_gain * params[GAIT_YAW]
        right_stride = jp.clip(forward + yaw_stride_gain * yaw, -1.0, 1.0)
        left_stride = jp.clip(forward - yaw_stride_gain * yaw, -1.0, 1.0)

        # Trot phasing: FR and BL together, FL and BR offset by half a cycle.
        leg_phases = jp.mod(
            phase
            + jp.array([0.0, 0.5, 0.5, 0.0], dtype=jp.float32)
            + params[GAIT_PHASE],
            1.0,
        )
        side_stride = jp.array(
            [right_stride, left_stride, right_stride, left_stride],
            dtype=jp.float32,
        )

        stride_mm = self.max_stride_mm * side_stride * params[GAIT_STRIDE]
        lift_mm = self.max_lift_mm * movement_scale * params[GAIT_LIFT]
        stance_drop_mm = self.stance_drop_mm * movement_scale
        foot_x, foot_y = self._foot_trajectory(leg_phases, stride_mm, lift_mm, stance_drop_mm)

        ik_targets = self._ik_360(foot_x, foot_y)
        nominal_ik = self._ik_360(
            jp.full_like(foot_x, self.x0_mm),
            jp.full_like(foot_y, self.y0_mm),
        )
        deltas = jp.ravel(ik_targets - nominal_ik)
        targets = self.default_pose + deltas
        return jp.clip(targets, self.ctrlrange[:, 0], self.ctrlrange[:, 1])

    def __call__(self, phase: jax.Array, command: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.step(phase, command)

    @staticmethod
    def _clip_command(command: jax.Array) -> tuple[jax.Array, jax.Array]:
        command = jp.clip(command, -1.0, 1.0)
        return command[0], command[1]

    def _foot_trajectory(
        self,
        phase: jax.Array,
        stride_mm: jax.Array,
        lift_mm: jax.Array,
        stance_drop_mm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Continuous version of the historical MiniQ base trajectory."""
        swing_fraction = self.swing_count / (self.swing_count + self.stance_count)
        in_swing = phase < swing_fraction
        swing_u = jp.clip(phase / swing_fraction, 0.0, 1.0)
        stance_u = jp.clip((phase - swing_fraction) / (1.0 - swing_fraction), 0.0, 1.0)

        x_start = self.x0_mm - stride_mm / 2.0
        x_end = self.x0_mm + stride_mm / 2.0
        swing_x = x_start + swing_u * stride_mm
        stance_x = x_end - stance_u * stride_mm
        swing_y = self.y0_mm - jp.sin(jp.pi * swing_u) * lift_mm
        stance_y = self.y0_mm + jp.sin(jp.pi * stance_u) * stance_drop_mm

        return jp.where(in_swing, swing_x, stance_x), jp.where(in_swing, swing_y, stance_y)

    def _ik_360(self, x_mm: jax.Array, y_mm: jax.Array) -> jax.Array:
        """Vectorized JAX form of `k_solver_360.ik_solve(..., deg=False)`."""
        r2 = x_mm * x_mm + y_mm * y_mm
        c2 = (r2 - self.link_1_mm**2 - self.link_2_mm**2) / (
            2.0 * self.link_1_mm * self.link_2_mm
        )
        c2 = jp.clip(c2, -0.999, 0.999)
        s2 = -jp.sqrt(jp.maximum(0.0, 1.0 - c2 * c2))
        q2 = jp.arctan2(s2, c2)
        q1 = jp.arctan2(y_mm, x_mm) - jp.arctan2(
            self.link_2_mm * s2,
            self.link_1_mm + self.link_2_mm * c2,
        )
        return jp.stack([q1, q1 + q2], axis=1)
