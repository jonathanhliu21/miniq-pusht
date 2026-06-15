from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp

from miniq_pusht.constants import GAIT_FREQUENCY_HZ


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GaitGenerator:
    """Continuous open-loop gait generator for one MiniQ.

    The generated target order matches the MJCF actuator order for one robot:
    FR hip, FR knee, FL hip, FL knee, BR hip, BR knee, BL hip, BL knee.

    Inputs are normalized commands `[forward, yaw, effort]`.  This intentionally
    does not enforce a calibrated m/s mapping; larger absolute forward/yaw
    commands produce larger stride and higher stepping frequency.
    """

    default_pose: jax.Array
    ctrlrange: jax.Array
    base_frequency: float = 1.2
    max_frequency: float = 4.0
    max_hip_amplitude: float = 0.45
    max_knee_amplitude: float = 0.35
    yaw_stride_gain: float = 0.75
    min_effort: float = 0.05
    stance_knee_gain: float = 0.25
    update_rate_hz: float = GAIT_FREQUENCY_HZ

    def frequency(self, command: jax.Array) -> jax.Array:
        """Returns oscillator frequency in Hz for `[forward, yaw, effort]`."""
        forward, yaw, effort = self._clip_command(command)
        command_mag = jp.maximum(jp.abs(forward), jp.abs(yaw))
        effort_mag = jp.abs(effort)
        speed_scale = jp.clip(command_mag * effort_mag, 0.0, 1.0)
        return self.base_frequency + (self.max_frequency - self.base_frequency) * speed_scale

    def step_phase(self, phase: jax.Array, command: jax.Array) -> jax.Array:
        """Advances a normalized phase in cycles, wrapping to `[0, 1)`."""
        dt = 1.0 / self.update_rate_hz
        return jp.mod(phase + self.frequency(command) * dt, 1.0)

    def step(self, phase: jax.Array, command: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advances phase at 50 Hz and returns `(next_phase, targets)`.

        The generator remains stateless so this method can be jitted.  Store the
        returned phase in the caller's env state/info and pass it back on the next
        control tick.
        """
        next_phase = self.step_phase(phase, command)
        return next_phase, self.targets(next_phase, command)

    def targets(self, phase: jax.Array, command: jax.Array) -> jax.Array:
        """Maps normalized gait phase and command to 8 actuator targets."""
        forward, yaw, effort = self._clip_command(command)
        effort_scale = jp.clip(jp.abs(effort), self.min_effort, 1.0)

        right_stride = jp.clip(forward + self.yaw_stride_gain * yaw, -1.0, 1.0)
        left_stride = jp.clip(forward - self.yaw_stride_gain * yaw, -1.0, 1.0)

        # Trot phasing: FR and BL together, FL and BR offset by half a cycle.
        leg_phases = jp.mod(
            phase + jp.array([0.0, 0.5, 0.5, 0.0], dtype=jp.float32), 1.0
        )
        side_stride = jp.array(
            [right_stride, left_stride, right_stride, left_stride],
            dtype=jp.float32,
        )

        theta = 2.0 * jp.pi * leg_phases
        hip_delta = self.max_hip_amplitude * effort_scale * side_stride * jp.sin(theta)

        swing = jp.maximum(jp.sin(theta), 0.0)
        stance = jp.maximum(-jp.sin(theta), 0.0)
        knee_delta = self.max_knee_amplitude * effort_scale * (
            swing - self.stance_knee_gain * stance
        )

        deltas = jp.ravel(jp.stack([hip_delta, knee_delta], axis=1))
        targets = self.default_pose + deltas
        return jp.clip(targets, self.ctrlrange[:, 0], self.ctrlrange[:, 1])

    def __call__(self, phase: jax.Array, command: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.step(phase, command)

    @staticmethod
    def _clip_command(command: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        command = jp.clip(command, -1.0, 1.0)
        return command[0], command[1], command[2]
