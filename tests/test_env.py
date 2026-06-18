"""Environment correctness: reset, step, scalar outputs."""
from __future__ import annotations

import jax
import jax.numpy as jp
import pytest


def test_reset_qpos_finite(initial_state):
    assert jp.all(jp.isfinite(initial_state.data.qpos))


def test_reset_qvel_finite(initial_state):
    assert jp.all(jp.isfinite(initial_state.data.qvel))


def test_reset_obs_finite(initial_state):
    for key, val in initial_state.obs.items():
        assert jp.all(jp.isfinite(val)), f"obs['{key}'] has non-finite values at reset"


def test_reset_obs_keys(initial_state):
    assert "state" in initial_state.obs
    assert "privileged_state" in initial_state.obs


def test_reset_reward_zero(initial_state):
    assert float(initial_state.reward) == pytest.approx(0.0)


def test_reset_done_zero(initial_state):
    assert float(initial_state.done) == pytest.approx(0.0)


def test_zero_action_obs_finite(zero_step_state):
    for key, val in zero_step_state.obs.items():
        assert jp.all(jp.isfinite(val)), f"obs['{key}'] has NaN/Inf after zero action"


def test_zero_action_reward_finite(zero_step_state):
    assert jp.isfinite(zero_step_state.reward)


def test_reward_is_scalar(zero_step_state):
    assert zero_step_state.reward.shape == ()


def test_done_is_scalar(zero_step_state):
    assert zero_step_state.done.shape == ()


def test_done_is_float(zero_step_state):
    assert zero_step_state.done.dtype == jp.float32


def test_step_increments_step_count(env, initial_state):
    action = jp.zeros(env.action_size)
    s1 = env.step(initial_state, action)
    assert int(s1.info["step_count"]) == 1
    s2 = env.step(s1, action)
    assert int(s2.info["step_count"]) == 2


def test_last_act_updated(env, initial_state):
    action = jp.ones(env.action_size) * 0.5
    s = env.step(initial_state, action)
    assert jp.allclose(s.info["last_act"], action * env._config.action_scale)


def test_reset_has_r2_push_site_adr(initial_state):
    assert "r2_push_site_adr" in initial_state.info
    adr = initial_state.info["r2_push_site_adr"]
    assert adr.shape == (3,)


def test_prev_site_error_in_info(initial_state):
    assert "prev_site_error" in initial_state.info
    assert jp.isfinite(initial_state.info["prev_site_error"])
