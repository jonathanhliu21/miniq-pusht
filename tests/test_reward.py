"""Reward component unit tests."""
from __future__ import annotations

import jax.numpy as jp
import pytest


def _components(env, initial_state, action=None, is_success=False):
    info = dict(initial_state.info)
    if is_success:
        info["is_success"] = jp.array(True)
    if action is None:
        action = jp.zeros(env.action_size)
    return env._reward.get_reward_components(
        initial_state.data, action, info, jp.array(0.0)
    )


def test_all_components_present(env, initial_state):
    comps = _components(env, initial_state)
    expected = {"endpoint_error", "progress", "useful_push",
                "r1_push_site", "r2_push_site", "action_rate",
                "action_penalty", "success"}
    assert set(comps.keys()) == expected


def test_all_components_finite(env, initial_state):
    comps = _components(env, initial_state)
    for name, val in comps.items():
        assert jp.isfinite(val), f"reward component '{name}' is not finite"


def test_endpoint_error_raw_in_unit_interval(env, initial_state):
    comps = _components(env, initial_state)
    scale = env._config.reward_config.scales.endpoint_error
    raw = float(comps["endpoint_error"]) / scale
    assert 0.0 < raw <= 1.0, f"raw endpoint_error={raw} not in (0, 1]"


def test_push_site_raw_in_unit_interval(env, initial_state):
    comps = _components(env, initial_state)
    for key in ("r1_push_site", "r2_push_site"):
        scale = env._config.reward_config.scales[key]
        raw = float(comps[key]) / scale
        assert 0.0 < raw <= 1.0, f"raw {key}={raw} not in (0, 1]"


def test_action_penalty_nonpositive(env, initial_state):
    comps = _components(env, initial_state, action=jp.ones(env.action_size))
    assert float(comps["action_penalty"]) <= 0.0


def test_action_rate_zero_when_action_unchanged(env, initial_state):
    # action == last_act → rate term raw value is 0
    action = initial_state.info["last_act"]
    comps = _components(env, initial_state, action=action)
    assert float(comps["action_rate"]) == pytest.approx(0.0, abs=1e-6)


def test_success_bonus_fires_when_flagged(env, initial_state):
    comps = _components(env, initial_state, is_success=True)
    scale = float(env._config.reward_config.scales.success)
    assert float(comps["success"]) == pytest.approx(scale)


def test_success_bonus_zero_when_not_flagged(env, initial_state):
    comps = _components(env, initial_state, is_success=False)
    assert float(comps["success"]) == pytest.approx(0.0)


def test_reward_value_finite(env, initial_state):
    comps = _components(env, initial_state)
    total = env._reward.get_reward_value(comps, env.dt)
    assert jp.isfinite(total)


def test_reward_value_is_scalar(env, initial_state):
    comps = _components(env, initial_state)
    total = env._reward.get_reward_value(comps, env.dt)
    assert total.shape == ()


def test_only_positive_rewards_clip(env, initial_state):
    if not env._config.reward_config.only_positive_rewards:
        pytest.skip("only_positive_rewards is disabled in config")
    comps = _components(env, initial_state)
    total = env._reward.get_reward_value(comps, env.dt)
    assert float(total) >= 0.0


def test_site_mean_error_nonnegative(env, initial_state):
    err = env._reward.site_mean_error(initial_state.data)
    assert float(err) >= 0.0


def test_abs_yaw_error_in_range(env, initial_state):
    err = env._reward.abs_yaw_error(initial_state.data)
    import math
    assert 0.0 <= float(err) <= math.pi
