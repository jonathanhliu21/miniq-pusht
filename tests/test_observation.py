"""Observation shape, structure, and noise tests."""
from __future__ import annotations

import jax
import jax.numpy as jp
import pytest


def test_state_shape_matches_declared(env, initial_state):
    declared = env.observation_size["state"]
    actual = initial_state.obs["state"].shape[0]
    assert actual == declared, f"state: actual={actual} declared={declared}"


def test_privileged_state_shape_matches_declared(env, initial_state):
    declared = env.observation_size["privileged_state"]
    actual = initial_state.obs["privileged_state"].shape[0]
    assert actual == declared, f"privileged_state: actual={actual} declared={declared}"


def test_state_is_1d(initial_state):
    assert initial_state.obs["state"].ndim == 1


def test_privileged_state_is_1d(initial_state):
    assert initial_state.obs["privileged_state"].ndim == 1


def test_privileged_state_larger_than_state(env):
    sizes = env.observation_size
    assert sizes["privileged_state"] > sizes["state"]


def test_state_is_prefix_of_privileged_state(initial_state):
    # state is the first chunk of privileged_state
    state_len = initial_state.obs["state"].shape[0]
    priv = initial_state.obs["privileged_state"]
    assert jp.allclose(priv[:state_len], initial_state.obs["state"])


def test_noiseless_obs_identical_across_rngs(env, initial_state):
    """With noise level 0, obs must be identical regardless of RNG."""
    from ml_collections import config_dict
    from miniq_pusht.open_loop.observation import Observation

    raw = env._config.to_dict()
    raw["noise_config"]["level"] = 0.0
    noiseless_cfg = config_dict.ConfigDict(raw)

    obs_builder = Observation(env._spec, noiseless_cfg)

    data = initial_state.data
    info_a = {**initial_state.info, "rng": jax.random.PRNGKey(0)}
    info_b = {**initial_state.info, "rng": jax.random.PRNGKey(999)}

    obs_a = obs_builder.get_obs(data, info_a)
    obs_b = obs_builder.get_obs(data, info_b)

    assert jp.allclose(obs_a["state"], obs_b["state"]), \
        "noiseless observations differ across RNG keys"


def test_noisy_obs_differs_across_rngs(env, initial_state):
    """With nonzero noise, different RNG keys produce different observations."""
    data = initial_state.data
    info_a = {**initial_state.info, "rng": jax.random.PRNGKey(1)}
    info_b = {**initial_state.info, "rng": jax.random.PRNGKey(2)}

    obs_a = env._obs_builder.get_obs(data, info_a)
    obs_b = env._obs_builder.get_obs(data, info_b)

    assert not jp.allclose(obs_a["state"], obs_b["state"]), \
        "noisy observations should differ across RNG keys"


def test_obs_state_finite_after_step(zero_step_state):
    assert jp.all(jp.isfinite(zero_step_state.obs["state"]))
    assert jp.all(jp.isfinite(zero_step_state.obs["privileged_state"]))


def test_t_xy_is_centroid_of_sites(env, initial_state):
    """t_xy() returns the mean of the three T site xy positions."""
    data = initial_state.data
    sites = env._obs_builder.t_sites(data)      # (3, 3)
    centroid_xy = jp.mean(sites[:, :2], axis=0)  # (2,)
    t_xy = env._obs_builder.t_xy(data)
    assert jp.allclose(t_xy, centroid_xy)


def test_sin_cos_unit_norm(env, initial_state):
    """sin_cos encoding should have unit norm (sin²+cos²=1)."""
    yaw = env._obs_builder.robot_yaw(initial_state.data, env._spec.miniq1)
    sc = env._obs_builder.sin_cos(yaw)
    assert jp.allclose(jp.linalg.norm(sc), jp.array(1.0), atol=1e-5)


def test_t_site_error_shape(env, initial_state):
    err = env._obs_builder.t_site_error(initial_state.data)
    assert err.shape == (3, 3)


def test_contact_summary_shape(env, initial_state):
    summary = env._obs_builder.contact_summary(
        initial_state.data,
        env._spec.miniq1.geom_ids,
        env._spec.push_t_geom_ids,
    )
    assert summary.shape == (2,)


def test_contact_summary_flag_in_range(env, initial_state):
    summary = env._obs_builder.contact_summary(
        initial_state.data,
        env._spec.miniq1.geom_ids,
        env._spec.push_t_geom_ids,
    )
    flag = float(summary[0])
    assert flag in (0.0, 1.0)
