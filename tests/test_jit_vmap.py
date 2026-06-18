"""JIT and vmap compatibility tests."""
from __future__ import annotations

import jax
import jax.numpy as jp
import pytest


def test_reset_jittable(env):
    jit_reset = jax.jit(env.reset)
    state = jit_reset(jax.random.PRNGKey(0))
    assert jp.all(jp.isfinite(state.obs["state"]))


def test_step_jittable(env, initial_state):
    jit_step = jax.jit(env.step)
    action = jp.zeros(env.action_size)
    state = jit_step(initial_state, action)
    assert jp.isfinite(state.reward)
    assert state.done.shape == ()


def test_reset_vmappable(env):
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    states = jax.vmap(env.reset)(keys)
    assert states.obs["state"].shape == (n, env.observation_size["state"])
    assert jp.all(jp.isfinite(states.obs["state"]))


def test_step_vmappable(env):
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    states = jax.vmap(env.reset)(keys)
    actions = jp.zeros((n, env.action_size))
    next_states = jax.vmap(env.step)(states, actions)
    assert next_states.reward.shape == (n,)
    assert next_states.done.shape == (n,)
    assert jp.all(jp.isfinite(next_states.reward))


def test_jit_reset_deterministic(env):
    """JIT-compiled reset with same key must produce identical results."""
    jit_reset = jax.jit(env.reset)
    key = jax.random.PRNGKey(7)
    s1 = jit_reset(key)
    s2 = jit_reset(key)
    assert jp.allclose(s1.obs["state"], s2.obs["state"])


def test_jit_step_deterministic(env, initial_state):
    """JIT-compiled step with same inputs must produce identical results."""
    jit_step = jax.jit(env.step)
    action = jp.ones(env.action_size) * 0.3
    s1 = jit_step(initial_state, action)
    s2 = jit_step(initial_state, action)
    assert jp.allclose(s1.reward, s2.reward)
    assert jp.allclose(s1.obs["state"], s2.obs["state"])


def test_vmap_reset_different_keys_differ(env):
    """vmap over different keys should produce different states."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    # Use domain randomization off (session env), so obs differ only due to noise
    states = jax.vmap(env.reset)(keys)
    # Different RNG keys → different noisy observations
    assert not jp.allclose(
        states.obs["state"][0], states.obs["state"][1]
    )


def test_jit_step_multi_step(env, initial_state):
    """Several JIT steps in sequence must stay finite."""
    jit_step = jax.jit(env.step)
    action = jp.zeros(env.action_size)
    state = initial_state
    for _ in range(5):
        state = jit_step(state, action)
    assert jp.all(jp.isfinite(state.obs["state"]))
