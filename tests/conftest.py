"""Shared fixtures for MiniQ-PushT tests."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jp
import pytest
from ml_collections import config_dict

from miniq_pusht.load_config import load_config

REPO_ROOT = pathlib.Path(__file__).parent.parent


def make_config(
    episode_length: int = 50,
) -> config_dict.ConfigDict:
    config = load_config(REPO_ROOT / "configs/miniq.yaml")
    config.unlock()
    config.mjcf.path = str(REPO_ROOT / "miniq_mjcf/urdf/scene.xml")
    config.impl = "jax"
    config.naconmax = 512
    config.njmax = 96
    config.episode_length = episode_length
    config.lock()
    return config


@pytest.fixture(scope="session")
def env():
    from miniq_pusht.open_loop.env import Env
    return Env(make_config(), config_overrides={"domain_randomization.enable": False})


@pytest.fixture(scope="module")
def initial_state(env):
    return env.reset(jax.random.PRNGKey(42))


@pytest.fixture(scope="module")
def zero_step_state(env, initial_state):
    return env.step(initial_state, jp.zeros(env.action_size))
