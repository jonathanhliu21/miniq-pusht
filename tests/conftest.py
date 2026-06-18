"""Shared fixtures for MiniQ-PushT tests."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jp
import pytest
import yaml
from ml_collections import config_dict

REPO_ROOT = pathlib.Path(__file__).parent.parent


def make_config(
    domain_rand: bool = False,
    episode_length: int = 50,
    noise_level: float = 1.0,
) -> config_dict.ConfigDict:
    with open(REPO_ROOT / "configs/miniq.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mjcf"]["path"] = str(REPO_ROOT / "miniq_mjcf/urdf/scene.xml")
    cfg["env"]["extra_configs"] = str(REPO_ROOT / "configs/open_loop.yaml")
    cfg["impl"] = "jax"
    cfg["naconmax"] = 512
    cfg["njmax"] = 96
    cfg["episode_length"] = episode_length
    cfg["domain_randomization"]["enable"] = domain_rand
    return config_dict.ConfigDict(cfg)


@pytest.fixture(scope="session")
def env():
    from miniq_pusht.open_loop.env import Env
    return Env(make_config(domain_rand=False))


@pytest.fixture(scope="module")
def initial_state(env):
    return env.reset(jax.random.PRNGKey(42))


@pytest.fixture(scope="module")
def zero_step_state(env, initial_state):
    return env.step(initial_state, jp.zeros(env.action_size))
