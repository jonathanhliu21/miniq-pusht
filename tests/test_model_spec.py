"""Model indexing: addresses, geom IDs, sensordata reads."""
from __future__ import annotations

import jax.numpy as jp
import pytest


# ---------------------------------------------------------------------------
# Address range helpers
# ---------------------------------------------------------------------------

def _check_addrs_in_range(addrs, upper, label):
    for a in addrs.ravel():
        assert 0 <= int(a) < upper, f"{label}: address {int(a)} out of range [0, {upper})"


# ---------------------------------------------------------------------------
# qpos / qvel address tests
# ---------------------------------------------------------------------------

def test_r1_base_qposadr_in_range(env):
    _check_addrs_in_range(env._spec.miniq1.base_qposadr, env._mj_model.nq, "r1.base_qposadr")


def test_r2_base_qposadr_in_range(env):
    _check_addrs_in_range(env._spec.miniq2.base_qposadr, env._mj_model.nq, "r2.base_qposadr")


def test_r1_actuator_qposadr_in_range(env):
    _check_addrs_in_range(env._spec.miniq1.actuator_qposadr, env._mj_model.nq, "r1.actuator_qposadr")


def test_r2_actuator_qposadr_in_range(env):
    _check_addrs_in_range(env._spec.miniq2.actuator_qposadr, env._mj_model.nq, "r2.actuator_qposadr")


def test_push_t_qposadr_in_range(env):
    _check_addrs_in_range(env._spec.push_t_qposadr, env._mj_model.nq, "push_t_qposadr")


def test_base_qposadr_length(env):
    # freejoint: 7 qpos entries (xyz + quat)
    assert env._spec.miniq1.base_qposadr.shape == (7,)
    assert env._spec.miniq2.base_qposadr.shape == (7,)


def test_actuator_qposadr_length(env):
    # 8 actuated joints per robot
    assert env._spec.miniq1.actuator_qposadr.shape == (8,)
    assert env._spec.miniq2.actuator_qposadr.shape == (8,)


# ---------------------------------------------------------------------------
# Sensor address tests
# ---------------------------------------------------------------------------

def test_r1_upvector_adr_in_range(env, initial_state):
    ns = initial_state.data.sensordata.shape[0]
    _check_addrs_in_range(env._spec.miniq1.upvector_adr, ns, "r1.upvector_adr")


def test_r2_upvector_adr_in_range(env, initial_state):
    ns = initial_state.data.sensordata.shape[0]
    _check_addrs_in_range(env._spec.miniq2.upvector_adr, ns, "r2.upvector_adr")


def test_push_t_site_pos_adr_in_range(env, initial_state):
    ns = initial_state.data.sensordata.shape[0]
    _check_addrs_in_range(env._spec.push_t_site_pos_adr, ns, "push_t_site_pos_adr")


def test_r1_push_site_adr_in_range(env, initial_state):
    ns = initial_state.data.sensordata.shape[0]
    _check_addrs_in_range(env._spec.push_t_r1_push_site_adr, ns, "r1_push_site_adr")


def test_r2_push_site_l_adr_in_range(env, initial_state):
    ns = initial_state.data.sensordata.shape[0]
    _check_addrs_in_range(env._spec.push_t_r2_push_site_adr_l, ns, "r2_push_site_adr_l")


def test_r2_push_site_r_adr_in_range(env, initial_state):
    ns = initial_state.data.sensordata.shape[0]
    _check_addrs_in_range(env._spec.push_t_r2_push_site_adr_r, ns, "r2_push_site_adr_r")


def test_push_t_site_pos_adr_shape(env):
    # 3 sites × 3 coords each
    assert env._spec.push_t_site_pos_adr.shape == (3, 3)


def test_r1_push_site_adr_shape(env):
    assert env._spec.push_t_r1_push_site_adr.shape == (3,)


def test_r2_push_site_l_and_r_shape(env):
    assert env._spec.push_t_r2_push_site_adr_l.shape == (3,)
    assert env._spec.push_t_r2_push_site_adr_r.shape == (3,)


# ---------------------------------------------------------------------------
# Geom ID tests
# ---------------------------------------------------------------------------

def test_robot_geom_ids_nonempty(env):
    assert env._spec.miniq1.geom_ids.size > 0
    assert env._spec.miniq2.geom_ids.size > 0


def test_geom_ids_unique_across_robots(env):
    r1 = set(int(g) for g in env._spec.miniq1.geom_ids)
    r2 = set(int(g) for g in env._spec.miniq2.geom_ids)
    assert r1.isdisjoint(r2), "robots share geom IDs"


def test_push_t_geom_ids_nonempty(env):
    assert env._spec.push_t_geom_ids.size > 0


def test_geom_contype_robot1(env):
    # After _setup_inter_robot_collisions, robot1 geoms have contype=2
    for gid in env._spec.miniq1.geom_ids:
        assert env._mj_model.geom_contype[int(gid)] == 2, \
            f"r1 geom {int(gid)} contype != 2"


def test_geom_contype_robot2(env):
    for gid in env._spec.miniq2.geom_ids:
        assert env._mj_model.geom_contype[int(gid)] == 4, \
            f"r2 geom {int(gid)} contype != 4"


# ---------------------------------------------------------------------------
# Sensordata readability
# ---------------------------------------------------------------------------

def test_upvector_readable_and_finite(env, initial_state):
    data = initial_state.data
    r1_up = data.sensordata[env._spec.miniq1.upvector_adr]
    r2_up = data.sensordata[env._spec.miniq2.upvector_adr]
    assert r1_up.shape == (3,)
    assert r2_up.shape == (3,)
    assert jp.all(jp.isfinite(r1_up))
    assert jp.all(jp.isfinite(r2_up))


def test_upvector_z_positive_at_reset(env, initial_state):
    data = initial_state.data
    r1_up_z = float(data.sensordata[env._spec.miniq1.upvector_adr][2])
    r2_up_z = float(data.sensordata[env._spec.miniq2.upvector_adr][2])
    assert r1_up_z > 0.9, f"r1 upvector z={r1_up_z:.3f} too low at reset"
    assert r2_up_z > 0.9, f"r2 upvector z={r2_up_z:.3f} too low at reset"


def test_t_sites_readable_and_finite(env, initial_state):
    sites = env._obs_builder.t_sites(initial_state.data)
    assert sites.shape == (3, 3)
    assert jp.all(jp.isfinite(sites))


def test_r1_push_site_readable_and_finite(env, initial_state):
    pos = initial_state.data.sensordata[env._spec.push_t_r1_push_site_adr]
    assert pos.shape == (3,)
    assert jp.all(jp.isfinite(pos))


def test_r2_push_sites_readable_and_finite(env, initial_state):
    pos_l = initial_state.data.sensordata[env._spec.push_t_r2_push_site_adr_l]
    pos_r = initial_state.data.sensordata[env._spec.push_t_r2_push_site_adr_r]
    assert jp.all(jp.isfinite(pos_l))
    assert jp.all(jp.isfinite(pos_r))


# ---------------------------------------------------------------------------
# r2 push site selection
# ---------------------------------------------------------------------------

def test_r2_push_site_adr_is_one_of_lr(env, initial_state):
    chosen = initial_state.info["r2_push_site_adr"]
    l_adr = env._spec.push_t_r2_push_site_adr_l
    r_adr = env._spec.push_t_r2_push_site_adr_r
    is_l = jp.all(chosen == l_adr)
    is_r = jp.all(chosen == r_adr)
    assert bool(is_l) or bool(is_r), "r2_push_site_adr is neither _l nor _r"


def test_r2_push_site_adr_closer_than_other(env, initial_state):
    """The chosen site should be at most as far as the unchosen one."""
    data = initial_state.data
    r2_xy = env._obs_builder.robot_xy(data, env._spec.miniq2)
    push_l_xy = data.sensordata[env._spec.push_t_r2_push_site_adr_l][:2]
    push_r_xy = data.sensordata[env._spec.push_t_r2_push_site_adr_r][:2]
    d_l = float(jp.linalg.norm(r2_xy - push_l_xy))
    d_r = float(jp.linalg.norm(r2_xy - push_r_xy))

    chosen = initial_state.info["r2_push_site_adr"]
    chosen_is_l = bool(jp.all(chosen == env._spec.push_t_r2_push_site_adr_l))
    if chosen_is_l:
        assert d_l <= d_r + 1e-6
    else:
        assert d_r <= d_l + 1e-6


# ---------------------------------------------------------------------------
# Body count sanity
# ---------------------------------------------------------------------------

def test_robot_body_ids_count(env):
    # Each MiniQ robot has 13 bodies (base_link + 4 knee_rotors + 8 limb links)
    assert env._spec.miniq1.body_ids.size == 13
    assert env._spec.miniq2.body_ids.size == 13
