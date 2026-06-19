import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import copy
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from miniq_pusht.open_loop.env import Env
from miniq_pusht.load_config import load_config


def configure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


def _scalar(value) -> float:
    array = np.asarray(jax.device_get(value))
    return float(array.reshape(-1)[0])


def _series(diagnostics: list[dict[str, float]], key: str) -> list[float]:
    return [record[key] for record in diagnostics]


def save_diagnostics_plot(
    diagnostics: list[dict[str, float]], out_path: Path
) -> Path:
    """Interactive plotly HTML: robot positions, T error, T velocity, last action, reward components."""
    if not diagnostics:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    time = _series(diagnostics, "time")
    fieldnames = {key for record in diagnostics for key in record.keys()}
    last_act_keys = sorted(k for k in fieldnames if k.startswith("obs/last_act_"))
    reward_keys = sorted(
        key
        for key in fieldnames
        if key.startswith("reward/") and key != "reward/total"
    )

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Robot Positions (relative to T)",
            "T Pose Error (relative to target)",
            "T Velocity Estimate",
            "Last Action",
            "Reward Components",
        ),
        vertical_spacing=0.04,
        row_heights=[0.17, 0.17, 0.17, 0.12, 0.37],
    )

    for key, label in [
        ("obs/r1_rel_x", "r1_rel_x"), ("obs/r1_rel_y", "r1_rel_y"),
        ("obs/r2_rel_x", "r2_rel_x"), ("obs/r2_rel_y", "r2_rel_y"),
    ]:
        fig.add_trace(
            go.Scatter(x=time, y=_series(diagnostics, key), name=label, legendgroup="rpos"),
            row=1, col=1,
        )

    for key, label in [
        ("obs/t_err_x", "t_err_x"), ("obs/t_err_y", "t_err_y"),
        ("obs/t_yaw_sin", "t_yaw_sin"), ("obs/t_yaw_cos", "t_yaw_cos"),
    ]:
        fig.add_trace(
            go.Scatter(x=time, y=_series(diagnostics, key), name=label, legendgroup="terr"),
            row=2, col=1,
        )

    for key, label in [
        ("obs/t_vel_x", "t_vel_x"), ("obs/t_vel_y", "t_vel_y"), ("obs/t_vel_yaw", "t_vel_yaw"),
    ]:
        fig.add_trace(
            go.Scatter(x=time, y=_series(diagnostics, key), name=label, legendgroup="tvel"),
            row=3, col=1,
        )

    for key in last_act_keys:
        fig.add_trace(
            go.Scatter(x=time, y=_series(diagnostics, key), name=key, legendgroup="lact"),
            row=4, col=1,
        )

    for key in reward_keys:
        fig.add_trace(
            go.Scatter(x=time, y=_series(diagnostics, key), name=key, legendgroup="comp"),
            row=5, col=1,
        )

    fig.update_xaxes(title_text="Time (s)", row=5, col=1)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="m / rad", row=2, col=1)
    fig.update_yaxes(title_text="m/s or rad/s", row=3, col=1)
    fig.update_yaxes(title_text="action", row=4, col=1)
    fig.update_yaxes(title_text="Reward", row=5, col=1)
    fig.update_layout(height=1200, hovermode="x unified", legend=dict(groupclick="toggleitem"))

    fig.write_html(out_path, config={"scrollZoom": True})
    return out_path


def done_reason(env, state) -> str:
    obs = env._obs_builder
    spec = env._spec
    cfg = env._config

    reasons = []

    if bool(jax.device_get(state.info["is_success"])):
        reasons.append("success")

    data = state.data
    min_h = cfg.termination.min_base_height_m
    min_up = cfg.termination.min_upvector_z
    on_t_thresh = cfg.termination.robot_on_t_height_m

    for name, robot in [("r1", spec.miniq1), ("r2", spec.miniq2)]:
        base_z = float(jax.device_get(obs.robot_base_pose(data, robot)[2]))
        up_z = float(jax.device_get(obs.robot_upvector(data, robot)[2]))
        if base_z < min_h:
            reasons.append(f"{name}_too_low")
        if up_z < min_up:
            reasons.append(f"{name}_tilted")
        if base_z > on_t_thresh:
            reasons.append(f"{name}_on_t")

    step_count = int(jax.device_get(state.info["step_count"]))
    if step_count >= cfg.episode_length:
        reasons.append("timeout")

    return ",".join(reasons) if reasons else "unknown"


def _load_policy(args, cfg, config_overrides=None):
    """Load params and build env + JIT-compiled inference function."""
    if config_overrides is None:
        config_overrides = {"impl": "jax"}

    params_path = Path(args.logdir) / "policy_params.pkl"
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    print(f"Loaded params from {params_path}")

    env = Env(config=copy.deepcopy(cfg), config_overrides=config_overrides)
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(cfg.network.actor_hidden_dims),
        value_hidden_layer_sizes=tuple(cfg.network.critic_hidden_dims),
        policy_obs_key=cfg.network.policy_obs_key,
        value_obs_key=cfg.network.value_obs_key,
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    inference_fn = jax.jit(make_policy(params, deterministic=args.deterministic))
    return env, inference_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True,
                        help="Path to run folder containing policy_params.pkl")
    parser.add_argument("--config", type=str, default="configs/miniq.yaml")
    parser.add_argument(
        "--rewards",
        type=str,
        default=None,
        metavar="REWARDS_YAML",
        help=(
            "Optional YAML containing only a top-level reward_config: block "
            "(same shape as train.py's --rewards). When given, the merged "
            "config snapshotted to rollouts/<ts>/config.yaml reflects this "
            "override — useful for verifying that a curriculum-stage policy "
            "is being evaluated under the rewards it was trained on."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use the mode of the action distribution")
    parser.add_argument("--video-seconds", type=float, default=30.0)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-camera", type=str, default="track")
    parser.add_argument("--output-dir", type=str, default="logs/rollouts")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.rewards is not None:
        rewards_cfg = load_config(Path(args.rewards))
        if "reward_config" not in rewards_cfg:
            raise ValueError(
                f"--rewards file {args.rewards!r} must have a top-level "
                f"reward_config: block; got keys {list(rewards_cfg.keys())}"
            )
        cfg.reward_config = rewards_cfg.reward_config
        print(f"Overrode reward_config from: {args.rewards}")

    config_overrides = {"impl": "jax"}
    env, inference_fn = _load_policy(args, cfg, config_overrides=config_overrides)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    print(f"ctrl_dt={env.dt}s  deterministic={args.deterministic}")

    n_steps = round(args.video_seconds / env.dt)
    render_every = max(args.render_every, 1)
    rollout = []
    diagnostics: list[dict[str, float]] = []
    terminated_early = False

    for step_idx in range(n_steps):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        host_state = jax.device_get(state)
        state_obs = np.asarray(host_state.obs["state"])

        # state obs layout (19 dims):
        #   [0:2]  r1_rel_xy    [2:4]  r2_rel_xy
        #   [4:6]  r1_sin_cos   [6:8]  r2_sin_cos
        #   [8:10] t_rel_target_xy  [10:12] t_yaw_sin_cos
        #   [12:15] t_vel_estimate  [15:19] last_action
        record = {
            "time": step_idx * env.dt,
            "obs/r1_rel_x":   float(state_obs[0]),
            "obs/r1_rel_y":   float(state_obs[1]),
            "obs/r2_rel_x":   float(state_obs[2]),
            "obs/r2_rel_y":   float(state_obs[3]),
            "obs/r1_sin_yaw": float(state_obs[4]),
            "obs/r1_cos_yaw": float(state_obs[5]),
            "obs/r2_sin_yaw": float(state_obs[6]),
            "obs/r2_cos_yaw": float(state_obs[7]),
            "obs/t_err_x":    float(state_obs[8]),
            "obs/t_err_y":    float(state_obs[9]),
            "obs/t_yaw_sin":  float(state_obs[10]),
            "obs/t_yaw_cos":  float(state_obs[11]),
            "obs/t_vel_x":    float(state_obs[12]),
            "obs/t_vel_y":    float(state_obs[13]),
            "obs/t_vel_yaw":  float(state_obs[14]),
        }
        for i in range(env.action_size):
            record[f"obs/last_act_{i:02d}"] = float(state_obs[15 + i])
        for key, value in host_state.metrics.items():
            record[key] = _scalar(value)
        diagnostics.append(record)

        done_now = bool(state.done)
        if step_idx % render_every == 0 or done_now:
            rollout.append(host_state)
        if done_now:
            reason = done_reason(env, state)
            reward = float(jax.device_get(state.reward))
            site_err = float(jax.device_get(env._reward.site_mean_error(state.data)))
            print(
                f"done step={step_idx + 1} reason={reason} reward={reward:+.4f} "
                f"site_mean_error={site_err:.4f}"
            )
            terminated_early = True
            break

    if not terminated_early:
        print(f"Completed {n_steps} steps without termination.")

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = env.render(
        rollout,
        height=args.video_height,
        width=args.video_width,
        camera=args.video_camera,
        scene_option=scene_option,
    )
    fps = 1.0 / env.dt / render_every

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "rollout.mp4"

    config_dst = run_dir / "config.yaml"
    with config_dst.open("w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    if args.rewards is not None:
        shutil.copy(args.rewards, run_dir / "rewards.yaml")
    overrides_dst = run_dir / "config_overrides.txt"
    overrides_dst.write_text(
        "# Applied in-process by view_policy.py on top of config.yaml above.\n"
        + "".join(f"{k}: {v!r}\n" for k, v in config_overrides.items())
    )
    print(f"Config snapshot saved to: {config_dst}")
    if args.rewards is not None:
        print(f"Source rewards copied to: {run_dir / 'rewards.yaml'}")
    print(f"Override note saved to:   {overrides_dst}")

    configure_ffmpeg()
    media.write_video(video_path, frames, fps=fps)
    print(f"Rollout video saved to: {video_path}")

    plot_path = save_diagnostics_plot(diagnostics, run_dir / "diagnostics.html")
    print(f"Diagnostics plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
