from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / ".matplotlib"))

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from miniq_pusht.open_loop.env import Env
from miniq_pusht.load_config import load_config


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
}


def make_command_plot(env: Env, trajectory: list, times: np.ndarray) -> go.Figure:
    ctrl = np.asarray([jax.device_get(state.data.ctrl) for state in trajectory])

    fig = go.Figure()
    for actuator_id in range(env.mj_model.nu):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=ctrl[:, actuator_id],
                mode="lines",
                name=env.mj_model.actuator(actuator_id).name,
            )
        )

    fig.update_layout(
        title="Commanded Motor Positions",
        xaxis_title="Time (s)",
        yaxis_title="Actuator target (rad)",
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
    )
    return fig


def _body_xmat(data, body_id: int) -> np.ndarray:
    xmat = np.asarray(jax.device_get(data.xmat[body_id]))
    return xmat.reshape(3, 3)


def _base_frame_foot_positions(env: Env, trajectory: list) -> dict[str, np.ndarray]:
    spec = env._spec
    robot_specs = {
        "r1": spec.miniq1,
        "r2": spec.miniq2,
    }
    positions = {}

    for robot_name, robot in robot_specs.items():
        base_body_id = int(robot.base_body_id)
        foot_site_ids = np.asarray(jax.device_get(robot.feet_site_ids), dtype=int)
        robot_positions = []

        for state in trajectory:
            data = state.data
            base_pos = np.asarray(jax.device_get(data.xpos[base_body_id]))
            base_xmat = _body_xmat(data, base_body_id)
            foot_world = np.asarray(jax.device_get(data.site_xpos[foot_site_ids]))
            foot_local = (foot_world - base_pos) @ base_xmat
            robot_positions.append(foot_local)

        positions[robot_name] = np.asarray(robot_positions)

    return positions


def make_base_frame_foot_plot(env: Env, trajectory: list, times: np.ndarray) -> go.Figure:
    foot_positions = _base_frame_foot_positions(env, trajectory)
    leg_names = ("FR", "FL", "BR", "BL")
    colors = {
        "FR": "#1f77b4",
        "FL": "#d62728",
        "BR": "#2ca02c",
        "BL": "#9467bd",
    }
    subplot_specs = (
        ("r1", 0, "Robot 1 foot x positions"),
        ("r1", 2, "Robot 1 foot z positions"),
        ("r2", 0, "Robot 2 foot x positions"),
        ("r2", 2, "Robot 2 foot z positions"),
    )

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[title for _, _, title in subplot_specs],
    )

    for row_id, (robot_name, axis_id, _) in enumerate(subplot_specs, start=1):
        robot_positions = foot_positions[robot_name]
        axis_name = "x" if axis_id == 0 else "z"
        for leg_id, leg_name in enumerate(leg_names):
            trace_name = f"{robot_name} {leg_name} {axis_name}"
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=robot_positions[:, leg_id, axis_id],
                    mode="lines",
                    name=trace_name,
                    legendgroup=leg_name,
                    line={"color": colors[leg_name]},
                    showlegend=row_id == 1,
                ),
                row=row_id,
                col=1,
            )
        fig.update_yaxes(title_text=f"{axis_name} (m)", row=row_id, col=1)

    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_layout(
        title="Foot Positions Relative to Each Robot Base",
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        height=1200,
    )
    return fig


def save_plot_report(env: Env, trajectory: list, times: np.ndarray, out_path: Path) -> None:
    command_fig = make_command_plot(env, trajectory, times)
    foot_fig = make_base_frame_foot_plot(env, trajectory, times)
    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<title>MiniQ Locomotion Plots</title>",
            '<style>body{font-family:Arial,sans-serif;margin:24px;} .plot{margin-bottom:40px;}</style>',
            "</head>",
            "<body>",
            "<h1>MiniQ Locomotion Plots</h1>",
            '<div class="plot">',
            command_fig.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config=PLOTLY_CONFIG,
            ),
            "</div>",
            '<div class="plot">',
            foot_fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config=PLOTLY_CONFIG,
            ),
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r1-command",
        type=float,
        nargs=2,
        metavar=("FWD", "YAW"),
        default=[0.5, 0.0],
        help="Forward and yaw speed for robot 1 (default: 0.5 0.0)",
    )
    parser.add_argument(
        "--r2-command",
        type=float,
        nargs=2,
        metavar=("FWD", "YAW"),
        default=[0.5, 0.0],
        help="Forward and yaw speed for robot 2 (default: 0.5 0.0)",
    )
    args = parser.parse_args()
    action = jp.array([*args.r1_command, *args.r2_command], dtype=jp.float32)

    config = load_config(ROOT / "configs" / "miniq.yaml")
    config.impl = "jax"
    # Disable start randomization: it places the T at a random position in the
    # arena, where walking robots can step on the 3 cm-tall T and launch it.
    config.domain_randomization.starts.enable = False
    env = Env(config)

    state = env.reset(jax.random.PRNGKey(config.env.seed))
    trajectory = [state]
    step_fn = jax.jit(env.step)

    rollout_seconds = 10.0
    num_steps = int(round(rollout_seconds / env.dt))

    for step in range(num_steps):
        state = step_fn(state, action)
        jax.block_until_ready(state.data.qpos)
        trajectory.append(state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "logs" / "env_test" / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    times = np.arange(len(trajectory)) * env.dt
    save_plot_report(env, trajectory, times, out_dir / "locomotion_plots.html")

    render_stride = 1
    rendered_trajectory = trajectory[::render_stride]
    frames = env.render(
        rendered_trajectory,
        height=360,
        width=640,
        camera="track",
    )

    out_path = out_dir / "rollout.mp4"
    media.write_video(out_path, frames, fps=int(round(1.0 / env.dt / render_stride)))
    print(out_path)


if __name__ == "__main__":
    main()
