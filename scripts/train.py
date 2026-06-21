import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
# Let JAX share the GPU with the warp physics backend instead of preallocating
# 75% of VRAM. Cap JAX at 80% so warp's contact pool has room.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import copy
import pickle
import shutil
from datetime import datetime
from pathlib import Path

from etils import epath
import jax

# replace outdated jax implementation device_put_replicated
if not hasattr(jax, "device_put_replicated"):
    from jax._src.api import device_put_replicated as _device_put_replicated
    jax.device_put_replicated = _device_put_replicated
    del _device_put_replicated

import mediapy as media
import mujoco
import wandb

from miniq_pusht.open_loop.env import Env
from miniq_pusht.load_config import load_config
from miniq_pusht.open_loop.train_fn import get_train_fn


def configure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


def save_rollout_video(
    config_path: str,
    make_inference_fn,
    params,
    seed: int,
    out_path: epath.Path,
    video_seconds: float,
    video_length: int | None,
    render_every: int,
    width: int,
    height: int,
    camera: str,
) -> epath.Path:
    render_cfg = load_config(Path(config_path))
    render_env = Env(render_cfg, config_overrides={"impl": "jax"})
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(render_env.reset)
    jit_step = jax.jit(render_env.step)

    rng = jax.random.PRNGKey(seed + 1)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    rollout = []
    n_steps = video_length or round(video_seconds / render_env.dt)
    render_every = max(render_every, 1)

    for step_idx in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, action_rng)
        state = jit_step(state, action)
        if step_idx % render_every == 0:
            rollout.append(jax.device_get(state))

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = render_env.render(
        rollout,
        height=height,
        width=width,
        camera=camera,
        scene_option=scene_option,
    )
    fps = 1.0 / render_env.dt / render_every
    configure_ffmpeg()
    media.write_video(out_path, frames, fps=fps)
    print(f"Rollout video saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/miniq.yaml")
    parser.add_argument("--logdir", type=str, default="logs/train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--restore-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Path to a previous run folder under logs/miniq_ppo/ — its "
            "policy_params.pkl is loaded as the initial policy. Fresh "
            "optimizer state; combine with --rewards to advance the "
            "curriculum. Mutually exclusive with --resume."
        ),
    )
    parser.add_argument(
        "--rewards",
        type=str,
        default=None,
        metavar="REWARDS_YAML",
        help=(
            "Optional YAML containing only a top-level reward_config: block. "
            "When given, its contents replace cfg.reward_config from --config "
            "for this run; everything else (env, network, training) stays "
            "exactly as --config specifies."
        ),
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="miniq")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default=None
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-seconds", type=float, default=30.0)
    parser.add_argument("--video-length", type=int, default=None)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-camera", type=str, default="track")
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

    env = Env(config=copy.deepcopy(cfg))
    eval_env = Env(config=copy.deepcopy(cfg), config_overrides={"impl": "jax"})

    # Curriculum warm-start: load a previous run's final policy_params.pkl.
    init_params = None
    init_params_src = None
    if args.init_from is not None:
        if args.resume:
            raise ValueError(
                "--init-from and --resume are mutually exclusive: the former "
                "warm-starts a fresh run, the latter continues an existing one."
            )
        init_dir = Path(args.init_from)
        init_path = init_dir / "policy_params.pkl"
        if not init_path.exists():
            raise FileNotFoundError(
                f"No policy_params.pkl found in {init_dir} — pass the run "
                f"folder that contains the saved checkpoint."
            )
        with init_path.open("rb") as f:
            init_params = pickle.load(f)
        init_params_src = str(init_path)
        print(f"Initializing policy params from: {init_path}")

    seed = int(cfg.env.seed if args.seed is None else args.seed)

    run_name = f"{cfg.env.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logdir = epath.Path(args.logdir).resolve() / run_name
    ckpt_dir = logdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_dir = logdir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, config_dir / Path(args.config).name)
    extra_cfg_str = cfg.get("env", {}).get("extra_configs", "")
    if extra_cfg_str:
        extra_path = Path(extra_cfg_str)
        if not extra_path.is_absolute():
            extra_path = Path(args.config).parent / extra_path
        if extra_path.exists():
            shutil.copy(str(extra_path), config_dir / extra_path.name)
    if args.rewards is not None:
        shutil.copy(args.rewards, config_dir / Path(args.rewards).name)
    if init_params_src is not None:
        (logdir / "init_from.txt").write_text(init_params_src + "\n")

    print(f"Logging to: {logdir}")
    print(f"Device: {jax.devices()}")
    print(f"Observation size: {env.observation_size}")
    print(f"Action size: {env.action_size}")
    print(f"Episode length: {cfg.episode_length}")
    print(f"Control dt: {env.dt}")
    print(f"Substeps: {env.n_substeps}")
    print("Starting PPO training...")

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            dir=str(logdir),
            mode=args.wandb_mode,
            config={"seed": seed, "config_path": args.config, "cfg": cfg.to_dict()},
        )

    try:
        train_fn = get_train_fn(
            cfg=cfg,
            env=env,
            seed=seed,
            checkpoint_path=ckpt_dir,
            restore_path=args.restore_path,
            resume=args.resume,
            init_params=init_params,
        )
        make_inference_fn, params, metrics = train_fn(
            environment=env,
            eval_env=eval_env,
        )

        print("Training done.")
        print(metrics)

        ckpt_path = os.path.join(str(logdir), "policy_params.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Saved policy params to: {ckpt_path}")

        if not args.no_video:
            video_path = logdir / "rollout.mp4"
            save_rollout_video(
                config_path=args.config,
                make_inference_fn=make_inference_fn,
                params=params,
                seed=seed,
                out_path=video_path,
                video_seconds=args.video_seconds,
                video_length=args.video_length,
                render_every=args.render_every,
                width=args.video_width,
                height=args.video_height,
                camera=args.video_camera,
            )
            if wandb_run is not None:
                fps = 1.0 / env.dt / max(args.render_every, 1)
                wandb.log(
                    {
                        "rollout/video": wandb.Video(
                            str(video_path), fps=fps, format="mp4"
                        )
                    }
                )

        rng = jax.random.PRNGKey(seed)
        state = jax.jit(env.reset)(rng)
        inference_fn = make_inference_fn(params)
        action, _ = inference_fn(state.obs, rng)
        print("Sample action:", action)
        print("Sample action shape:", action.shape)
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
