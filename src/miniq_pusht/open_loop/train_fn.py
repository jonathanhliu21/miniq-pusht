import functools
from typing import Optional

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
from ml_collections import ConfigDict
from mujoco_playground._src.wrapper import wrap_for_brax_training
import wandb


def _scalar_metrics(metrics: dict) -> dict:
    scalars = {}
    for name, value in metrics.items():
        value = jax.device_get(value)
        if hasattr(value, "shape") and value.shape != ():
            continue
        try:
            scalars[name] = value.item()
        except AttributeError:
            scalars[name] = value
    return scalars


def get_train_fn(
    cfg: ConfigDict,
    env,
    seed: int,
    checkpoint_path: Optional[str | epath.Path] = None,
    restore_path: Optional[str | epath.Path] = None,
    resume: bool = False,
    init_params=None,
):
    """Build a brax PPO training function with our config + bookkeeping.

    Two distinct restore modes are exposed:

    * ``restore_path`` + ``resume=True``: orbax checkpoint folder. Restores
      params *and* optimizer state — use for resuming an interrupted run.
    * ``init_params``: a pickled brax params PyTree (typically the
      ``policy_params.pkl`` written at end-of-training). Restores the policy
      and value networks but starts a fresh optimizer — use for curriculum-
      style warm-starts where the loss landscape is about to change
      (different reward weights, new domain randomization, etc.).

    Passing both is an error: ``restore_path`` would win in brax and
    ``init_params`` would be silently ignored.
    """
    if init_params is not None and resume and restore_path is not None:
        raise ValueError(
            "Pass either init_params (curriculum warm-start) or "
            "restore_path+resume (continue training), not both."
        )

    def progress(num_steps: int, metrics: dict):
        avg_episode_length = metrics.get("eval/avg_episode_length", float("nan"))
        print(
            f"steps={num_steps} "
            f"eval/episode_reward="
            f"{metrics.get('eval/episode_reward', float('nan')):.3f} "
            f"eval/avg_episode_length={avg_episode_length:.1f}"
        )
        if wandb.run is not None:
            wandb.log(_scalar_metrics(metrics), step=int(num_steps))

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.network.actor_hidden_dims),
        value_hidden_layer_sizes=tuple(cfg.network.critic_hidden_dims),
        policy_obs_key=cfg.network.policy_obs_key,
        value_obs_key=cfg.network.value_obs_key,
    )

    randomization_fn = None
    if cfg.get("domain_randomization") is not None and cfg.domain_randomization.get(
        "enable", False
    ):
        randomization_fn = env.domain_randomize

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=int(cfg.training.num_timesteps),
        num_evals=int(cfg.training.num_evals),
        reward_scaling=float(cfg.training.reward_scaling),
        episode_length=int(cfg.episode_length),
        normalize_observations=bool(cfg.training.normalize_observations),
        action_repeat=int(cfg.training.action_repeat),
        unroll_length=int(cfg.training.unroll_length),
        num_minibatches=int(cfg.training.num_minibatches),
        num_updates_per_batch=int(cfg.training.num_updates_per_batch),
        discounting=float(cfg.training.discounting),
        learning_rate=float(cfg.training.learning_rate),
        entropy_cost=float(cfg.training.entropy_cost),
        num_envs=int(cfg.training.num_envs),
        batch_size=int(cfg.training.batch_size),
        max_grad_norm=float(cfg.training.max_grad_norm),
        clipping_epsilon=float(cfg.training.clipping_epsilon),
        network_factory=network_factory,
        seed=int(seed),
        progress_fn=progress,
        save_checkpoint_path=(
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        restore_checkpoint_path=str(restore_path) if resume and restore_path else None,
        restore_params=init_params,
        wrap_env_fn=wrap_for_brax_training,
        num_eval_envs=int(cfg.training.num_eval_envs),
        randomization_fn=randomization_fn,
    )

    return train_fn
