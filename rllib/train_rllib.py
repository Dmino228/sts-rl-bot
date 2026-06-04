"""Ray RLlib PPO training entrypoint for the STS RL bot."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from typing import Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rllib.env_wrapper import (
    DEFAULT_RLLIB_BASE_PORT,
    RLLIB_ENV_NAME,
    register_rllib_env,
)
from rllib.sb3_transfer import try_transfer_sb3_policy
from rllib.smoke_env import RLLIB_SMOKE_ENV_NAME, register_smoke_env


LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray RLlib PPO training for STS")
    parser.add_argument("--workers", type=int, default=2, help="Ray EnvRunner/rollout workers.")
    parser.add_argument("--envs-per-worker", type=int, default=1, help="Vector envs per Ray worker.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Additional env steps to train.")
    parser.add_argument("--base-env-dir", default=os.path.join(PROJECT_ROOT, "SlayTheSpire"))
    parser.add_argument("--workspace-dir", default=os.path.join(PROJECT_ROOT, "rllib_workers"))
    parser.add_argument("--character", default="IRONCLAD", choices=["IRONCLAD", "SILENT", "DEFECT", "WATCHER"])
    parser.add_argument("--multi-character", action="store_true", help="Round-robin all four characters.")
    parser.add_argument("--ram-usage", choices=["low", "default", "safe"], default="default")
    parser.add_argument("--base-port", type=int, default=DEFAULT_RLLIB_BASE_PORT)
    parser.add_argument("--use-xvfb", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--debug-env-info", action="store_true")
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--rollout-fragment-length", type=int, default=128)
    parser.add_argument("--checkpoint-freq", type=int, default=1, help="Save every N RLlib train iterations.")
    parser.add_argument("--resume-from", default="", help="Path to an RLlib checkpoint directory.")
    parser.add_argument("--init-from-sb3", default="", help="Optional SB3 .zip checkpoint for warm-start weights.")
    parser.add_argument("--local-mode", action="store_true", help="Run Ray local_mode for debugging.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny masked env instead of launching STS.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"rllib_training_{TIMESTAMP}.log")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    logger = logging.getLogger("train_rllib")

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    from rllib.action_mask_model import ACTION_MASK_MODEL, register_action_mask_model

    logger.info("Starting RLlib training session %s", TIMESTAMP)
    logger.info("Log file: %s", log_file)
    logger.info("Workers=%d envs_per_worker=%d timesteps=%d", args.workers, args.envs_per_worker, args.timesteps)

    ray.init(ignore_reinit_error=True, local_mode=args.local_mode, log_to_driver=True)
    register_action_mask_model()

    if args.smoke_test:
        register_smoke_env()
        env_name = RLLIB_SMOKE_ENV_NAME
        env_config: dict[str, Any] = {}
        logger.info("Using RLlib smoke env; Slay the Spire will not be launched.")
    else:
        register_rllib_env()
        env_name = RLLIB_ENV_NAME
        env_config = {
            "base_env_dir": args.base_env_dir,
            "workspace_dir": args.workspace_dir,
            "character_class": args.character,
            "multi_character": args.multi_character,
            "ram_usage": args.ram_usage,
            "base_port": args.base_port,
            "use_xvfb": args.use_xvfb,
            "force_rebuild": args.force_rebuild,
            "debug_env_info": args.debug_env_info,
            "num_envs_per_env_runner": args.envs_per_worker,
        }

    config = PPOConfig()
    config = _configure_api_stack(config)
    config = config.environment(env=env_name, env_config=env_config)
    config = config.framework("torch")
    config = _configure_rollout_workers(config, args)
    config = _configure_training(config, args)
    config = _configure_resources(config, args)
    config.model["custom_model"] = ACTION_MASK_MODEL
    config.model["fcnet_hiddens"] = [64, 64]
    config.model["vf_share_layers"] = False

    algo = _build_algorithm(config)
    try:
        if args.resume_from:
            logger.info("Restoring RLlib checkpoint from %s", args.resume_from)
            algo.restore(args.resume_from)

        current_steps = _algorithm_env_steps(algo)
        logger.info("Current RLlib env timesteps before training: %d", current_steps)

        if args.init_from_sb3 and not args.resume_from:
            try_transfer_sb3_policy(algo, args.init_from_sb3, logger)
        elif args.init_from_sb3:
            logger.info("Ignoring --init-from-sb3 because --resume-from was provided.")

        target_steps = current_steps + args.timesteps
        while current_steps < target_steps:
            result = algo.train()
            current_steps = _result_env_steps(result, fallback=current_steps)
            reward = _nested_get(result, ("env_runners", "episode_return_mean"))
            if reward is None:
                reward = result.get("episode_reward_mean")
            logger.info(
                "RLlib iteration=%s env_steps=%d reward_mean=%s",
                result.get("training_iteration"),
                current_steps,
                reward,
            )
            iteration = int(result.get("training_iteration", 0) or 0)
            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                checkpoint_path = _save_checkpoint(algo, logger)
                logger.info("Saved RLlib checkpoint: %s", checkpoint_path)

        checkpoint_path = _save_checkpoint(algo, logger)
        logger.info("Training complete. Final RLlib checkpoint: %s", checkpoint_path)
    except KeyboardInterrupt:
        logger.warning("Training interrupted. Saving RLlib checkpoint...")
        logger.info("Saved RLlib checkpoint: %s", _save_checkpoint(algo, logger))
    finally:
        algo.stop()
        ray.shutdown()


def _configure_api_stack(config: Any) -> Any:
    if not hasattr(config, "api_stack"):
        return config
    try:
        return config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    except TypeError:
        return config


def _configure_rollout_workers(config: Any, args: argparse.Namespace) -> Any:
    if hasattr(config, "rollouts"):
        try:
            return config.rollouts(
                num_rollout_workers=args.workers,
                num_envs_per_worker=args.envs_per_worker,
                rollout_fragment_length=args.rollout_fragment_length,
            )
        except (TypeError, ValueError):
            pass
    if hasattr(config, "env_runners"):
        return config.env_runners(
            num_env_runners=args.workers,
            num_envs_per_env_runner=args.envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
        )
    return config


def _configure_training(config: Any, args: argparse.Namespace) -> Any:
    try:
        return config.training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.minibatch_size,
            num_sgd_iter=args.num_epochs,
        )
    except TypeError:
        return config.training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
        )


def _configure_resources(config: Any, args: argparse.Namespace) -> Any:
    if hasattr(config, "resources"):
        return config.resources(num_gpus=args.num_gpus)
    return config


def _build_algorithm(config: Any) -> Any:
    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


def _save_checkpoint(algo: Any, logger: logging.Logger) -> str:
    checkpoint_dir = os.path.join(MODELS_DIR, "rllib")
    os.makedirs(checkpoint_dir, exist_ok=True)
    result = algo.save(checkpoint_dir)
    if isinstance(result, str):
        return result
    checkpoint = getattr(result, "checkpoint", None)
    path = getattr(checkpoint, "path", None)
    if path:
        return str(path)
    logger.debug("Unknown checkpoint result type: %r", result)
    return str(result)


def _algorithm_env_steps(algo: Any) -> int:
    counters = getattr(algo, "_counters", {})
    for key in ("num_env_steps_sampled", "num_agent_steps_sampled"):
        value = counters.get(key) if isinstance(counters, dict) else None
        if value is not None:
            return int(value)
    return 0


def _result_env_steps(result: dict[str, Any], fallback: int) -> int:
    candidates = (
        result.get("num_env_steps_sampled_lifetime"),
        result.get("timesteps_total"),
        _nested_get(result, ("env_runners", "num_env_steps_sampled_lifetime")),
        _nested_get(result, ("sampler_results", "num_env_steps_sampled")),
    )
    for value in candidates:
        if value is not None:
            return int(value)
    return fallback


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


if __name__ == "__main__":
    main()
