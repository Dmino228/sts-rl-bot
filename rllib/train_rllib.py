"""Ray RLlib training entrypoint for the STS RL bot."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import sys
import threading
import time
from typing import Any

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rllib.config import (
    build_env_config,
    handle_dry_run,
    parse_args,
    resolve_config,
)
from rllib.console import TrainingConsole
from rllib.preflight import run_preflight
from rllib.run_folder import RunFolder, create_run_folder

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "sts_rl_checkpoint_v1"
RLLIB_CUSTOM_MODEL_NAME = "sts_torch_action_mask_model"


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    handle_dry_run(config)

    # Create run folder
    experiment_name = config.get("_training_stage", "experiment")
    run_folder = create_run_folder(RUNS_DIR, experiment_name, config)
    config["_run_folder_path"] = run_folder.path

    # Logging: file handler to run folder, console handler conditional
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    console_mode = str(config.get("console_mode", "compact")).strip().lower()

    # File handler always gets full verbose output
    file_handler = logging.FileHandler(run_folder.train_log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )

    # Console handler: in compact mode, only show warnings+ (rich handles the rest)
    console_handler = logging.StreamHandler(sys.stderr)
    if console_mode == "compact":
        console_handler.setLevel(logging.WARNING)
    elif console_mode == "quiet":
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(getattr(logging, config.get("log_level", "INFO")))
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
    )
    logger = logging.getLogger("train_rllib")

    import ray

    from rllib.action_mask_model import ACTION_MASK_MODEL, register_action_mask_model
    from rllib.progress_metrics import ProgressMetricsCallback

    logger.info("Starting RLlib training session %s", TIMESTAMP)
    logger.info("Log file: %s", run_folder.train_log_path)
    logger.info("Run folder: %s", run_folder.path)

    game_key = config["_game_key"]
    checkpoint_dir = config["_checkpoint_dir"]

    # Preflight validation (before Ray)
    run_preflight(config, logger)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    register_action_mask_model()

    # Build env
    from rllib.env_wrapper import RLLIB_ENV_NAME, register_rllib_env
    from rllib.smoke_env import RLLIB_SMOKE_ENV_NAME, register_smoke_env

    if config.get("smoke_test", False):
        register_smoke_env()
        env_name = RLLIB_SMOKE_ENV_NAME
        env_config: dict[str, Any] = {}
        logger.info("Using RLlib smoke env; Slay the Spire will not be launched.")
    else:
        register_rllib_env()
        env_name = RLLIB_ENV_NAME
        env_config = build_env_config(config)

    from ray.rllib.algorithms.ppo import PPOConfig

    ppo_config = PPOConfig()
    ppo_config = _configure_api_stack(ppo_config)
    ppo_config = ppo_config.environment(env=env_name, env_config=env_config)
    ppo_config = ppo_config.framework("torch")
    ppo_config = _configure_rollout_workers(ppo_config, config)
    ppo_config = _configure_training(ppo_config, config)
    ppo_config = _configure_resources(ppo_config, config)
    ppo_config = _configure_seed(ppo_config, config)
    ppo_config = _configure_fault_tolerance(ppo_config, config)
    ppo_config = _configure_callbacks(ppo_config, ProgressMetricsCallback)
    ppo_config.model["custom_model"] = ACTION_MASK_MODEL
    ppo_config.model["fcnet_hiddens"] = [64, 64]
    ppo_config.model["vf_share_layers"] = False

    algo = _build_algorithm(ppo_config)
    heartbeat = _TrainHeartbeat(logger, float(config.get("train_heartbeat_s", 30.0)))
    heartbeat.start()

    # Training console
    current_steps = 0
    source_checkpoint = ""
    curriculum_mode = str(config.get("sts2_curriculum_mode", "full_run"))
    target_timesteps = int(config.get("timesteps", 1_000_000))

    try:
        resume_from = _resolve_resume_path(config, checkpoint_dir, logger=logger)
        init_from_rllib = _resolve_init_from_rllib_path(config)
        if resume_from:
            _validate_checkpoint_metadata_compatibility(
                config,
                resume_from,
                mode="resume",
                logger=logger,
            )
            _warn_resume_stage_safety(config, checkpoint_dir, resume_from, logger)
            logger.info("Restoring RLlib checkpoint from %s", resume_from)
            algo.restore(resume_from)
            if init_from_rllib:
                logger.info(
                    "Ignoring --init-from-rllib=%s because --resume-from/auto-resume restored %s.",
                    init_from_rllib,
                    resume_from,
                )
        elif init_from_rllib:
            _validate_checkpoint_metadata_compatibility(
                config,
                init_from_rllib,
                mode="init_from_rllib",
                logger=logger,
            )
            _init_from_rllib_checkpoint(
                algo=algo,
                source_checkpoint=init_from_rllib,
                logger=logger,
            )
        source_checkpoint = _source_checkpoint(config, resume_from, init_from_rllib)

        current_steps = _algorithm_env_steps(algo)
        logger.info("Current RLlib env timesteps before training: %d", current_steps)

        init_sb3 = str(config.get("init_from_sb3", "") or "")
        if init_sb3 and not resume_from and not init_from_rllib:
            from rllib.sb3_transfer import try_transfer_sb3_policy

            try_transfer_sb3_policy(algo, init_sb3, logger)
        elif init_sb3 and init_from_rllib:
            logger.info("Ignoring --init-from-sb3 because --init-from-rllib was used.")
        elif init_sb3:
            logger.info("Ignoring --init-from-sb3 because an RLlib checkpoint was restored.")

        # Random baseline at startup
        eval_random = int(config.get("eval_random_baseline", 0) or 0)
        if game_key == "sts2" and not config.get("smoke_test") and eval_random > 0:
            _run_random_combat_baseline(env_config=env_config, episodes=eval_random, logger=logger)
        eval_greedy = int(config.get("eval_greedy_baseline", 0) or 0)
        if game_key == "sts2" and not config.get("smoke_test") and eval_greedy > 0:
            _run_greedy_combat_baseline(env_config=env_config, episodes=eval_greedy, logger=logger)

        target_steps = current_steps + target_timesteps

        # Init console
        console = TrainingConsole(
            mode=console_mode,
            curriculum_mode=curriculum_mode,
            target_steps=target_steps,
            logger=logger,
        )

        checkpoint_freq = int(config.get("checkpoint_freq", 1) or 1)
        eval_combat_episodes = int(config.get("eval_combat_episodes", 0) or 0)
        eval_combat_freq = int(config.get("eval_combat_freq", 0) or 0)
        eval_random_freq = int(config.get("eval_random_baseline_freq", 0) or 0)
        eval_deterministic = bool(config.get("eval_combat_deterministic", False))
        slow_iteration_s = float(config.get("slow_iteration_s", 60.0) or 60.0)

        try:
            if bool(config.get("eval_only", False)):
                if eval_combat_episodes <= 0:
                    raise SystemExit("--eval-only requires --eval-combat-episodes > 0.")
                metrics = _run_policy_combat_eval(
                    algo=algo,
                    env_config=env_config,
                    episodes=eval_combat_episodes,
                    deterministic=eval_deterministic,
                    logger=logger,
                )
                console.on_eval("ppo_eval", metrics)
                run_folder.save_metrics_line(
                    {
                        "iteration": 0,
                        "eval_only": True,
                        "env_steps": current_steps,
                        **{f"eval_{key}": value for key, value in metrics.items()},
                    }
                )
                console.on_finish({
                    "total_steps": current_steps,
                    "checkpoint_path": "n/a (eval-only)",
                })
                return

            while current_steps < target_steps:
                previous_steps = current_steps
                iteration_started_at = time.perf_counter()
                heartbeat.begin(next_iteration=_algorithm_iteration(algo) + 1)
                try:
                    result = algo.train()
                finally:
                    heartbeat.end()
                iteration_seconds = time.perf_counter() - iteration_started_at
                current_steps = _result_env_steps(result, fallback=current_steps)
                step_delta = _result_env_step_delta(
                    result,
                    previous_steps=previous_steps,
                    current_steps=current_steps,
                )

                reward = _nested_get(result, ("env_runners", "episode_return_mean"))
                if reward is None:
                    reward = result.get("episode_reward_mean")

                progress = _progress_log_metrics(result)
                combat = _combat_log_metrics(result)
                grouped = _grouped_combat_log_metrics(result)

                # Console output (mode-dependent)
                metrics_line = console.on_iteration(
                    iteration=int(result.get("training_iteration", 0) or 0),
                    current_steps=current_steps,
                    step_delta=step_delta,
                    iteration_seconds=iteration_seconds,
                    reward_mean=reward,
                    progress_metrics=progress,
                    combat_metrics=combat,
                    grouped_combat_metrics=grouped,
                )

                # Write to metrics.jsonl (always)
                metrics_line["deck_mode"] = str(config.get("deck_mode", "") or "starter")
                run_folder.save_metrics_line(metrics_line)

                # Verbose file log (always, regardless of console mode)
                _log_full_iteration(
                    logger=logger,
                    result=result,
                    current_steps=current_steps,
                    step_delta=step_delta,
                    iteration_seconds=iteration_seconds,
                    reward=reward,
                    progress=progress,
                    combat=combat,
                    grouped=grouped,
                )

                if slow_iteration_s > 0 and iteration_seconds > slow_iteration_s:
                    logger.warning(
                        "Slow RLlib iteration: %.1fs for %d env steps.",
                        iteration_seconds,
                        step_delta,
                    )

                iteration = int(result.get("training_iteration", 0) or 0)

                # Periodic random baseline
                if (
                    game_key == "sts2"
                    and not config.get("smoke_test")
                    and eval_random > 0
                    and eval_random_freq > 0
                    and iteration > 0
                    and iteration % eval_random_freq == 0
                ):
                    metrics = _run_random_combat_baseline(
                        env_config=env_config, episodes=eval_random, logger=logger
                    )
                    console.on_eval("random_baseline", metrics)

                # Periodic policy eval
                if (
                    game_key == "sts2"
                    and not config.get("smoke_test")
                    and eval_combat_episodes > 0
                    and eval_combat_freq > 0
                    and iteration > 0
                    and iteration % eval_combat_freq == 0
                ):
                    metrics = _run_policy_combat_eval(
                        algo=algo,
                        env_config=env_config,
                        episodes=eval_combat_episodes,
                        deterministic=eval_deterministic,
                        logger=logger,
                    )
                    console.on_eval("ppo_eval", metrics)

                # Checkpoint
                if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
                    checkpoint_path = _save_checkpoint_with_metadata(
                        algo, logger, checkpoint_dir, config, game_key,
                        total_steps=current_steps, source_checkpoint=source_checkpoint,
                    )
                    logger.info("Saved RLlib checkpoint: %s", checkpoint_path)

            # Final checkpoint
            checkpoint_path = _save_checkpoint_with_metadata(
                algo, logger, checkpoint_dir, config, game_key,
                total_steps=current_steps, source_checkpoint=source_checkpoint,
            )
            console.on_finish({
                "total_steps": current_steps,
                "checkpoint_path": checkpoint_path,
            })
        finally:
            console.close()

    except KeyboardInterrupt:
        logger.warning("Training interrupted. Saving RLlib checkpoint...")
        logger.info(
            "Saved RLlib checkpoint: %s",
            _save_checkpoint_with_metadata(
                algo, logger, checkpoint_dir, config, game_key,
                total_steps=current_steps, source_checkpoint=source_checkpoint,
            ),
        )
    finally:
        heartbeat.stop()
        algo.stop()
        ray.shutdown()


# ---------------------------------------------------------------------------
# RLlib configuration helpers
# ---------------------------------------------------------------------------

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


def _configure_rollout_workers(config: Any, resolved: dict[str, Any]) -> Any:
    workers = int(resolved.get("workers", 2) or 2)
    envs_per_worker = int(resolved.get("envs_per_worker", 1) or 1)
    rollout_fragment = int(resolved.get("rollout_fragment_length", 128) or 128)
    sample_timeout = float(resolved.get("sample_timeout_s", 600.0) or 600.0)

    if hasattr(config, "rollouts"):
        try:
            return config.rollouts(
                num_rollout_workers=workers,
                num_envs_per_worker=envs_per_worker,
                rollout_fragment_length=rollout_fragment,
                sample_timeout_s=sample_timeout,
            )
        except (TypeError, ValueError):
            pass
    if hasattr(config, "env_runners"):
        return config.env_runners(
            num_env_runners=workers,
            num_envs_per_env_runner=envs_per_worker,
            rollout_fragment_length=rollout_fragment,
            sample_timeout_s=sample_timeout,
        )
    return config


def _configure_training(config: Any, resolved: dict[str, Any]) -> Any:
    batch = int(resolved.get("train_batch_size", 1024) or 1024)
    mini = int(resolved.get("minibatch_size", 256) or 256)
    epochs = int(resolved.get("num_epochs", 4) or 4)
    try:
        return config.training(
            train_batch_size=batch,
            sgd_minibatch_size=mini,
            num_sgd_iter=epochs,
        )
    except TypeError:
        return config.training(
            train_batch_size=batch,
            minibatch_size=mini,
            num_epochs=epochs,
        )


def _configure_resources(config: Any, resolved: dict[str, Any]) -> Any:
    gpus = float(resolved.get("num_gpus", 0.0) or 0.0)
    cpus_per = float(resolved.get("cpus_per_worker", 1.0) or 1.0)
    if hasattr(config, "resources"):
        if cpus_per == 1.0:
            return config.resources(num_gpus=gpus)
        try:
            return config.resources(num_gpus=gpus, num_cpus_per_worker=cpus_per)
        except TypeError:
            return config.resources(num_gpus=gpus)
    return config


def _configure_seed(config: Any, resolved: dict[str, Any]) -> Any:
    seed = resolved.get("seed")
    if seed is None or not hasattr(config, "debugging"):
        return config
    try:
        return config.debugging(seed=int(seed))
    except TypeError:
        return config


def _configure_fault_tolerance(config: Any, resolved: dict[str, Any]) -> Any:
    if resolved.get("disable_env_runner_fault_tolerance") or not hasattr(config, "fault_tolerance"):
        return config
    workers = int(resolved.get("workers", 2) or 2)
    try:
        return config.fault_tolerance(
            restart_failed_env_runners=True,
            ignore_env_runner_failures=True,
            restart_failed_sub_environments=True,
            env_runner_health_probe_timeout_s=float(
                resolved.get("env_runner_health_timeout_s", 10.0)
            ),
            env_runner_restore_timeout_s=float(
                resolved.get("env_runner_restore_timeout_s", 60.0)
            ),
            num_consecutive_env_runner_failures_tolerance=max(workers, 1) * 4,
        )
    except TypeError:
        return config


def _configure_callbacks(config: Any, callback_cls: Any) -> Any:
    if not hasattr(config, "callbacks"):
        return config
    try:
        return config.callbacks(callbacks_class=callback_cls)
    except TypeError:
        return config.callbacks(callback_cls)


def _build_algorithm(config: Any) -> Any:
    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(algo: Any, logger: logging.Logger, checkpoint_dir: str) -> str:
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


def _save_checkpoint_with_metadata(
    algo: Any,
    logger: logging.Logger,
    checkpoint_dir: str,
    config: dict[str, Any],
    game_key: str,
    *,
    total_steps: int,
    source_checkpoint: str,
) -> str:
    checkpoint_path = _save_checkpoint(algo, logger, checkpoint_dir)
    _write_checkpoint_metadata(
        checkpoint_path, config, game_key,
        total_steps=total_steps, source_checkpoint=source_checkpoint, logger=logger,
    )
    return checkpoint_path


def _write_checkpoint_metadata(
    checkpoint_path: str,
    config: dict[str, Any],
    game_key: str,
    *,
    total_steps: int,
    source_checkpoint: str,
    logger: logging.Logger | None = None,
) -> str:
    metadata_dir = _checkpoint_metadata_dir(checkpoint_path)
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, CHECKPOINT_METADATA_FILENAME)
    payload = _checkpoint_metadata_payload(
        config, game_key,
        total_steps=total_steps, checkpoint_path=checkpoint_path,
        source_checkpoint=source_checkpoint,
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if logger is not None:
        logger.info("Saved checkpoint metadata: %s", metadata_path)
    return metadata_path


def _checkpoint_metadata_payload(
    config: dict[str, Any],
    game_key: str,
    *,
    total_steps: int,
    checkpoint_path: str,
    source_checkpoint: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "framework": "RLlib",
        "algorithm": "PPO",
        "game_version": game_key,
        "training_stage": config.get("_training_stage") or config.get("training_stage") or "",
        "character": config.get("character", ""),
        "multi_character": bool(config.get("multi_character", False)),
        "deck_mode": config.get("deck_mode", "") or "unspecified",
        "enemy_pool": config.get("enemy_pool", "") or "unspecified",
        "total_steps": int(total_steps),
        "source_checkpoint": os.path.abspath(source_checkpoint) if source_checkpoint else None,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "notes": config.get("run_notes", "") or "",
        "heuristic_mode": config.get("heuristic_mode", "none"),
        "heuristic_top_k": max(1, int(config.get("heuristic_top_k", 1) or 1)),
        "seed": config.get("seed"),
        "preset": config.get("_preset_name") or config.get("preset") or "none",
        "transfer_mode": _checkpoint_transfer_mode(config),
        "model": {
            "custom_model": RLLIB_CUSTOM_MODEL_NAME,
            "fcnet_hiddens": [64, 64],
            "vf_share_layers": False,
        },
        "training": {
            "workers": int(config.get("workers", 0) or 0),
            "envs_per_worker": int(config.get("envs_per_worker", 0) or 0),
            "train_batch_size": int(config.get("train_batch_size", 0) or 0),
            "minibatch_size": int(config.get("minibatch_size", 0) or 0),
            "num_epochs": int(config.get("num_epochs", 0) or 0),
            "rollout_fragment_length": int(config.get("rollout_fragment_length", 0) or 0),
            "checkpoint_freq": int(config.get("checkpoint_freq", 0) or 0),
            "eval_combat_episodes": int(config.get("eval_combat_episodes", 0) or 0),
            "eval_combat_freq": int(config.get("eval_combat_freq", 0) or 0),
            "eval_random_baseline": int(config.get("eval_random_baseline", 0) or 0),
            "eval_greedy_baseline": int(config.get("eval_greedy_baseline", 0) or 0),
            "eval_random_baseline_freq": int(config.get("eval_random_baseline_freq", 0) or 0),
            "eval_sts2_recycle_every_episodes": int(
                config.get("eval_sts2_recycle_every_episodes", 0) or 0
            ),
        },
        "engine": {
            "sts2_curriculum_mode": config.get("sts2_curriculum_mode", "full_run"),
            "sts2_reward_mode": config.get("sts2_reward_mode", "full_v3_2"),
            "sts2_combat_room_type": config.get("sts2_combat_room_type", "combat"),
            "sts2_combat_encounter": config.get("sts2_combat_encounter", "SHRINKER_BEETLE_WEAK"),
            "sts2_combat_enemy_pool": config.get("sts2_combat_enemy_pool", "fixed"),
            "sts2_combat_damage_reward_scale": float(
                config.get("sts2_combat_damage_reward_scale", 0.01) or 0.01
            ),
            "sts2_combat_hp_loss_reward_scale": float(
                config.get("sts2_combat_hp_loss_reward_scale", 0.01) or 0.01
            ),
            "sts2_combat_action_penalty": float(
                config.get("sts2_combat_action_penalty", 0.001) or 0.001
            ),
            "sts2_debug_episodes": int(config.get("sts2_debug_episodes", 0) or 0),
            "sts2_deck_duplicate_cap": int(config.get("sts2_deck_duplicate_cap", 2) or 2),
            "sts2_deck_allow_problematic_cards": bool(
                config.get("sts2_deck_allow_problematic_cards", False)
            ),
            "sts2_encoder_mode": config.get("sts2_encoder_mode", "compact"),
            "curriculum_mix": config.get("curriculum_mix", ""),
            "sts2_cli_path": config.get("sts2_cli_path", ""),
            "sts2_cli_cwd": config.get("sts2_cli_cwd", ""),
            "sts2_cli_args": list(config.get("sts2_cli_args") or []),
            "sts2_recycle_every_episodes": int(
                config.get("sts2_recycle_every_episodes", 0) or 0
            ),
            "sts2_recycle_every_steps": int(
                config.get("sts2_recycle_every_steps", 0) or 0
            ),
            "sts2_recycle_rss_mb": float(
                config.get("sts2_recycle_rss_mb", 0.0) or 0.0
            ),
        },
    }


def _checkpoint_metadata_dir(checkpoint_path: str) -> str:
    if os.path.isdir(checkpoint_path):
        return checkpoint_path
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    return parent or os.getcwd()


# ---------------------------------------------------------------------------
# Resume / checkpoint discovery
# ---------------------------------------------------------------------------

def _resolve_resume_path(
    config: dict[str, Any],
    checkpoint_dir: str,
    logger: logging.Logger | None = None,
) -> str:
    resume = config.get("resume_from", "")
    if resume:
        explicit = _resolve_checkpoint_input_path(str(resume))
        auto = "" if config.get("no_auto_resume", False) else _find_latest_rllib_checkpoint(checkpoint_dir)
        if auto and logger is not None:
            logger.info(
                "Explicit --resume-from=%s wins over auto-resume checkpoint in this stage: %s",
                explicit,
                auto,
            )
        return explicit
    if str(config.get("init_from_rllib", "") or "").strip():
        auto = "" if config.get("no_auto_resume", False) else _find_latest_rllib_checkpoint(checkpoint_dir)
        if auto and logger is not None:
            logger.warning(
                "--init-from-rllib requests a fresh stage transfer, so auto-resume checkpoint "
                "in the target stage is ignored: %s",
                auto,
            )
        return ""
    if config.get("no_auto_resume", False):
        return ""
    return _find_latest_rllib_checkpoint(checkpoint_dir)


def _resolve_init_from_rllib_path(config: dict[str, Any]) -> str:
    raw = str(config.get("init_from_rllib", "") or "").strip()
    if not raw:
        return ""
    return _resolve_checkpoint_input_path(raw)


def _resolve_checkpoint_input_path(raw_path: str) -> str:
    path = os.path.abspath(str(raw_path))
    if _is_rllib_checkpoint_dir(path):
        return path
    latest = _find_latest_rllib_checkpoint(path)
    return latest or path


def _find_latest_rllib_checkpoint(checkpoint_dir: str) -> str:
    if _is_rllib_checkpoint_dir(checkpoint_dir):
        return checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        return ""

    candidates: list[str] = []
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path) and _is_rllib_checkpoint_dir(path):
            candidates.append(path)
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def _is_rllib_checkpoint_dir(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "algorithm_state.pkl"))
        or os.path.isfile(os.path.join(path, "rllib_checkpoint.json"))
    )


def _source_checkpoint(
    config: dict[str, Any],
    resume_from: str,
    init_from_rllib: str = "",
) -> str:
    if resume_from:
        return os.path.abspath(resume_from)
    if init_from_rllib:
        return os.path.abspath(init_from_rllib)
    init_sb3 = config.get("init_from_sb3", "")
    if init_sb3:
        return os.path.abspath(str(init_sb3))
    return ""


def _checkpoint_transfer_mode(config: dict[str, Any]) -> str:
    if config.get("resume_from"):
        return "resume_from"
    if config.get("init_from_rllib"):
        return "init_from_rllib"
    if config.get("init_from_sb3"):
        return "init_from_sb3"
    return "fresh_or_auto_resume"


def _init_from_rllib_checkpoint(
    *,
    algo: Any,
    source_checkpoint: str,
    logger: logging.Logger,
) -> None:
    if not _is_rllib_checkpoint_dir(source_checkpoint):
        raise SystemExit(f"--init-from-rllib is not an RLlib checkpoint: {source_checkpoint}")

    from ray.rllib.algorithms.algorithm import Algorithm

    source_algo: Any | None = None
    try:
        logger.info(
            "Initializing fresh Algorithm policy/value weights from RLlib checkpoint: %s",
            source_checkpoint,
        )
        source_algo = Algorithm.from_checkpoint(source_checkpoint)
        _copy_policy_weights(
            target_algo=algo,
            source_algo=source_algo,
            source_checkpoint=source_checkpoint,
        )
        logger.info(
            "Transferred compatible policy/value weights from %s. "
            "Optimizer state and training iteration remain fresh.",
            source_checkpoint,
        )
    finally:
        if source_algo is not None:
            try:
                source_algo.stop()
            except Exception:
                logger.debug("Could not stop temporary source Algorithm.", exc_info=True)


def _copy_policy_weights(
    *,
    target_algo: Any,
    source_algo: Any,
    source_checkpoint: str,
) -> None:
    target_policy = _default_policy(target_algo)
    source_policy = _default_policy(source_algo)
    target_weights = target_policy.get_weights()
    source_weights = source_policy.get_weights()
    _validate_policy_weight_compatibility(
        source_weights,
        target_weights,
        source_checkpoint=source_checkpoint,
    )
    target_policy.set_weights(source_weights)


def _default_policy(algo: Any) -> Any:
    getter = getattr(algo, "get_policy", None)
    if not callable(getter):
        raise SystemExit("RLlib Algorithm does not expose get_policy(); cannot transfer weights.")
    for args in ((), ("default_policy",), ("__default_policy__",)):
        try:
            policy = getter(*args)
        except TypeError:
            continue
        if policy is not None:
            return policy
    raise SystemExit("Could not locate the default RLlib policy for checkpoint transfer.")


def _validate_policy_weight_compatibility(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    *,
    source_checkpoint: str,
) -> None:
    source_keys = set(source_weights)
    target_keys = set(target_weights)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        extra = sorted(source_keys - target_keys)
        raise SystemExit(
            "RLlib policy weights are incompatible with the current model. "
            f"Missing keys={missing[:8]} extra keys={extra[:8]} source={source_checkpoint}"
        )
    mismatches: list[str] = []
    for key in sorted(source_keys):
        source_shape = _weight_shape(source_weights[key])
        target_shape = _weight_shape(target_weights[key])
        if source_shape != target_shape:
            mismatches.append(f"{key}: source={source_shape} target={target_shape}")
    if mismatches:
        details = "; ".join(mismatches[:8])
        raise SystemExit(
            "RLlib policy/value shapes are incompatible with the current stage. "
            "Check encoder_mode/action_space/model architecture. "
            f"{details} source={source_checkpoint}"
        )


def _weight_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except TypeError:
            pass
    try:
        return tuple(int(dim) for dim in np.asarray(value).shape)
    except Exception:
        return ()


def _validate_checkpoint_metadata_compatibility(
    config: dict[str, Any],
    checkpoint_path: str,
    *,
    mode: str,
    logger: logging.Logger,
) -> None:
    metadata = _read_checkpoint_metadata(checkpoint_path)
    if not metadata:
        logger.warning(
            "No checkpoint_metadata.json found for %s; compatibility will be checked by RLlib/weight shapes only.",
            checkpoint_path,
        )
        return

    source_game = str(metadata.get("game_version", "") or "")
    current_game = str(config.get("_game_key", "") or "")
    if source_game and current_game and source_game != current_game:
        raise SystemExit(
            f"Checkpoint game_version mismatch for {mode}: source={source_game} current={current_game}"
        )

    source_engine = metadata.get("engine") if isinstance(metadata.get("engine"), dict) else {}
    source_encoder = str(source_engine.get("sts2_encoder_mode", "") or "")
    current_encoder = str(config.get("sts2_encoder_mode", "") or "")
    if source_encoder and current_encoder and source_encoder != current_encoder:
        raise SystemExit(
            "Checkpoint encoder_mode mismatch. "
            f"source={source_encoder} current={current_encoder}. "
            "Use the same --sts2-encoder-mode or train a separate stage."
        )

    source_model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    if source_model:
        expected_model = {
            "custom_model": RLLIB_CUSTOM_MODEL_NAME,
            "fcnet_hiddens": [64, 64],
            "vf_share_layers": False,
        }
        for key, expected in expected_model.items():
            actual = source_model.get(key)
            if actual is not None and actual != expected:
                raise SystemExit(
                    f"Checkpoint model architecture mismatch for {mode}: "
                    f"{key} source={actual!r} current={expected!r}"
                )


def _warn_resume_stage_safety(
    config: dict[str, Any],
    checkpoint_dir: str,
    resume_from: str,
    logger: logging.Logger,
) -> None:
    metadata = _read_checkpoint_metadata(resume_from)
    source_stage = str(metadata.get("training_stage", "") if metadata else "")
    current_stage = str(config.get("_training_stage", "") or config.get("training_stage", "") or "")
    if source_stage and current_stage and source_stage != current_stage:
        logger.warning(
            "--resume-from restores a full RLlib Algorithm from stage %r while the current stage is %r. "
            "Use --init-from-rllib for curriculum transfer into a fresh stage.",
            source_stage,
            current_stage,
        )
    resolved_resume = os.path.abspath(resume_from)
    resolved_checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.normcase(resolved_resume) == os.path.normcase(resolved_checkpoint_dir):
        if source_stage and current_stage and source_stage != current_stage:
            logger.warning(
                "--resume-from points at the same path as --checkpoint-dir but training_stage changed. "
                "This can overwrite or confuse stage boundaries."
            )


def _read_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any]:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if not metadata_path:
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _checkpoint_metadata_path(checkpoint_path: str) -> str:
    candidates = [
        os.path.join(os.path.abspath(checkpoint_path), CHECKPOINT_METADATA_FILENAME),
        os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)),
            CHECKPOINT_METADATA_FILENAME,
        ),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _algorithm_env_steps(algo: Any) -> int:
    counters = getattr(algo, "_counters", {})
    for key in ("num_env_steps_sampled", "num_agent_steps_sampled"):
        value = counters.get(key) if isinstance(counters, dict) else None
        if value is not None:
            return int(value)
    return 0


def _algorithm_iteration(algo: Any) -> int:
    value = getattr(algo, "training_iteration", None)
    if value is None:
        return 0
    if callable(value):
        value = value()
    return int(value)


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


def _result_env_step_delta(
    result: dict[str, Any],
    *,
    previous_steps: int,
    current_steps: int,
) -> int:
    candidates = (
        result.get("num_env_steps_sampled_this_iter"),
        result.get("timesteps_this_iter"),
        _nested_get(result, ("env_runners", "num_env_steps_sampled_this_iter")),
        _nested_get(result, ("sampler_results", "num_env_steps_sampled")),
    )
    for value in candidates:
        if value is not None:
            return max(0, int(value))
    return max(0, current_steps - previous_steps)


def _progress_log_metrics(result: dict[str, Any]) -> dict[str, str]:
    return {
        "floor_mean": _format_metric(_custom_metric(result, "floor_mean")),
        "max_floor": _format_metric(
            _custom_metric(result, "max_floor_max")
            if _custom_metric(result, "max_floor_max") is not None
            else _custom_metric(result, "floor_max")
        ),
        "boss_reached_pct": _format_metric(_custom_metric(result, "boss_reached_pct_mean")),
        "boss_killed_pct": _format_metric(_custom_metric(result, "boss_killed_pct_mean")),
        "act2_pct": _format_metric(_custom_metric(result, "act2_pct_mean")),
    }


def _combat_log_metrics(result: dict[str, Any]) -> dict[str, str]:
    return {
        "combat_win_rate": _format_metric(_custom_metric(result, "combat_win_rate_mean")),
        "combat_loss_rate": _format_metric(_custom_metric(result, "combat_loss_rate_mean")),
        "combat_timeout_rate": _format_metric(
            _custom_metric(result, "combat_timeout_rate_mean")
        ),
        "avg_combat_steps": _format_metric(_custom_metric(result, "avg_combat_steps_mean")),
        "avg_hp_remaining_on_win": _format_metric(
            _custom_metric(result, "avg_hp_remaining_on_win_mean")
        ),
        "avg_hp_lost": _format_metric(_custom_metric(result, "avg_hp_lost_mean")),
        "avg_monster_hp_remaining_on_loss": _format_metric(
            _custom_metric(result, "avg_monster_hp_remaining_on_loss_mean")
        ),
        "avg_boss_hp_remaining_on_loss": _format_metric(
            _custom_metric(result, "boss_hp_remaining_on_loss_mean")
        ),
        "avg_boss_hp_fraction_removed": _format_metric(
            _custom_metric(result, "boss_hp_fraction_removed_mean")
        ),
        "avg_min_boss_hp_reached": _format_metric(
            _custom_metric(result, "min_boss_hp_reached_mean")
        ),
        "avg_damage_dealt_total": _format_metric(
            _custom_metric(result, "damage_dealt_total_mean")
        ),
        "avg_turns_survived": _format_metric(_custom_metric(result, "turns_survived_mean")),
        "end_turn_with_energy_rate": _format_metric(
            _custom_metric(result, "end_turn_with_energy_rate_mean")
        ),
        "end_turn_with_playable_attack_rate": _format_metric(
            _custom_metric(result, "end_turn_with_playable_attack_rate_mean")
        ),
        "end_turn_with_playable_block_when_incoming_damage_rate": _format_metric(
            _custom_metric(
                result,
                "end_turn_with_playable_block_when_incoming_damage_rate_mean",
            )
        ),
        "power_play_rate": _format_metric(_custom_metric(result, "power_play_rate_mean")),
        "block_when_incoming_damage_rate": _format_metric(
            _custom_metric(result, "block_when_incoming_damage_rate_mean")
        ),
        "avg_deck_size": _format_metric(_custom_metric(result, "deck_size_mean")),
        "encounters": _prefixed_custom_metrics(result, "encounter_id_"),
        "terminated_reasons": _prefixed_custom_metrics(result, "terminated_reason_"),
        "cards_played_by_id": _prefixed_custom_metrics(result, "card_played_"),
        "win_rate_by_boss": _boss_by_encounter_metrics(result, "win_rate"),
        "hp_lost_by_boss": _boss_by_encounter_metrics(result, "hp_lost"),
        "boss_hp_remaining_by_boss": _boss_by_encounter_metrics(
            result,
            "hp_remaining_on_loss",
        ),
    }


def _grouped_combat_log_metrics(result: dict[str, Any]) -> dict[str, str]:
    """Extract grouped category metrics (weak/normal/elite/boss)."""
    metrics: dict[str, str] = {}
    for category in ("weak", "normal", "elite", "boss"):
        wins = _custom_metric(result, f"{category}_win_count_mean")
        hp_sum = _custom_metric(result, f"{category}_hp_lost_sum_mean")
        cnt = _custom_metric(result, f"{category}_encounter_count_mean")
        wr = _ratio_or_none(wins, cnt)
        hp = _ratio_or_none(hp_sum, cnt)
        if wr is None:
            wr = _custom_metric(result, f"{category}_win_rate_mean")
        if hp is None:
            hp = _custom_metric(result, f"{category}_avg_hp_lost_mean")
        metrics[f"{category}_win_rate"] = _format_metric(wr)
        metrics[f"{category}_avg_hp_lost"] = _format_metric(hp)
        metrics[f"{category}_encounter_count"] = _format_metric(cnt)
    return metrics


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    try:
        if numerator is None or denominator is None:
            return None
        denom = float(denominator)
        if denom <= 0.0:
            return None
        return float(numerator) / denom
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _boss_by_encounter_metrics(result: dict[str, Any], metric: str) -> str:
    labels = _boss_metric_labels(result)
    items: list[str] = []
    for label in labels:
        fights = _custom_metric(result, f"boss_{label}_fight_count_mean")
        if fights is None:
            continue
        try:
            if float(fights) <= 0.0:
                continue
        except (TypeError, ValueError):
            continue
        if metric == "win_rate":
            value = _ratio_or_none(
                _custom_metric(result, f"boss_{label}_win_count_mean"),
                fights,
            )
        elif metric == "hp_lost":
            value = _ratio_or_none(
                _custom_metric(result, f"boss_{label}_hp_lost_sum_mean"),
                fights,
            )
        elif metric == "hp_remaining_on_loss":
            value = _ratio_or_none(
                _custom_metric(result, f"boss_{label}_hp_remaining_on_loss_sum_mean"),
                fights,
            )
        else:
            value = None
        items.append(f"{label}:{_format_metric(value)}")
    return ",".join(items) if items else "n/a"


def _boss_metric_labels(result: dict[str, Any]) -> list[str]:
    metric_sources: list[dict[str, Any]] = []
    custom_metrics = result.get("custom_metrics")
    if isinstance(custom_metrics, dict):
        metric_sources.append(custom_metrics)
    env_runners = result.get("env_runners")
    if isinstance(env_runners, dict):
        env_custom_metrics = env_runners.get("custom_metrics")
        if isinstance(env_custom_metrics, dict):
            metric_sources.append(env_custom_metrics)
        metric_sources.append(env_runners)
    labels: list[str] = []
    seen: set[str] = set()
    prefix = "boss_"
    suffix = "_fight_count_mean"
    for source in metric_sources:
        for key in sorted(source):
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            label = key[len(prefix): -len(suffix)]
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _custom_metric(result: dict[str, Any], key: str) -> Any:
    custom_metrics = result.get("custom_metrics")
    if isinstance(custom_metrics, dict) and key in custom_metrics:
        return custom_metrics[key]

    env_runners = result.get("env_runners")
    if isinstance(env_runners, dict):
        custom_metrics = env_runners.get("custom_metrics")
        if isinstance(custom_metrics, dict) and key in custom_metrics:
            return custom_metrics[key]
        if key in env_runners:
            return env_runners[key]

    return None


def _prefixed_custom_metrics(result: dict[str, Any], prefix: str) -> str:
    metric_sources: list[dict[str, Any]] = []
    custom_metrics = result.get("custom_metrics")
    if isinstance(custom_metrics, dict):
        metric_sources.append(custom_metrics)
    env_runners = result.get("env_runners")
    if isinstance(env_runners, dict):
        env_custom_metrics = env_runners.get("custom_metrics")
        if isinstance(env_custom_metrics, dict):
            metric_sources.append(env_custom_metrics)
        metric_sources.append(env_runners)

    items: list[str] = []
    seen: set[str] = set()
    for source in metric_sources:
        for key, value in sorted(source.items()):
            if not key.startswith(prefix):
                continue
            label = key[len(prefix):]
            suffix = "_mean"
            if not label.endswith(suffix):
                continue
            label = label[: -len(suffix)]
            if label in seen:
                continue
            try:
                if float(value) <= 0.0:
                    continue
            except (TypeError, ValueError):
                pass
            seen.add(label)
            items.append(f"{label}:{_format_metric(value)}")
    return ",".join(items) if items else "n/a"


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.2f}"


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Full verbose file log (always written regardless of console mode)
# ---------------------------------------------------------------------------

def _log_full_iteration(
    *,
    logger: logging.Logger,
    result: dict[str, Any],
    current_steps: int,
    step_delta: int,
    iteration_seconds: float,
    reward: Any,
    progress: dict[str, str],
    combat: dict[str, str],
    grouped: dict[str, str],
) -> None:
    """Always write full iteration details to the log file."""
    sps = step_delta / iteration_seconds if iteration_seconds > 0 else 0.0
    logger.info(
        (
            "RLlib iteration=%s env_steps=%d (+%d) iter_s=%.2f "
            "steps/sec=%.1f reward_mean=%s "
            "floor_mean=%s max_floor=%s boss_reached%%=%s "
            "boss_killed%%=%s act2%%=%s combat_win_rate=%s "
            "combat_loss_rate=%s combat_timeout_rate=%s avg_combat_steps=%s "
            "avg_hp_lost=%s avg_boss_hp_remaining_on_loss=%s "
            "avg_boss_hp_fraction_removed=%s avg_damage_dealt_total=%s "
            "avg_turns_survived=%s end_turn_energy_rate=%s "
            "end_turn_playable_attack_rate=%s block_when_incoming_rate=%s "
            "power_play_rate=%s "
            "weak_wr=%s normal_wr=%s elite_wr=%s boss_wr=%s "
            "win_rate_by_boss=%s boss_hp_remaining_by_boss=%s "
            "encounters=%s reasons=%s cards_played=%s"
        ),
        result.get("training_iteration"),
        current_steps,
        step_delta,
        iteration_seconds,
        sps,
        reward,
        progress["floor_mean"],
        progress["max_floor"],
        progress["boss_reached_pct"],
        progress["boss_killed_pct"],
        progress["act2_pct"],
        combat["combat_win_rate"],
        combat["combat_loss_rate"],
        combat["combat_timeout_rate"],
        combat["avg_combat_steps"],
        combat["avg_hp_lost"],
        combat.get("avg_boss_hp_remaining_on_loss", "n/a"),
        combat.get("avg_boss_hp_fraction_removed", "n/a"),
        combat.get("avg_damage_dealt_total", "n/a"),
        combat.get("avg_turns_survived", "n/a"),
        combat.get("end_turn_with_energy_rate", "n/a"),
        combat.get("end_turn_with_playable_attack_rate", "n/a"),
        combat.get("block_when_incoming_damage_rate", "n/a"),
        combat.get("power_play_rate", "n/a"),
        grouped.get("weak_win_rate", "n/a"),
        grouped.get("normal_win_rate", "n/a"),
        grouped.get("elite_win_rate", "n/a"),
        grouped.get("boss_win_rate", "n/a"),
        combat.get("win_rate_by_boss", "n/a"),
        combat.get("boss_hp_remaining_by_boss", "n/a"),
        combat["encounters"],
        combat["terminated_reasons"],
        combat.get("cards_played_by_id", "n/a"),
    )


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _run_random_combat_baseline(
    *,
    env_config: dict[str, Any],
    episodes: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    rng = np.random.default_rng(12345)

    def select_random_action(observation: dict[str, Any]) -> int:
        mask = np.asarray(observation.get("action_mask", []), dtype=np.float32)
        valid = np.flatnonzero(mask > 0.0)
        if len(valid) == 0:
            return 0
        return int(rng.choice(valid))

    from rllib.env_wrapper import make_sts_rllib_env

    metrics = _run_manual_combat_eval(
        env_config=env_config,
        episodes=episodes,
        worker_id=900001,
        action_selector=select_random_action,
    )
    logger.info(
        (
            "Random combat baseline: episodes=%d random_combat_win_rate=%s "
            "random_avg_combat_steps=%s random_avg_hp_lost=%s "
            "random_boss_hp_remaining_on_loss=%s random_win_rate_by_encounter=%s "
            "random_boss_hp_remaining_by_encounter=%s"
        ),
        episodes,
        _format_metric(metrics["combat_win_rate"]),
        _format_metric(metrics["avg_combat_steps"]),
        _format_metric(metrics["avg_hp_lost"]),
        _format_metric(metrics["avg_boss_hp_remaining_on_loss"]),
        metrics["win_rate_by_encounter"],
        metrics["avg_boss_hp_remaining_by_encounter"],
    )
    return metrics


def _run_greedy_combat_baseline(
    *,
    env_config: dict[str, Any],
    episodes: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    def select_greedy_action(observation: dict[str, Any], env: Any) -> int:
        return _select_greedy_combat_action(observation, env)

    metrics = _run_manual_combat_eval(
        env_config=env_config,
        episodes=episodes,
        worker_id=900003,
        action_selector=select_greedy_action,
    )
    logger.info(
        (
            "Greedy combat baseline: episodes=%d greedy_combat_win_rate=%s "
            "greedy_avg_combat_steps=%s greedy_avg_hp_lost=%s "
            "greedy_boss_hp_remaining_on_loss=%s greedy_win_rate_by_encounter=%s "
            "greedy_boss_hp_remaining_by_encounter=%s"
        ),
        episodes,
        _format_metric(metrics["combat_win_rate"]),
        _format_metric(metrics["avg_combat_steps"]),
        _format_metric(metrics["avg_hp_lost"]),
        _format_metric(metrics["avg_boss_hp_remaining_on_loss"]),
        metrics["win_rate_by_encounter"],
        metrics["avg_boss_hp_remaining_by_encounter"],
    )
    return metrics


def _select_greedy_combat_action(observation: dict[str, Any], env: Any) -> int:
    mask = np.asarray(observation.get("action_mask", []), dtype=np.float32)
    valid = set(int(idx) for idx in np.flatnonzero(mask > 0.0))
    if not valid:
        return 0
    base_env = getattr(env, "env", env)
    state = getattr(base_env, "current_state", {}) or {}
    try:
        from sts2.action_space import END_TURN_ACTION, build_legal_commands
    except Exception:
        return min(valid)
    commands = build_legal_commands(state)
    enemies = _greedy_enemies(state)
    incoming = _greedy_incoming_damage(state)
    player_block = _greedy_player_block(state)

    best_action = min(valid)
    best_score = -1e9
    for action_id in sorted(valid):
        command = commands.get(action_id)
        score = 0.0
        if isinstance(command, dict) and command.get("action") == "play_card":
            card = _greedy_card_for_command(state, command)
            card_type = str(card.get("type") or "").lower() if isinstance(card, dict) else ""
            damage = _greedy_card_number(card, "damage")
            block = _greedy_card_number(card, "block")
            target_hp = _greedy_target_hp(enemies, command)
            if card_type == "power":
                score += 35.0
            if damage > 0:
                score += 5.0 + damage
                if target_hp is not None and damage >= target_hp:
                    score += 100.0
            if block > 0:
                score += block * (2.0 if incoming > player_block else 0.6)
            cost = _greedy_card_number(card, "cost")
            score -= 0.05 * max(0.0, cost)
        elif isinstance(command, dict) and command.get("action") == "use_potion":
            score += 20.0
        elif action_id == END_TURN_ACTION:
            score -= 15.0 if incoming > player_block else 2.0
        else:
            score -= 1.0
        if score > best_score:
            best_score = score
            best_action = action_id
    return int(best_action)


def _greedy_card_for_command(state: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    args = command.get("args")
    card_index = -1
    if isinstance(args, dict):
        try:
            card_index = int(args.get("card_index", -1))
        except (TypeError, ValueError):
            card_index = -1
    hand = _greedy_hand(state)
    for slot, card in enumerate(hand):
        if not isinstance(card, dict):
            continue
        try:
            idx = int(card.get("index", slot))
        except (TypeError, ValueError):
            idx = slot
        if idx == card_index:
            return card
    return {}


def _greedy_hand(state: dict[str, Any]) -> list[Any]:
    if isinstance(state.get("hand"), list):
        return list(state["hand"])
    game_state = state.get("game_state", {})
    if isinstance(game_state, dict):
        combat_state = game_state.get("combat_state", {})
        if isinstance(combat_state, dict) and isinstance(combat_state.get("hand"), list):
            return list(combat_state["hand"])
    return []


def _greedy_enemies(state: dict[str, Any]) -> list[Any]:
    if isinstance(state.get("enemies"), list):
        return list(state["enemies"])
    game_state = state.get("game_state", {})
    if isinstance(game_state, dict):
        combat_state = game_state.get("combat_state", {})
        if isinstance(combat_state, dict) and isinstance(combat_state.get("monsters"), list):
            return list(combat_state["monsters"])
    return []


def _greedy_player_block(state: dict[str, Any]) -> int:
    player = state.get("player")
    if isinstance(player, dict):
        return int(player.get("block") or 0)
    game_state = state.get("game_state", {})
    if isinstance(game_state, dict):
        combat_state = game_state.get("combat_state", {})
        if isinstance(combat_state, dict) and isinstance(combat_state.get("player"), dict):
            return int(combat_state["player"].get("block") or 0)
    return 0


def _greedy_incoming_damage(state: dict[str, Any]) -> int:
    total = 0
    for enemy in _greedy_enemies(state):
        if not isinstance(enemy, dict) or enemy.get("is_gone", False):
            continue
        for intent in enemy.get("intents") or []:
            if isinstance(intent, dict) and "attack" in str(intent.get("type") or "").lower():
                total += int(intent.get("damage") or 0)
        for key in ("intent_damage", "damage"):
            if enemy.get(key) is not None:
                total += int(enemy.get(key) or 0)
                break
    return max(0, total)


def _greedy_target_hp(enemies: list[Any], command: dict[str, Any]) -> int | None:
    args = command.get("args")
    if not isinstance(args, dict) or args.get("target_index") is None:
        return None
    try:
        target_index = int(args.get("target_index"))
    except (TypeError, ValueError):
        return None
    if target_index < 0 or target_index >= len(enemies):
        return None
    enemy = enemies[target_index]
    if not isinstance(enemy, dict):
        return None
    return int(enemy.get("current_hp", enemy.get("hp", 0)) or 0)


def _greedy_card_number(card: Any, field: str) -> float:
    if not isinstance(card, dict):
        return 0.0
    stats = card.get("stats")
    value = stats.get(field) if isinstance(stats, dict) else None
    if value is None:
        value = card.get(field)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _run_policy_combat_eval(
    *,
    algo: Any,
    env_config: dict[str, Any],
    episodes: int,
    deterministic: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    def select_policy_action(observation: dict[str, Any]) -> int:
        try:
            result = algo.compute_single_action(
                observation,
                explore=not deterministic,
            )
        except TypeError:
            result = algo.compute_single_action(observation)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, np.ndarray):
            return int(result.item() if result.shape == () else result[0])
        return int(result)

    metrics = _run_manual_combat_eval(
        env_config=env_config,
        episodes=episodes,
        worker_id=900002,
        action_selector=select_policy_action,
    )
    logger.info(
        (
            "PPO combat eval: episodes=%d deterministic=%s eval_combat_win_rate=%s "
            "eval_avg_combat_steps=%s eval_avg_hp_lost=%s "
            "eval_boss_hp_remaining_on_loss=%s eval_win_rate_by_encounter=%s "
            "eval_avg_hp_lost_by_encounter=%s eval_boss_hp_remaining_by_encounter=%s "
            "eval_avg_steps_by_encounter=%s"
        ),
        episodes,
        bool(deterministic),
        _format_metric(metrics["combat_win_rate"]),
        _format_metric(metrics["avg_combat_steps"]),
        _format_metric(metrics["avg_hp_lost"]),
        _format_metric(metrics["avg_boss_hp_remaining_on_loss"]),
        metrics["win_rate_by_encounter"],
        metrics["avg_hp_lost_by_encounter"],
        metrics["avg_boss_hp_remaining_by_encounter"],
        metrics["avg_steps_by_encounter"],
    )
    return metrics


def _run_manual_combat_eval(
    *,
    env_config: dict[str, Any],
    episodes: int,
    worker_id: int,
    action_selector: Any,
) -> dict[str, Any]:
    from rllib.env_wrapper import make_sts_rllib_env
    from rllib.progress_metrics import classify_encounter

    config = dict(env_config)
    config["worker_id"] = worker_id
    config["sts2_curriculum_mode"] = "combat"
    if config.get("sts2_reward_mode") == "full_v3_2":
        config["sts2_reward_mode"] = "combat_sparse"
    config["debug_env_info"] = False
    eval_recycle_episodes = config.pop("eval_sts2_recycle_every_episodes", None)
    if eval_recycle_episodes is not None:
        config["sts2_recycle_every_episodes"] = max(0, int(eval_recycle_episodes))
    env = make_sts_rllib_env(config)
    stats = _new_combat_eval_stats()
    try:
        for _episode_index in range(max(0, int(episodes))):
            observation, info = env.reset()
            final_info = info
            terminated = False
            truncated = False
            guard = 0
            while not (terminated or truncated):
                try:
                    action = int(action_selector(observation, env))
                except TypeError:
                    action = int(action_selector(observation))
                observation, _reward, terminated, truncated, info = env.step(action)
                final_info = info
                guard += 1
                if guard > 1000:
                    final_info = dict(final_info or {})
                    final_info["combat_done_reason"] = "timeout"
                    break
            _record_combat_eval_episode(stats, final_info)
    finally:
        env.close()
    return _finalize_combat_eval_stats(stats)


def _new_combat_eval_stats() -> dict[str, Any]:
    return {
        "episodes": 0,
        "wins": 0.0,
        "losses": 0.0,
        "timeouts": 0.0,
        "steps": 0.0,
        "hp_lost": 0.0,
        "boss_hp_remaining_on_loss": 0.0,
        "damage_dealt_total": 0.0,
        "by_encounter": {},
        "by_category": {},
    }


def _record_combat_eval_episode(stats: dict[str, Any], info: Any) -> None:
    from rllib.progress_metrics import classify_encounter

    progress = {}
    if isinstance(info, dict) and isinstance(info.get("progress_metrics"), dict):
        progress = info["progress_metrics"]
    reason = str(progress.get("terminated_reason") or "").strip().lower()
    if not reason and isinstance(info, dict):
        reason = str(info.get("combat_done_reason") or "").strip().lower()
    reason = reason or "unknown"
    encounter = str(progress.get("encounter_id") or "unknown")
    steps = _safe_number(progress.get("combat_steps"))
    hp_lost = _safe_number(progress.get("hp_lost"))
    boss_hp_remaining = _safe_number(progress.get("boss_hp_remaining_on_loss"))
    damage_dealt_total = _safe_number(progress.get("damage_dealt_total"))
    category = classify_encounter(encounter)

    stats["episodes"] += 1
    stats["wins"] += float(reason == "win")
    stats["losses"] += float(reason == "loss")
    stats["timeouts"] += float(reason == "timeout")
    stats["steps"] += steps
    stats["hp_lost"] += hp_lost
    stats["boss_hp_remaining_on_loss"] += boss_hp_remaining
    stats["damage_dealt_total"] += damage_dealt_total

    by_encounter = stats["by_encounter"].setdefault(
        encounter,
        {
            "episodes": 0,
            "wins": 0.0,
            "steps": 0.0,
            "hp_lost": 0.0,
            "boss_hp_remaining_on_loss": 0.0,
            "damage_dealt_total": 0.0,
        },
    )
    by_encounter["episodes"] += 1
    by_encounter["wins"] += float(reason == "win")
    by_encounter["steps"] += steps
    by_encounter["hp_lost"] += hp_lost
    by_encounter["boss_hp_remaining_on_loss"] += boss_hp_remaining
    by_encounter["damage_dealt_total"] += damage_dealt_total

    # Category-level grouping
    by_category = stats["by_category"].setdefault(
        category,
        {
            "episodes": 0,
            "wins": 0.0,
            "steps": 0.0,
            "hp_lost": 0.0,
            "boss_hp_remaining_on_loss": 0.0,
            "damage_dealt_total": 0.0,
        },
    )
    by_category["episodes"] += 1
    by_category["wins"] += float(reason == "win")
    by_category["steps"] += steps
    by_category["hp_lost"] += hp_lost
    by_category["boss_hp_remaining_on_loss"] += boss_hp_remaining
    by_category["damage_dealt_total"] += damage_dealt_total


def _finalize_combat_eval_stats(stats: dict[str, Any]) -> dict[str, Any]:
    episodes = max(1, int(stats["episodes"]))
    result: dict[str, Any] = {
        "episodes": int(stats["episodes"]),
        "combat_win_rate": float(stats["wins"]) / episodes,
        "combat_loss_rate": float(stats["losses"]) / episodes,
        "combat_timeout_rate": float(stats["timeouts"]) / episodes,
        "avg_combat_steps": float(stats["steps"]) / episodes,
        "avg_hp_lost": float(stats["hp_lost"]) / episodes,
        "avg_boss_hp_remaining_on_loss": float(stats["boss_hp_remaining_on_loss"]) / episodes,
        "avg_damage_dealt_total": float(stats["damage_dealt_total"]) / episodes,
        "win_rate_by_encounter": _format_by_encounter(stats, "wins", "episodes"),
        "avg_hp_lost_by_encounter": _format_by_encounter(stats, "hp_lost", "episodes"),
        "avg_boss_hp_remaining_by_encounter": _format_by_encounter(
            stats,
            "boss_hp_remaining_on_loss",
            "episodes",
        ),
        "avg_damage_dealt_by_encounter": _format_by_encounter(
            stats,
            "damage_dealt_total",
            "episodes",
        ),
        "avg_steps_by_encounter": _format_by_encounter(stats, "steps", "episodes"),
    }

    # Grouped category metrics
    grouped: dict[str, Any] = {}
    for cat in ("weak", "normal", "elite", "boss"):
        cat_stats = stats.get("by_category", {}).get(cat, {})
        cat_episodes = max(1, int(cat_stats.get("episodes", 0)))
        cat_any = int(cat_stats.get("episodes", 0)) > 0
        grouped[f"{cat}_win_rate"] = float(cat_stats.get("wins", 0.0)) / cat_episodes if cat_any else None
        grouped[f"{cat}_avg_hp_lost"] = float(cat_stats.get("hp_lost", 0.0)) / cat_episodes if cat_any else None
        grouped[f"{cat}_avg_boss_hp_remaining_on_loss"] = (
            float(cat_stats.get("boss_hp_remaining_on_loss", 0.0)) / cat_episodes
            if cat_any else None
        )
        grouped[f"{cat}_avg_damage_dealt_total"] = (
            float(cat_stats.get("damage_dealt_total", 0.0)) / cat_episodes
            if cat_any else None
        )
        grouped[f"{cat}_episodes"] = int(cat_stats.get("episodes", 0))
    result["grouped_metrics"] = grouped

    return result


def _format_by_encounter(
    stats: dict[str, Any],
    numerator_key: str,
    denominator_key: str,
) -> str:
    items: list[str] = []
    for encounter, values in sorted(stats["by_encounter"].items()):
        denominator = max(1, int(values.get(denominator_key, 0)))
        value = float(values.get(numerator_key, 0.0)) / denominator
        items.append(f"{encounter}:{value:.2f}")
    return ",".join(items) if items else "n/a"


def _safe_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class _TrainHeartbeat:
    """Background logger that makes long algo.train() calls visible."""

    def __init__(self, logger: logging.Logger, interval_s: float) -> None:
        self._logger = logger
        self._interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._next_iteration: int | None = None
        self._last_report_at = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._interval_s <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="rllib-train-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def begin(self, next_iteration: int) -> None:
        if self._interval_s <= 0:
            return
        now = time.perf_counter()
        with self._lock:
            self._started_at = now
            self._next_iteration = next_iteration
            self._last_report_at = now

    def end(self) -> None:
        if self._interval_s <= 0:
            return
        with self._lock:
            self._started_at = None
            self._next_iteration = None
            self._last_report_at = 0.0

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        poll_s = min(max(self._interval_s / 4.0, 1.0), 5.0)
        while not self._stop_event.wait(poll_s):
            now = time.perf_counter()
            with self._lock:
                started_at = self._started_at
                next_iteration = self._next_iteration
                last_report_at = self._last_report_at
                if started_at is None:
                    continue
                elapsed = now - started_at
                if elapsed < self._interval_s or now - last_report_at < self._interval_s:
                    continue
                self._last_report_at = now

            self._logger.warning(
                (
                    "RLlib algo.train() still running: next_iteration=%s elapsed_s=%.1f. "
                    "This usually means the trainer is waiting for rollout sampling or worker cleanup."
                ),
                next_iteration,
                elapsed,
            )


if __name__ == "__main__":
    main()
