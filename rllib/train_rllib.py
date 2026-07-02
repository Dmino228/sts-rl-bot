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
        resume_from = _resolve_resume_path(config, checkpoint_dir)
        if resume_from:
            logger.info("Restoring RLlib checkpoint from %s", resume_from)
            algo.restore(resume_from)
        source_checkpoint = _source_checkpoint(config, resume_from)

        current_steps = _algorithm_env_steps(algo)
        logger.info("Current RLlib env timesteps before training: %d", current_steps)

        init_sb3 = str(config.get("init_from_sb3", "") or "")
        if init_sb3 and not resume_from:
            from rllib.sb3_transfer import try_transfer_sb3_policy

            try_transfer_sb3_policy(algo, init_sb3, logger)
        elif init_sb3:
            logger.info("Ignoring --init-from-sb3 because an RLlib checkpoint was restored.")

        # Random baseline at startup
        eval_random = int(config.get("eval_random_baseline", 0) or 0)
        if game_key == "sts2" and not config.get("smoke_test") and eval_random > 0:
            _run_random_combat_baseline(env_config=env_config, episodes=eval_random, logger=logger)

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

def _resolve_resume_path(config: dict[str, Any], checkpoint_dir: str) -> str:
    resume = config.get("resume_from", "")
    if resume:
        return os.path.abspath(str(resume))
    if config.get("no_auto_resume", False):
        return ""
    return _find_latest_rllib_checkpoint(checkpoint_dir)


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


def _source_checkpoint(config: dict[str, Any], resume_from: str) -> str:
    if resume_from:
        return os.path.abspath(resume_from)
    init_sb3 = config.get("init_from_sb3", "")
    if init_sb3:
        return os.path.abspath(str(init_sb3))
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
        "encounters": _prefixed_custom_metrics(result, "encounter_id_"),
        "terminated_reasons": _prefixed_custom_metrics(result, "terminated_reason_"),
    }


def _grouped_combat_log_metrics(result: dict[str, Any]) -> dict[str, str]:
    """Extract grouped category metrics (weak/normal/elite/boss)."""
    metrics: dict[str, str] = {}
    for category in ("weak", "normal", "elite", "boss"):
        wr = _custom_metric(result, f"{category}_win_rate_mean")
        hp = _custom_metric(result, f"{category}_avg_hp_lost_mean")
        cnt = _custom_metric(result, f"{category}_encounter_count_mean")
        metrics[f"{category}_win_rate"] = _format_metric(wr)
        metrics[f"{category}_avg_hp_lost"] = _format_metric(hp)
        metrics[f"{category}_encounter_count"] = _format_metric(cnt)
    return metrics


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
            "avg_hp_lost=%s "
            "weak_wr=%s normal_wr=%s elite_wr=%s boss_wr=%s "
            "encounters=%s reasons=%s"
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
        grouped.get("weak_win_rate", "n/a"),
        grouped.get("normal_win_rate", "n/a"),
        grouped.get("elite_win_rate", "n/a"),
        grouped.get("boss_win_rate", "n/a"),
        combat["encounters"],
        combat["terminated_reasons"],
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
            "random_win_rate_by_encounter=%s"
        ),
        episodes,
        _format_metric(metrics["combat_win_rate"]),
        _format_metric(metrics["avg_combat_steps"]),
        _format_metric(metrics["avg_hp_lost"]),
        metrics["win_rate_by_encounter"],
    )
    return metrics


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
            "eval_win_rate_by_encounter=%s eval_avg_hp_lost_by_encounter=%s "
            "eval_avg_steps_by_encounter=%s"
        ),
        episodes,
        bool(deterministic),
        _format_metric(metrics["combat_win_rate"]),
        _format_metric(metrics["avg_combat_steps"]),
        _format_metric(metrics["avg_hp_lost"]),
        metrics["win_rate_by_encounter"],
        metrics["avg_hp_lost_by_encounter"],
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
    category = classify_encounter(encounter)

    stats["episodes"] += 1
    stats["wins"] += float(reason == "win")
    stats["losses"] += float(reason == "loss")
    stats["timeouts"] += float(reason == "timeout")
    stats["steps"] += steps
    stats["hp_lost"] += hp_lost

    by_encounter = stats["by_encounter"].setdefault(
        encounter,
        {"episodes": 0, "wins": 0.0, "steps": 0.0, "hp_lost": 0.0},
    )
    by_encounter["episodes"] += 1
    by_encounter["wins"] += float(reason == "win")
    by_encounter["steps"] += steps
    by_encounter["hp_lost"] += hp_lost

    # Category-level grouping
    by_category = stats["by_category"].setdefault(
        category,
        {"episodes": 0, "wins": 0.0, "steps": 0.0, "hp_lost": 0.0},
    )
    by_category["episodes"] += 1
    by_category["wins"] += float(reason == "win")
    by_category["steps"] += steps
    by_category["hp_lost"] += hp_lost


def _finalize_combat_eval_stats(stats: dict[str, Any]) -> dict[str, Any]:
    episodes = max(1, int(stats["episodes"]))
    result: dict[str, Any] = {
        "episodes": int(stats["episodes"]),
        "combat_win_rate": float(stats["wins"]) / episodes,
        "combat_loss_rate": float(stats["losses"]) / episodes,
        "combat_timeout_rate": float(stats["timeouts"]) / episodes,
        "avg_combat_steps": float(stats["steps"]) / episodes,
        "avg_hp_lost": float(stats["hp_lost"]) / episodes,
        "win_rate_by_encounter": _format_by_encounter(stats, "wins", "episodes"),
        "avg_hp_lost_by_encounter": _format_by_encounter(stats, "hp_lost", "episodes"),
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
