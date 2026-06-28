"""Ray RLlib training entrypoint for the STS RL bot."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import threading
import time
from typing import Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_factory import normalize_game_version
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
    parser = argparse.ArgumentParser(description="Ray RLlib training for STS")
    parser.add_argument("--workers", type=int, default=2, help="Ray EnvRunner/rollout workers.")
    parser.add_argument("--envs-per-worker", type=int, default=1, help="Vector envs per Ray worker.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Additional env steps to train.")
    parser.add_argument("--base-env-dir", default=os.path.join(PROJECT_ROOT, "SlayTheSpire"))
    parser.add_argument("--workspace-dir", default=os.path.join(PROJECT_ROOT, "rllib_workers"))
    parser.add_argument("--game-version", default="1", choices=["1", "2", "sts1", "sts2"])
    parser.add_argument("--sts2-cli-path", default="sts2-cli")
    parser.add_argument(
        "--sts2-cli-cwd",
        default="",
        help=(
            "Working directory for sts2-cli/dotnet. Use the sts2-cli repo root "
            "when relying on its global.json."
        ),
    )
    parser.add_argument(
        "--sts2-cli-arg",
        action="append",
        default=[],
        dest="sts2_cli_args",
        help="Extra argument passed to sts2-cli. Can be repeated.",
    )
    parser.add_argument(
        "--character",
        default="IRONCLAD",
        help=(
            "Character/run archetype to request from the selected engine. "
            "StS1 validates IRONCLAD/SILENT/DEFECT/WATCHER."
        ),
    )
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--sts2-lang", default="en")
    parser.add_argument(
        "--sts2-capture-stderr",
        action="store_true",
        help="Write each sts2-cli worker stderr stream to sts2-cli.stderr.log for debugging.",
    )
    parser.add_argument(
        "--multi-character",
        action="store_true",
        help="Round-robin the default roster for the selected game version.",
    )
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
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "RLlib checkpoint directory. Defaults to models/rllib/sts1 or "
            "models/rllib/sts2 based on --game-version."
        ),
    )
    parser.add_argument("--resume-from", default="", help="Path to an RLlib checkpoint directory.")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not automatically resume from the default per-game RLlib checkpoint directory.",
    )
    parser.add_argument("--init-from-sb3", default="", help="Optional SB3 .zip checkpoint for warm-start weights.")
    parser.add_argument(
        "--process-timeout-s",
        type=float,
        default=None,
        help="Per-env game I/O timeout. Defaults to 120s for StS1 and 30s for StS2.",
    )
    parser.add_argument(
        "--sample-timeout-s",
        type=float,
        default=None,
        help=(
            "Seconds Ray waits for workers to produce rollout fragments. "
            "Defaults are resolved per game version."
        ),
    )

    parser.add_argument(
        "--train-heartbeat-s",
        type=float,
        default=30.0,
        help="Log a warning every N seconds while algo.train() is still running. Use 0 to disable.",
    )
    parser.add_argument(
        "--slow-iteration-s",
        type=float,
        default=60.0,
        help="Warn when one returned RLlib train iteration exceeds this duration. Use 0 to disable.",
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=float,
        default=1.0,
        help="Ray CPU resources reserved per rollout worker actor.",
    )
    parser.add_argument(
        "--disable-env-runner-fault-tolerance",
        action="store_true",
        help="Disable RLlib EnvRunner restart/ignore fault-tolerance settings.",
    )
    parser.add_argument(
        "--env-runner-health-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for Ray EnvRunner health probes when fault tolerance is enabled.",
    )
    parser.add_argument(
        "--env-runner-restore-timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for Ray EnvRunner restoration when fault tolerance is enabled.",
    )

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

    from rllib.action_mask_model import ACTION_MASK_MODEL, register_action_mask_model

    logger.info("Starting RLlib training session %s", TIMESTAMP)
    logger.info("Log file: %s", log_file)
    logger.info(
        "Algorithm=PPO workers=%d envs_per_worker=%d timesteps=%d",
        args.workers,
        args.envs_per_worker,
        args.timesteps,
    )

    game_key = _checkpoint_game_key(args)
    args.process_timeout_s = _resolve_process_timeout(args, game_key)
    args.sample_timeout_s = _resolve_sample_timeout(args, game_key)
    checkpoint_dir = _resolve_checkpoint_dir(args, game_key)
    logger.info("Checkpoint directory: %s", checkpoint_dir)
    logger.info(
        "Timeouts: process=%.1fs sample=%.1fs heartbeat=%.1fs env_runner_health=%.1fs env_runner_restore=%.1fs",
        args.process_timeout_s,
        args.sample_timeout_s,
        args.train_heartbeat_s,
        args.env_runner_health_timeout_s,
        args.env_runner_restore_timeout_s,
    )
    _warn_if_worker_count_is_aggressive(args, game_key, logger)

    ray.init(ignore_reinit_error=True, log_to_driver=True)
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
            "game_version": args.game_version,
            "character_class": args.character,
            "multi_character": args.multi_character,
            "ram_usage": args.ram_usage,
            "base_port": args.base_port,
            "use_xvfb": args.use_xvfb,
            "force_rebuild": args.force_rebuild,
            "debug_env_info": args.debug_env_info,
            "num_envs_per_env_runner": args.envs_per_worker,
            "sts2_cli_path": args.sts2_cli_path,
            "sts2_cli_args": args.sts2_cli_args,
            "sts2_cli_cwd": args.sts2_cli_cwd,
            "sts2_capture_stderr": args.sts2_capture_stderr,
            "process_timeout": args.process_timeout_s,
            "ascension": args.ascension,
            "sts2_lang": args.sts2_lang,
        }

    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig()
    config = _configure_api_stack(config)
    config = config.environment(env=env_name, env_config=env_config)
    config = config.framework("torch")
    config = _configure_rollout_workers(config, args)
    config = _configure_training(config, args)
    config = _configure_resources(config, args)
    config = _configure_fault_tolerance(config, args)
    config.model["custom_model"] = ACTION_MASK_MODEL
    config.model["fcnet_hiddens"] = [64, 64]
    config.model["vf_share_layers"] = False

    algo = _build_algorithm(config)
    heartbeat = _TrainHeartbeat(logger, args.train_heartbeat_s)
    heartbeat.start()
    try:
        resume_from = _resolve_resume_path(args, checkpoint_dir)
        if resume_from:
            logger.info("Restoring RLlib checkpoint from %s", resume_from)
            algo.restore(resume_from)

        current_steps = _algorithm_env_steps(algo)
        logger.info("Current RLlib env timesteps before training: %d", current_steps)

        if args.init_from_sb3 and not resume_from:
            try_transfer_sb3_policy(algo, args.init_from_sb3, logger)
        elif args.init_from_sb3:
            logger.info("Ignoring --init-from-sb3 because an RLlib checkpoint was restored.")

        target_steps = current_steps + args.timesteps
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
            steps_per_second = step_delta / iteration_seconds if iteration_seconds > 0 else 0.0
            ms_per_step = 1000.0 / steps_per_second if steps_per_second > 0 else float("inf")
            reward = _nested_get(result, ("env_runners", "episode_return_mean"))
            if reward is None:
                reward = result.get("episode_reward_mean")
            logger.info(
                (
                    "RLlib iteration=%s env_steps=%d (+%d) iter_s=%.2f "
                    "steps/sec=%.1f avg_step_ms=%.2f reward_mean=%s"
                ),
                result.get("training_iteration"),
                current_steps,
                step_delta,
                iteration_seconds,
                steps_per_second,
                ms_per_step,
                reward,
            )
            if args.slow_iteration_s > 0 and iteration_seconds > args.slow_iteration_s:
                logger.warning(
                    (
                        "Slow RLlib iteration: %.1fs for %d env steps. "
                        "If this repeats, lower --process-timeout-s/--sample-timeout-s "
                        "or reduce --train-batch-size/--rollout-fragment-length while diagnosing stragglers."
                    ),
                    iteration_seconds,
                    step_delta,
                )
            iteration = int(result.get("training_iteration", 0) or 0)
            if args.checkpoint_freq > 0 and iteration % args.checkpoint_freq == 0:
                checkpoint_path = _save_checkpoint(algo, logger, checkpoint_dir)
                logger.info("Saved RLlib checkpoint: %s", checkpoint_path)

        checkpoint_path = _save_checkpoint(algo, logger, checkpoint_dir)
        logger.info("Training complete. Final RLlib checkpoint: %s", checkpoint_path)
    except KeyboardInterrupt:
        logger.warning("Training interrupted. Saving RLlib checkpoint...")
        logger.info("Saved RLlib checkpoint: %s", _save_checkpoint(algo, logger, checkpoint_dir))
    finally:
        heartbeat.stop()
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
                sample_timeout_s=args.sample_timeout_s,
            )
        except (TypeError, ValueError):
            pass
    if hasattr(config, "env_runners"):
        return config.env_runners(
            num_env_runners=args.workers,
            num_envs_per_env_runner=args.envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            sample_timeout_s=args.sample_timeout_s,
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
        if float(args.cpus_per_worker) == 1.0:
            return config.resources(num_gpus=args.num_gpus)
        try:
            return config.resources(
                num_gpus=args.num_gpus,
                num_cpus_per_worker=args.cpus_per_worker,
            )
        except TypeError:
            return config.resources(num_gpus=args.num_gpus)
    return config


def _configure_fault_tolerance(config: Any, args: argparse.Namespace) -> Any:
    if args.disable_env_runner_fault_tolerance or not hasattr(config, "fault_tolerance"):
        return config
    try:
        return config.fault_tolerance(
            restart_failed_env_runners=True,
            ignore_env_runner_failures=True,
            restart_failed_sub_environments=True,
            env_runner_health_probe_timeout_s=args.env_runner_health_timeout_s,
            env_runner_restore_timeout_s=args.env_runner_restore_timeout_s,
            num_consecutive_env_runner_failures_tolerance=max(args.workers, 1) * 4,
        )
    except TypeError:
        return config


def _build_algorithm(config: Any) -> Any:
    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


def _save_checkpoint(
    algo: Any,
    logger: logging.Logger,
    checkpoint_dir: str,
) -> str:
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


def _checkpoint_game_key(args: argparse.Namespace) -> str:
    if args.smoke_test:
        return "smoke"
    return normalize_game_version(args.game_version)


def _resolve_checkpoint_dir(args: argparse.Namespace, game_key: str) -> str:
    if args.checkpoint_dir:
        return os.path.abspath(args.checkpoint_dir)
    return os.path.join(MODELS_DIR, "rllib", game_key)


def _resolve_resume_path(args: argparse.Namespace, checkpoint_dir: str) -> str:
    if args.resume_from:
        return os.path.abspath(args.resume_from)
    if args.no_auto_resume:
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


def _resolve_process_timeout(args: argparse.Namespace, game_key: str) -> float:
    if args.process_timeout_s is not None:
        return float(args.process_timeout_s)
    if game_key == "sts2":
        return 30.0
    return 120.0


def _resolve_sample_timeout(args: argparse.Namespace, game_key: str) -> float:
    if args.sample_timeout_s is not None:
        return float(args.sample_timeout_s)
    if game_key == "sts2":
        return 15.0
    if game_key == "smoke":
        return 60.0
    return 600.0


def _warn_if_worker_count_is_aggressive(
    args: argparse.Namespace,
    game_key: str,
    logger: logging.Logger,
) -> None:
    logical_cpus = os.cpu_count() or 1
    env_count = max(args.workers, 0) * max(args.envs_per_worker, 1)
    if game_key == "sts2" and env_count >= logical_cpus:
        logger.warning(
            (
                "StS2 worker count is at or above logical CPU count: %d envs on %d CPUs. "
                "Each env also owns an external C# process, so 16 workers on an 8C/16T "
                "CPU can expose scheduler stalls. Treat 8-12 workers as the first "
                "performance sweep before trying 16 again."
            ),
            env_count,
            logical_cpus,
        )


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


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


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
