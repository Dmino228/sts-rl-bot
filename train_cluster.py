"""
train_cluster.py — Parallel local cluster training for Slay the Spire.

Orchestrates N isolated workers running SlayTheSpireEnv in parallel
using a Stable Baselines3 VecEnv backend. Handles round-robin character
assignment, directory cloning/isolation, and clean teardown.
"""

import sys
import os
import glob
import stat
import shutil
import logging
import argparse
import datetime
import traceback
import multiprocessing
import time
from typing import List, Callable

# Setup centralized logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Cleanup old log files
for pattern in ["training_*.log", "cluster_training_*.log"]:
    for f in glob.glob(os.path.join(LOGS_DIR, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"cluster_training_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("train_cluster")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local STS Parallel Cluster Training"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel environment workers (default: 4).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1,000,000).",
    )
    parser.add_argument(
        "--base-env-dir", default=os.path.join(BASE_DIR, "SlayTheSpire"),
        help="Path to the master portable Slay the Spire folder.",
    )
    parser.add_argument(
        "--workspace-dir", default=os.path.join(BASE_DIR, "cluster_workers"),
        help="Directory where worker directories are stored.",
    )
    parser.add_argument(
        "--save-freq", type=int, default=10_000,
        help="Checkpoint save frequency (steps per worker).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="PPO rollout buffer size per worker.",
    )
    parser.add_argument(
        "--use-xvfb", action="store_true",
        help="Wrap Java launch in xvfb-run (required on headless Linux).",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Force rebuild of worker directories (delete and copy again).",
    )
    parser.add_argument(
        "--debug-env-info", action="store_true",
        help="Include full raw_state in VecEnv infos for debugging (slower IPC).",
    )
    parser.add_argument(
        "--progress-log-freq", type=int, default=250,
        help="Log rollout throughput every N vector steps (0 disables).",
    )
    parser.add_argument(
        "--throughput-file", type=str, default="",
        help="Optional path to a CSV file where overall steps per second will be appended.",
    )
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "threaded", "subproc"],
        default="auto",
        help="VecEnv backend. auto uses Dummy for 1 worker and Threaded on Windows.",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="Limit PyTorch CPU worker threads used by the coordinator process.",
    )
    return parser.parse_args()


def clean_worker_state(worker_dir: str) -> None:
    """Clear volatile saves, runs, and logs from a worker directory."""
    def remove_readonly(func, path, excinfo):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    for subdir in ["saves", "runs", "sendToDevs", "logs"]:
        path = os.path.join(worker_dir, subdir)
        if os.path.isdir(path):
            try:
                shutil.rmtree(path, onerror=remove_readonly)
            except Exception as e:
                logger.warning("Could not clean worker folder %s: %s", path, e)
        os.makedirs(path, exist_ok=True)


def make_env(
    worker_id: int,
    assigned_char: str,
    worker_dir: str,
    use_xvfb: bool,
    debug_env_info: bool,
) -> Callable:
    """Closure factory to create a single wrapped environment."""
    def _init():
        # Late imports inside the worker process to avoid issues under spawn start method
        from env import SlayTheSpireEnv
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker

        base_env = SlayTheSpireEnv(
            character_class=assigned_char,
            worker_dir=worker_dir,
            use_xvfb=use_xvfb,
            include_raw_state_in_info=debug_env_info,
            include_action_mask_in_info=True,
        )
        # Monitor must wrap BEFORE ActionMasker so that info["episode"]
        # (populated on done=True) propagates through VecEnv to SB3's logger.
        env = Monitor(base_env)
        # Note: env.reset() will automatically launch the Java subprocess
        return ActionMasker(env, lambda _: base_env.get_action_mask())

    return _init


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("STS Parallel Local Cluster Training Started")
    logger.info("Timestamp: %s", TIMESTAMP)
    logger.info("Workers: %d", args.workers)
    logger.info("Target Timesteps: %d", args.timesteps)
    logger.info("Log file: %s", log_file)
    logger.info("=" * 60)

    # 1. Late imports of ML/RL heavy dependencies
    logger.info("Loading PyTorch and Stable Baselines3 ...")
    import torch as th
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    from mask_cache_vec_env import CachedActionMaskVecEnv
    from threaded_vec_env import ThreadedVecEnv
    logger.info("Imports completed.")

    if args.torch_threads > 0:
        th.set_num_threads(args.torch_threads)
        try:
            th.set_num_interop_threads(1)
        except RuntimeError:
            pass
        logger.info(
            "PyTorch CPU threads limited: num_threads=%d interop_threads=1",
            args.torch_threads,
        )

    class ClusterProgressCallback(BaseCallback):
        """Log rollout throughput before a full PPO iteration completes."""

        def __init__(self, log_freq: int = 250, throughput_file: str = "") -> None:
            super().__init__()
            self.log_freq = log_freq
            self.throughput_file = throughput_file
            self._last_time = time.perf_counter()
            self._last_timesteps = 0

            if self.throughput_file and not os.path.exists(self.throughput_file):
                os.makedirs(os.path.dirname(os.path.abspath(self.throughput_file)), exist_ok=True)
                with open(self.throughput_file, "w", encoding="utf-8") as f:
                    f.write("timestamp,total_timesteps,env_steps_per_sec\n")

        def _on_training_start(self) -> None:
            self._last_time = time.perf_counter()
            self._last_timesteps = self.num_timesteps

        def _on_step(self) -> bool:
            if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
                now = time.perf_counter()
                elapsed = max(now - self._last_time, 1e-9)
                delta = self.num_timesteps - self._last_timesteps
                steps_per_sec = delta / elapsed
                logger.info(
                    "Rollout progress: timesteps=%d (+%d), %.2f env steps/s over %d vector steps",
                    self.num_timesteps,
                    delta,
                    steps_per_sec,
                    self.log_freq,
                )
                if self.throughput_file:
                    try:
                        with open(self.throughput_file, "a", encoding="utf-8") as f:
                            f.write(f"{time.time():.3f},{self.num_timesteps},{steps_per_sec:.2f}\n")
                    except Exception as e:
                        logger.error("Failed to write to throughput file: %s", e)
                
                self._last_time = now
                self._last_timesteps = self.num_timesteps
            return True

    # 2. Worker folder setup and character assignment
    os.makedirs(args.workspace_dir, exist_ok=True)
    worker_dirs = []
    env_fns = []
    
    chars = ["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]

    for i in range(args.workers):
        worker_dir = os.path.join(args.workspace_dir, f"worker_{i}")
        assigned_char = chars[i % len(chars)]
        
        # Directory isolation and copying
        if os.path.exists(worker_dir):
            if args.force_rebuild:
                logger.info("Removing existing worker_%d directory...", i)
                def remove_readonly(func, path, excinfo):
                    try:
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    except Exception:
                        pass
                shutil.rmtree(worker_dir, onerror=remove_readonly)
                os.makedirs(worker_dir, exist_ok=True)
                logger.info("Copying master env -> worker_%d...", i)
                shutil.copytree(args.base_env_dir, worker_dir, dirs_exist_ok=True)
            else:
                logger.info("Worker_%d directory already exists. Skipping copy.", i)
        else:
            logger.info("Creating worker_%d directory...", i)
            os.makedirs(worker_dir, exist_ok=True)
            logger.info("Copying master env -> worker_%d...", i)
            shutil.copytree(args.base_env_dir, worker_dir, dirs_exist_ok=True)

        # Clear any state leftover from template
        clean_worker_state(worker_dir)
        worker_dirs.append(worker_dir)

        # Append to environment list
        env_fns.append(
            make_env(
                i,
                assigned_char,
                worker_dir,
                args.use_xvfb,
                args.debug_env_info,
            )
        )

    # 3. Vectorized Environment Setup
    backend = args.vec_env
    if backend == "auto":
        if args.workers == 1:
            backend = "dummy"
        elif sys.platform == "win32":
            backend = "threaded"
        else:
            backend = "subproc"

    if backend == "dummy":
        logger.info("Creating DummyVecEnv with %d worker(s)...", args.workers)
        vec_env_base = DummyVecEnv(env_fns)
    elif backend == "threaded":
        logger.info("Creating ThreadedVecEnv with %d workers...", args.workers)
        vec_env_base = ThreadedVecEnv(env_fns)
    else:
        # Linux supports 'fork' (much faster startup); Windows/macOS MUST use 'spawn'.
        start_method = "fork" if sys.platform != "win32" else "spawn"
        logger.info(
            "Creating SubprocVecEnv with %d workers (start method: %s)...",
            args.workers,
            start_method,
        )
        vec_env_base = SubprocVecEnv(env_fns, start_method=start_method)

    vec_env = CachedActionMaskVecEnv(vec_env_base)

    # 4. Centralized TensorBoard and model checkpoint configuration
    tensorboard_path = os.path.join(BASE_DIR, "logs", "ppo_sts_cluster")
    models_path = os.path.join(BASE_DIR, "models")
    os.makedirs(models_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=models_path,
        name_prefix="ppo_sts_cluster",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList([
        checkpoint_callback,
        ClusterProgressCallback(
            log_freq=args.progress_log_freq,
            throughput_file=args.throughput_file,
        ),
    ])

    # Initialize model
    logger.info("Initializing MaskablePPO model...")
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=args.n_steps,
        tensorboard_log=tensorboard_path,
    )

    # Centralize logger output
    custom_logger = configure(tensorboard_path, ["tensorboard", "csv", "stdout"])
    model.set_logger(custom_logger)

    # 5. Training Loop with Try-Finally shutdown protection
    try:
        logger.info("Starting training run for %d total timesteps...", args.timesteps)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
        )
        logger.info("Training finished successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Saving progress...")
    except Exception as e:
        logger.error("Training crashed: %s", e, exc_info=True)
        raise
    finally:
        # Save model
        final_model_file = os.path.join(models_path, "ppo_sts_cluster_final")
        try:
            model.save(final_model_file)
            logger.info("Model saved to %s.zip", final_model_file)
        except Exception as e:
            logger.error("Failed to save final model: %s", e)

        # Robust cleanup of VecEnv workers
        logger.info("Closing VecEnv workers and terminating Java subprocesses...")
        try:
            vec_env.close()
        except Exception as e:
            logger.error("Error closing VecEnv: %s", e)
        logger.info("Cleanup completed.")


if __name__ == "__main__":
    # Multi-processing check for Windows/macOS spawn
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        logger.critical("FATAL: Unhandled training exception:\n%s", traceback.format_exc())
        sys.exit(1)
