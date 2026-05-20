"""
train_cluster.py — Parallel local cluster training for Slay the Spire.

Orchestrates N isolated workers running SlayTheSpireEnv in parallel
using stable-baselines3's SubprocVecEnv. Handles round-robin character
assignment, directory cloning/isolation, and clean subprocess teardown.
"""

import sys
import os
import stat
import shutil
import logging
import argparse
import datetime
import traceback
import multiprocessing
from typing import List, Callable

# Setup centralized logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

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
        description="Local STS Parallel Cluster Training with SubprocVecEnv"
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


def make_env(worker_id: int, assigned_char: str, worker_dir: str, use_xvfb: bool) -> Callable:
    """Closure factory to create a single wrapped environment."""
    def _init():
        # Late imports inside the worker process to avoid issues under spawn start method
        from env import SlayTheSpireEnv
        from sb3_contrib.common.wrappers import ActionMasker

        env = SlayTheSpireEnv(
            character_class=assigned_char,
            worker_dir=worker_dir,
            use_xvfb=use_xvfb,
        )
        # Note: env.reset() will automatically launch the Java subprocess
        return ActionMasker(env, lambda _: env.get_action_mask())

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
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    logger.info("Imports completed.")

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
        env_fns.append(make_env(i, assigned_char, worker_dir, args.use_xvfb))

    # 3. Vectorized Environment Setup
    # Determine the safest multiprocessing start method
    # Linux supports 'fork' (much faster startup); Windows/macOS MUST use 'spawn'.
    start_method = "fork" if sys.platform != "win32" else "spawn"
    logger.info("Creating SubprocVecEnv with %d workers (start method: %s)...", args.workers, start_method)
    
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)

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
            callback=checkpoint_callback,
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

        # Robust cleanup of SubprocVecEnv workers
        logger.info("Closing SubprocVecEnv workers and terminating Java subprocesses...")
        try:
            vec_env.close()
        except Exception as e:
            logger.error("Error closing SubprocVecEnv: %s", e)
        logger.info("Cleanup completed.")


if __name__ == "__main__":
    # Multi-processing check for Windows/macOS spawn
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        logger.critical("FATAL: Unhandled training exception:\n%s", traceback.format_exc())
        sys.exit(1)
