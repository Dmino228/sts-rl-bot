"""
train_colab.py — Colab-optimized MaskablePPO training with SubprocVecEnv.

This is the Colab counterpart to train.py. Instead of being launched BY
CommunicationMod, this script IS the parent process that spawns N game
instances in parallel using ClusterManager.

Usage (in Colab cell):
    !python train_colab.py --num-workers 2 --character IRONCLAD --timesteps 500000
"""

import sys
import os
import glob
import datetime
import logging
import traceback
import argparse

# ──────────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_DIR = PROJECT_ROOT
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"colab_training_{TIMESTAMP}.log")

tensorboard_path = os.path.join(BASE_DIR, "ppo_sts_tensorboard")
models_path = os.path.join(BASE_DIR, "models")
os.makedirs(tensorboard_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
train_logger = logging.getLogger("train_colab")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Colab STS Training with SubprocVecEnv"
    )
    parser.add_argument(
        "--base-env-dir",
        default="/content/sts_env",
        help="Path to the extracted base STS environment.",
    )
    parser.add_argument(
        "--workspace-dir",
        default="/content/workers",
        help="Parent directory for worker dirs.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of parallel environment workers.",
    )
    parser.add_argument(
        "--character",
        default="IRONCLAD",
        choices=["IRONCLAD", "SILENT", "DEFECT", "WATCHER"],
        help="Default character class for all workers.",
    )
    parser.add_argument(
        "--multi-character",
        action="store_true",
        help="Cycle through all 4 characters across workers.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of worker directories.",
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="PPO rollout buffer size per worker.",
    )
    parser.add_argument(
        "--save-freq", type=int, default=2048,
        help="Checkpoint save frequency (steps per worker).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_logger.info("=" * 60)
    train_logger.info("Colab MaskablePPO Training — session %s", TIMESTAMP)
    train_logger.info("Workers: %d | Character: %s | Timesteps: %d",
                      args.num_workers, args.character, args.timesteps)
    train_logger.info("Log file: %s", log_file)
    train_logger.info("=" * 60)

    # ── 1. Import heavy deps ──
    train_logger.info("Loading PyTorch + Stable Baselines3 ...")
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    from sb3.cluster_manager import ClusterManager
    train_logger.info("Imports complete.")

    # ── 2. Build character schedule ──
    characters = ["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]
    if args.multi_character:
        character_schedule = [
            characters[i % len(characters)]
            for i in range(args.num_workers)
        ]
        train_logger.info("Multi-character schedule: %s", character_schedule)
    else:
        character_schedule = None

    # ── 3. Initialize cluster ──
    cluster = ClusterManager(
        base_env_dir=args.base_env_dir,
        workspace_dir=args.workspace_dir,
        num_workers=args.num_workers,
        character_class=args.character,
        use_xvfb=True,
    )

    cluster.initialize_workers(
        force_rebuild=args.force_rebuild,
        character_schedule=character_schedule,
    )

    # Log status
    import json
    train_logger.info("Cluster status:\n%s", json.dumps(cluster.status(), indent=2))

    # ── 4. Create SubprocVecEnv ──
    vec_env = cluster.make_vec_env(character_schedule=character_schedule)
    train_logger.info("SubprocVecEnv created with %d workers.", args.num_workers)

    # ── 5. Checkpoint callback ──
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=models_path,
        name_prefix="ppo_sts_colab",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ── 6. Initialize or resume model ──
    final_model_path = os.path.join(models_path, "ppo_sts_final.zip")
    checkpoints = glob.glob(os.path.join(models_path, "ppo_sts_*_steps.zip"))

    if os.path.exists(final_model_path):
        latest_model = final_model_path
    elif checkpoints:
        latest_model = max(checkpoints, key=os.path.getmtime)
    else:
        latest_model = None

    if latest_model:
        train_logger.info("Resuming from %s ...", latest_model)
        model = MaskablePPO.load(
            latest_model,
            env=vec_env,
            custom_objects={"n_steps": args.n_steps},
            tensorboard_log=tensorboard_path,
        )
    else:
        train_logger.info("Initializing new MaskablePPO model ...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=args.n_steps,
            tensorboard_log=tensorboard_path,
        )

    custom_logger = configure(tensorboard_path, ["tensorboard", "csv", "stdout"])
    model.set_logger(custom_logger)

    # ── 7. Train ──
    try:
        train_logger.info("Starting training for %d timesteps ...", args.timesteps)
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )
        train_logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        train_logger.warning("Training interrupted (Ctrl+C). Saving...")
    except EOFError:
        train_logger.warning("Worker pipe broken. Saving...")
    except Exception as e:
        train_logger.error("Training crashed: %s", e, exc_info=True)
        raise
    finally:
        model.save(os.path.join(models_path, "ppo_sts_final"))
        train_logger.info("Model saved to %s", os.path.join(models_path, "ppo_sts_final"))
        vec_env.close()
        cluster.cleanup(remove_dirs=False)
        train_logger.info("Cleanup complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        train_logger.critical(
            "FATAL — unhandled exception:\n%s", traceback.format_exc()
        )
        sys.exit(1)
