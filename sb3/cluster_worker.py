"""
cluster_worker — Per-worker entrypoint for Colab SubprocVecEnv training.

This module is imported by the auto-generated agent_shim.py inside each
worker directory. It runs a single training loop as a CommunicationMod
subprocess within that worker's isolated game instance.
"""

import sys
import os
import logging

logger = logging.getLogger(__name__)


def worker_main(worker_id: int, worker_dir: str) -> None:
    """Main loop for a single SubprocVecEnv worker process.

    This is called when CommunicationMod launches the worker's agent_shim.py.
    The handshake ("ready") has already been sent by the shim.

    In SubprocVecEnv mode, this function is NOT typically invoked directly —
    the env is created inline via ClusterManager.make_vec_env(). This
    entrypoint exists as a fallback for CommunicationMod-launched mode
    where the game process starts the Python script.

    Args:
        worker_id: Integer identifier for this worker (0-indexed).
        worker_dir: Absolute path to this worker's isolated game directory.
    """
    # Configure logging for this worker
    log_dir = os.path.join(worker_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[worker_{worker_id}] [%(asctime)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, f"worker_{worker_id}.log"),
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.__stderr__),
        ],
    )

    logger.info("Worker %d starting in %s", worker_id, worker_dir)

    # Import env (project root was already added to sys.path by shim)
    from env import SlayTheSpireEnv

    env = SlayTheSpireEnv(
        character_class="IRONCLAD",  # overridden by SubprocVecEnv factory
        worker_dir=worker_dir,
    )

    # In CommunicationMod-launched mode, we operate on stdin/stdout
    # (process_manager defaults to sys.stdin / sys.__stdout__)
    try:
        obs, info = env.reset()
        logger.info("Worker %d: initial reset complete.", worker_id)

        step = 0
        while True:
            # In worker mode, we just keep the env alive for SubprocVecEnv
            # The actual action selection is done by the vectorized policy
            import numpy as np

            mask = info.get("action_mask", np.zeros(100, dtype=np.int8))
            valid = np.where(mask == 1)[0]
            action = int(np.random.choice(valid)) if len(valid) > 0 else 98

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                logger.info(
                    "Worker %d: episode ended at step %d. Resetting...",
                    worker_id, step,
                )
                obs, info = env.reset()
                step = 0

    except (EOFError, BrokenPipeError):
        logger.info("Worker %d: pipe closed, shutting down.", worker_id)
    except Exception as e:
        logger.error("Worker %d: fatal error: %s", worker_id, e, exc_info=True)
    finally:
        env.close()
        logger.info("Worker %d: shutdown complete.", worker_id)
