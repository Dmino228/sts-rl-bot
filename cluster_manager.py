"""
ClusterManager — Parallel STS environment orchestration for Colab.

Manages N isolated worker directories, each containing a full copy of
the portable game environment. Integrates with SubprocVecEnv for
parallel MaskablePPO training.

Architecture:
  ┌──────────────┐
  │ ClusterMgr   │──► worker_0/  (full game copy, unique saves/config)
  │              │──► worker_1/  (full game copy, unique saves/config)
  │              │──► worker_N/  (full game copy, unique saves/config)
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ SubprocVecEnv│  N processes, each running SlayTheSpireEnv
  └──────────────┘
"""

import os
import sys
import shutil
import logging
import time
from typing import Optional, List, Callable, Dict, Any

logger = logging.getLogger(__name__)


class ClusterManager:
    """Manages isolated worker environments for parallel STS training.

    Each worker gets a full copy of the base game environment to prevent
    Java ConcurrentModificationException from shared save/config files.
    """

    def __init__(
        self,
        base_env_dir: str,
        workspace_dir: str,
        num_workers: int = 2,
        character_class: str = "IRONCLAD",
        use_xvfb: bool = True,
        python_command: str = "python3",
    ) -> None:
        """
        Args:
            base_env_dir: Path to the extracted base STS environment
                          (the unpacked sts_env_v1.zip contents).
            workspace_dir: Parent directory where worker_N dirs are created.
            num_workers: Number of parallel environments (N).
            character_class: Default character for all workers.
                             Can be overridden per-worker via character_schedule.
            use_xvfb: Wrap Java launch in xvfb-run (required on headless Linux).
            python_command: Python executable for CommunicationMod config.
        """
        self.base_env_dir = os.path.abspath(base_env_dir)
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.num_workers = num_workers
        self.character_class = character_class.upper()
        self.use_xvfb = use_xvfb
        self.python_command = python_command

        self.worker_dirs: List[str] = []
        self._initialized = False

    # ──────────────────────────────────────────────────────────────
    # WORKER DIRECTORY MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def get_worker_dir(self, worker_id: int) -> str:
        """Return the absolute path for a given worker directory."""
        return os.path.join(self.workspace_dir, f"worker_{worker_id}")

    def initialize_workers(
        self,
        force_rebuild: bool = False,
        character_schedule: Optional[List[str]] = None,
    ) -> List[str]:
        """Create N isolated worker directories by copying the base environment.

        Args:
            force_rebuild: If True, delete and recreate existing worker dirs.
            character_schedule: Optional list of character classes per worker.
                                Length must equal num_workers if provided.
                                e.g. ["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]

        Returns:
            List of absolute paths to worker directories.
        """
        if character_schedule is not None:
            if len(character_schedule) != self.num_workers:
                raise ValueError(
                    f"character_schedule length ({len(character_schedule)}) "
                    f"must match num_workers ({self.num_workers})"
                )

        os.makedirs(self.workspace_dir, exist_ok=True)
        self.worker_dirs = []

        for i in range(self.num_workers):
            worker_dir = self.get_worker_dir(i)
            char = (
                character_schedule[i].upper()
                if character_schedule
                else self.character_class
            )

            if os.path.exists(worker_dir):
                if force_rebuild:
                    logger.info("[CLUSTER] Removing existing worker_%d ...", i)
                    shutil.rmtree(worker_dir)
                else:
                    logger.info(
                        "[CLUSTER] worker_%d already exists, skipping copy.", i
                    )
                    self._patch_communicationmod_config(worker_dir, i)
                    self.worker_dirs.append(worker_dir)
                    continue

            logger.info(
                "[CLUSTER] Copying base env → worker_%d (char=%s)...", i, char
            )
            t0 = time.time()
            shutil.copytree(
                self.base_env_dir,
                worker_dir,
                symlinks=False,
                dirs_exist_ok=False,
            )
            elapsed = time.time() - t0
            logger.info(
                "[CLUSTER] worker_%d ready (%.1fs copy time).", i, elapsed
            )

            # Patch CommunicationMod config to point to the correct Python script
            self._patch_communicationmod_config(worker_dir, i)

            # Clear any stale saves/runs from the template
            self._clean_worker_state(worker_dir)

            self.worker_dirs.append(worker_dir)

        self._initialized = True
        logger.info(
            "[CLUSTER] %d workers initialized in %s",
            self.num_workers, self.workspace_dir,
        )
        return self.worker_dirs

    def _patch_communicationmod_config(
        self, worker_dir: str, worker_id: int
    ) -> None:
        """Write CommunicationMod config.properties for this worker.

        The config tells CommunicationMod which external process to launch
        for bidirectional communication. We point it to a shim script that
        imports and runs our training agent.
        """
        # Locate the script directory (where env.py, train.py, etc. live)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create a per-worker launcher shim
        shim_path = os.path.join(worker_dir, "agent_shim.py")
        shim_content = f'''#!/usr/bin/env python3
"""Auto-generated worker shim for worker_{worker_id}."""
import sys
import os

# Add project root to path
sys.path.insert(0, {repr(script_dir)})

# Signal ready immediately
sys.__stdout__.write("ready\\n")
sys.__stdout__.flush()

# Import and run the worker entrypoint
from cluster_worker import worker_main

worker_main(
    worker_id={worker_id},
    worker_dir={repr(worker_dir)},
)
'''
        with open(shim_path, "w", encoding="utf-8") as f:
            f.write(shim_content)
        os.chmod(shim_path, 0o755)

        # Write CommunicationMod config
        config_path = os.path.join(worker_dir, "config.properties")
        config_content = (
            f"# Auto-generated by ClusterManager for worker_{worker_id}\n"
            f"command={self.python_command} {shim_path}\n"
            f"runAtStartup=true\n"
        )
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        logger.debug(
            "[CLUSTER] Patched config.properties for worker_%d", worker_id
        )

    def _clean_worker_state(self, worker_dir: str) -> None:
        """Remove stale game saves and run history from a worker directory."""
        for subdir in ["saves", "runs", "sendToDevs", "logs"]:
            path = os.path.join(worker_dir, subdir)
            if os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
                logger.debug("[CLUSTER] Cleaned %s", path)

    # ──────────────────────────────────────────────────────────────
    # SubprocVecEnv FACTORY
    # ──────────────────────────────────────────────────────────────

    def make_vec_env(
        self,
        character_schedule: Optional[List[str]] = None,
    ):
        """Create a SubprocVecEnv with N parallel SlayTheSpireEnv instances.

        Each environment is pointed to its isolated worker directory and
        wrapped with the ActionMasker for MaskablePPO compatibility.

        Args:
            character_schedule: Optional per-worker character override.

        Returns:
            A SubprocVecEnv ready for MaskablePPO.learn().
        """
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from mask_cache_vec_env import CachedActionMaskVecEnv

        if not self._initialized:
            self.initialize_workers(character_schedule=character_schedule)

        def _make_env_fn(
            worker_id: int, worker_dir: str, character: str
        ) -> Callable:
            """Closure that creates a single env for SubprocVecEnv."""

            def _init():
                from env import SlayTheSpireEnv
                from stable_baselines3.common.monitor import Monitor

                env = SlayTheSpireEnv(
                    character_class=character,
                    worker_dir=worker_dir,
                    use_xvfb=self.use_xvfb,
                    include_raw_state_in_info=False,
                    include_action_mask_in_info=True,
                )

                # Launch the game subprocess (Python-as-parent mode)
                env.process_manager.launch_game()
                env.process_manager.signal_ready()

                # Monitor tracks episode rewards/lengths for TensorBoard
                env = Monitor(env)

                # Wrap for MaskablePPO
                return ActionMasker(env, lambda _: env.get_action_mask())

            return _init

        env_fns = []
        for i in range(self.num_workers):
            worker_dir = self.worker_dirs[i]
            char = (
                character_schedule[i].upper()
                if character_schedule
                else self.character_class
            )
            env_fns.append(_make_env_fn(i, worker_dir, char))

        logger.info(
            "[CLUSTER] Creating SubprocVecEnv with %d workers...",
            self.num_workers,
        )
        return CachedActionMaskVecEnv(SubprocVecEnv(env_fns, start_method="fork"))

    # ──────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────

    def cleanup(self, remove_dirs: bool = False) -> None:
        """Shutdown all workers and optionally remove directories.

        Args:
            remove_dirs: If True, delete all worker directories.
        """
        if remove_dirs:
            for worker_dir in self.worker_dirs:
                if os.path.exists(worker_dir):
                    logger.info("[CLUSTER] Removing %s", worker_dir)
                    shutil.rmtree(worker_dir, ignore_errors=True)
            self.worker_dirs = []
            self._initialized = False

    def status(self) -> Dict[str, Any]:
        """Return cluster status summary."""
        worker_status = []
        for i, d in enumerate(self.worker_dirs):
            exists = os.path.isdir(d)
            has_java = os.path.isfile(os.path.join(d, "jre", "bin", "java")) if exists else False
            worker_status.append({
                "id": i,
                "dir": d,
                "exists": exists,
                "has_java": has_java,
            })

        return {
            "num_workers": self.num_workers,
            "workspace_dir": self.workspace_dir,
            "base_env_dir": self.base_env_dir,
            "character_class": self.character_class,
            "use_xvfb": self.use_xvfb,
            "initialized": self._initialized,
            "workers": worker_status,
        }


# ══════════════════════════════════════════════════════════════════
# CLI — Quick health check and diagnostics
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="STS Cluster Manager")
    parser.add_argument(
        "--base-dir", required=True,
        help="Path to extracted base STS environment",
    )
    parser.add_argument(
        "--workspace", default="./workers",
        help="Parent directory for worker dirs",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--character", default="IRONCLAD",
        choices=["IRONCLAD", "SILENT", "DEFECT", "WATCHER"],
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Initialize worker directories",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild (delete existing workers)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print cluster status",
    )

    args = parser.parse_args()

    mgr = ClusterManager(
        base_env_dir=args.base_dir,
        workspace_dir=args.workspace,
        num_workers=args.num_workers,
        character_class=args.character,
    )

    if args.init:
        dirs = mgr.initialize_workers(force_rebuild=args.force)
        print(f"\nInitialized {len(dirs)} workers:")
        for d in dirs:
            print(f"  → {d}")

    if args.status:
        print(json.dumps(mgr.status(), indent=2))
