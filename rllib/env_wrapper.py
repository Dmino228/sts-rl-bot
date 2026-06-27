"""Gymnasium wrappers and env registration helpers for RLlib training."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Mapping
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env import SlayTheSpireEnv
from engine_factory import normalize_game_version


RLLIB_ENV_NAME = "sts-rllib-action-mask-v0"
DEFAULT_RLLIB_BASE_PORT = 22340
DEFAULT_STS1_CHARACTERS = ("IRONCLAD", "SILENT", "DEFECT", "WATCHER")
DEFAULT_STS2_CHARACTERS = ("Ironclad", "Silent", "Defect", "Necrobinder", "Regent")


class RLLibActionMaskEnv(gym.Wrapper):
    """Expose SlayTheSpireEnv masks in RLlib's dict-observation format.

    RLlib's legacy ModelV2 action masking path expects observations shaped as
    {"observations": flat_observation, "action_mask": binary_mask}. The wrapped
    base env remains a normal Gymnasium env and stays framework agnostic.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("RLLibActionMaskEnv requires a Discrete action space.")

        self.observation_space = spaces.Dict(
            {
                "observations": env.observation_space,
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(env.action_space.n,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = env.action_space
        self._last_action_mask = np.ones(self.action_space.n, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        mask = self._extract_mask(info)
        return self._wrap_observation(observation, mask), info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        mask_before_step = self._current_base_mask()
        original_action = int(action)
        if not self._is_valid_action(original_action, mask_before_step):
            valid_actions = np.flatnonzero(mask_before_step > 0.0)
            action = int(valid_actions[0]) if len(valid_actions) else 98

        observation, reward, terminated, truncated, info = self.env.step(int(action))
        if original_action != int(action):
            info = dict(info)
            info["invalid_action_remapped"] = {
                "requested": original_action,
                "used": int(action),
            }

        mask = self._extract_mask(info)
        return self._wrap_observation(observation, mask), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Compatibility hook for tooling that still asks the env directly."""
        return self._current_base_mask().astype(np.float32, copy=True)

    def close(self) -> None:
        """Ensure the underlying env (and its Java process) is stopped."""
        try:
            self.env.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """Fallback cleanup when the wrapper is garbage-collected after a crash."""
        try:
            self.env.close()
        except Exception:
            pass

    def _wrap_observation(
        self,
        observation: Any,
        mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float32)
        obs = np.clip(obs, self.env.observation_space.low, self.env.observation_space.high)
        return {
            "observations": obs,
            "action_mask": mask.astype(np.float32, copy=False),
        }

    def _extract_mask(self, info: Mapping[str, Any]) -> np.ndarray:
        raw_mask = info.get("action_mask")
        if raw_mask is None:
            raw_mask = self._current_base_mask()
        mask = np.asarray(raw_mask, dtype=np.float32)
        if mask.shape != (self.action_space.n,):
            raise ValueError(
                f"Expected action mask shape {(self.action_space.n,)}, got {mask.shape}."
            )
        if not np.any(mask > 0.0):
            mask = np.zeros(self.action_space.n, dtype=np.float32)
            mask[98 if self.action_space.n > 98 else 0] = 1.0
        self._last_action_mask = mask
        return mask

    def _current_base_mask(self) -> np.ndarray:
        getter = getattr(self.env, "get_action_mask", None)
        if callable(getter):
            return np.asarray(getter(), dtype=np.float32)
        return self._last_action_mask

    @staticmethod
    def _is_valid_action(action: int, mask: np.ndarray) -> bool:
        return 0 <= action < len(mask) and bool(mask[action] > 0.0)


def register_rllib_env() -> None:
    """Register the STS RLlib env with Ray Tune's env registry."""
    from ray.tune.registry import register_env

    register_env(RLLIB_ENV_NAME, make_sts_rllib_env)


def make_sts_rllib_env(env_config: Mapping[str, Any]) -> RLLibActionMaskEnv:
    """Create one RLlib-compatible STS env from an EnvContext/config dict."""
    worker_id = resolve_worker_id(env_config)
    game_version = _config_value(env_config, "game_version", 1)
    normalized_game = normalize_game_version(game_version)
    workspace_dir = os.path.abspath(
        str(_config_value(env_config, "workspace_dir", "rllib_workers"))
    )
    if normalized_game == "sts1":
        base_env_dir = os.path.abspath(
            str(_config_value(env_config, "base_env_dir", "SlayTheSpire"))
        )
        force_rebuild = bool(_config_value(env_config, "force_rebuild", False))
        worker_dir = prepare_worker_dir(
            base_env_dir=base_env_dir,
            workspace_dir=workspace_dir,
            worker_id=worker_id,
            force_rebuild=force_rebuild,
        )
    else:
        worker_dir = prepare_sts2_worker_dir(
            workspace_dir=workspace_dir,
            worker_id=worker_id,
        )

    env = SlayTheSpireEnv(
        character_class=select_character(worker_id, env_config, normalized_game),
        worker_dir=worker_dir,
        worker_id=worker_id,
        base_port=int(_config_value(env_config, "base_port", DEFAULT_RLLIB_BASE_PORT)),
        use_xvfb=bool(_config_value(env_config, "use_xvfb", False)),
        include_raw_state_in_info=bool(_config_value(env_config, "debug_env_info", False)),
        include_action_mask_in_info=True,
        ram_usage=str(_config_value(env_config, "ram_usage", "default")),
        game_version=game_version,
        sts2_cli_path=str(_config_value(env_config, "sts2_cli_path", "sts2-cli")),
        sts2_cli_args=list(_config_value(env_config, "sts2_cli_args", [])),
        sts2_cli_cwd=_optional_str(_config_value(env_config, "sts2_cli_cwd", None)),
        sts2_ascension=int(_config_value(env_config, "ascension", 0)),
        sts2_lang=str(_config_value(env_config, "sts2_lang", "en")),
    )
    return RLLibActionMaskEnv(env)


def resolve_worker_id(env_config: Mapping[str, Any]) -> int:
    """Return a deterministic unique id for Ray worker/vector env placement."""
    explicit = _config_value(env_config, "worker_id", None)
    if explicit is not None:
        return int(explicit)

    worker_index = int(_config_value(env_config, "worker_index", 0))
    vector_index = int(_config_value(env_config, "vector_index", 0))
    envs_per_runner = int(
        _config_value(
            env_config,
            "num_envs_per_env_runner",
            _config_value(env_config, "num_envs_per_worker", 1),
        )
    )
    return max(worker_index, 0) * max(envs_per_runner, 1) + max(vector_index, 0)


def select_character(
    worker_id: int,
    env_config: Mapping[str, Any],
    game_version: str = "sts1",
) -> str:
    """Select character class by explicit schedule or round-robin default."""
    raw_schedule = _config_value(env_config, "character_schedule", None)
    if raw_schedule is None and bool(_config_value(env_config, "multi_character", False)):
        raw_schedule = (
            DEFAULT_STS2_CHARACTERS
            if normalize_game_version(game_version) == "sts2"
            else DEFAULT_STS1_CHARACTERS
        )

    if raw_schedule:
        schedule = [str(item) for item in raw_schedule]
        if normalize_game_version(game_version) == "sts1":
            schedule = [item.upper() for item in schedule]
        return schedule[worker_id % len(schedule)]

    selected = str(_config_value(env_config, "character_class", "IRONCLAD"))
    if normalize_game_version(game_version) == "sts1":
        return selected.upper()
    return selected


def prepare_worker_dir(
    base_env_dir: str,
    workspace_dir: str,
    worker_id: int,
    force_rebuild: bool = False,
) -> str:
    """Create or refresh an isolated game directory for a Ray env worker."""
    worker_dir = os.path.join(workspace_dir, f"worker_{worker_id}")
    if force_rebuild and os.path.isdir(worker_dir):
        shutil.rmtree(worker_dir, onerror=_remove_readonly)

    if not os.path.isdir(worker_dir):
        if not os.path.isdir(base_env_dir):
            raise FileNotFoundError(
                f"Base STS environment directory not found: {base_env_dir}"
            )
        os.makedirs(workspace_dir, exist_ok=True)
        shutil.copytree(base_env_dir, worker_dir, dirs_exist_ok=False)

    clean_worker_state(worker_dir)
    return worker_dir


def prepare_sts2_worker_dir(workspace_dir: str, worker_id: int) -> str:
    """Create an isolated lightweight workspace for one sts2-cli worker."""
    worker_dir = os.path.join(workspace_dir, f"sts2_worker_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    return worker_dir


def clean_worker_state(worker_dir: str) -> None:
    """Clear volatile per-run state without deleting the game installation."""
    for subdir in ("saves", "runs", "sendToDevs", "logs"):
        path = os.path.join(worker_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=_remove_readonly)
        os.makedirs(path, exist_ok=True)


def _remove_readonly(func: Any, path: str, _excinfo: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _config_value(
    env_config: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    if isinstance(env_config, Mapping) and key in env_config:
        return env_config[key]
    return getattr(env_config, key, default)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None
