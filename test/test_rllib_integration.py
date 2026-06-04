import os
import sys
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process_manager import GameProcessManager
from rllib.env_wrapper import (
    RLLibActionMaskEnv,
    resolve_worker_id,
    select_character,
)


class StubMaskedEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(205,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(100)
        self.last_action: Optional[int] = None
        self._mask = np.zeros(100, dtype=np.int8)
        self._mask[66] = 1

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        return np.zeros(205, dtype=np.float32), {"action_mask": self._mask.copy()}

    def step(self, action: int):
        self.last_action = int(action)
        return (
            np.ones(205, dtype=np.float32),
            1.0,
            False,
            False,
            {"action_mask": self._mask.copy()},
        )

    def get_action_mask(self) -> np.ndarray:
        return self._mask.copy()


def test_rllib_wrapper_exposes_dict_observation_with_action_mask():
    env = RLLibActionMaskEnv(StubMaskedEnv())
    obs, info = env.reset()

    assert set(obs.keys()) == {"observations", "action_mask"}
    assert obs["observations"].shape == (205,)
    assert obs["action_mask"].shape == (100,)
    assert obs["action_mask"].dtype == np.float32
    assert info["action_mask"][66] == 1


def test_rllib_wrapper_remaps_invalid_action_to_valid_fallback():
    base_env = StubMaskedEnv()
    env = RLLibActionMaskEnv(base_env)
    env.reset()

    _, _, _, _, info = env.step(0)

    assert base_env.last_action == 66
    assert info["invalid_action_remapped"] == {"requested": 0, "used": 66}


def test_resolve_worker_id_uses_ray_worker_and_vector_indices():
    env_config = {
        "worker_index": 3,
        "vector_index": 2,
        "num_envs_per_env_runner": 4,
    }

    assert resolve_worker_id(env_config) == 14


def test_select_character_round_robins_multi_character_schedule():
    env_config = {"multi_character": True}

    assert select_character(0, env_config) == "IRONCLAD"
    assert select_character(1, env_config) == "SILENT"
    assert select_character(4, env_config) == "IRONCLAD"


def test_process_manager_explicit_worker_id_overrides_directory_suffix():
    manager = GameProcessManager(worker_id=42, base_port=22340)

    assert manager._resolve_worker_id(r"C:\tmp\worker_7") == 42


def test_train_rllib_module_imports_without_ray_model_dependency():
    import importlib

    module = importlib.import_module("rllib.train_rllib")
    assert hasattr(module, "parse_args")


def test_rllib_smoke_training_one_optimization_step(tmp_path, monkeypatch):
    pytest.importorskip("ray")

    from rllib import train_rllib

    monkeypatch.setattr(train_rllib, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(train_rllib, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rllib.py",
            "--smoke-test",
            "--workers",
            "0",
            "--timesteps",
            "32",
            "--train-batch-size",
            "32",
            "--minibatch-size",
            "16",
            "--rollout-fragment-length",
            "16",
            "--checkpoint-freq",
            "0",
            "--local-mode",
        ],
    )

    train_rllib.main()
