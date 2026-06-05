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

def test_rllib_wrapper_clips_out_of_bounds_observations():
    """RLLibActionMaskEnv must clip observations that exceed the declared space."""
    base_env = StubMaskedEnv()
    env = RLLibActionMaskEnv(base_env)

    # Inject an out-of-bounds observation into the base env
    original_step = base_env.step

    def step_with_oob(action):
        obs, reward, terminated, truncated, info = original_step(action)
        # Simulate state_encoder producing values > 1.0 (e.g. pile sizes)
        obs[13] = 1.5   # draw_pile / 40 with 60 cards
        obs[17] = -1.3  # negative strength
        obs[107] = 2.0  # monster power
        return obs, reward, terminated, truncated, info

    base_env.step = step_with_oob
    env.reset()
    obs, _, _, _, _ = env.step(66)  # valid action

    assert obs["observations"][13] == 1.0, "Values > 1.0 must be clipped to 1.0"
    assert obs["observations"][17] == -1.0, "Values < -1.0 must be clipped to -1.0"
    assert obs["observations"][107] == 1.0, "All out-of-bounds values must be clipped"
    assert env.observation_space.contains(obs), "Clipped obs must be within declared space"


def test_state_encoder_always_within_bounds():
    """StateEncoder.encode() must always return values within [-1, 1]."""
    from state_encoder import StateEncoder

    encoder = StateEncoder()

    # Simulate an extreme combat state with large values
    extreme_state = {
        "game_state": {
            "screen_type": "COMBAT",
            "floor": 999,
            "gold": 99999,
            "ascension_level": 20,
            "current_hp": 1,
            "max_hp": 1,
            "potions": [{"id": f"potion_{i}"} for i in range(10)],
            "combat_state": {
                "player": {
                    "current_hp": 50,
                    "max_hp": 80,
                    "energy": 99,
                    "block": 999,
                    "powers": [
                        {"id": "Strength", "amount": 999},
                        {"id": "Dexterity", "amount": -50},
                        {"id": "Vulnerable", "amount": 99},
                        {"id": "Weak", "amount": 99},
                        {"id": "Frail", "amount": 99},
                    ],
                },
                "hand": [
                    {"cost": 10, "damage": 999, "block": 999, "type": "ATTACK"}
                    for _ in range(10)
                ],
                "draw_pile": [{}] * 100,
                "discard_pile": [{}] * 100,
                "exhaust_pile": [{}] * 100,
                "monsters": [
                    {
                        "current_hp": 500,
                        "max_hp": 500,
                        "block": 999,
                        "intent": "ATTACK",
                        "move_adjusted_damage": 999,
                        "move_hits": 99,
                        "powers": [
                            {"id": "Strength", "amount": 999},
                            {"id": "Vulnerable", "amount": 99},
                            {"id": "Weak", "amount": 99},
                            {"id": "Ritual", "amount": 99},
                        ],
                    }
                    for _ in range(5)
                ],
            },
        }
    }

    obs = encoder.encode(extreme_state)
    assert np.all(obs >= -1.0), f"Min value {obs.min()} is below -1.0 at index {obs.argmin()}"
    assert np.all(obs <= 1.0), f"Max value {obs.max()} exceeds 1.0 at index {obs.argmax()}"
    assert encoder.observation_space.contains(obs), "Encoded obs must be within declared space"


def test_rllib_smoke_training_one_optimization_step(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")

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
        ],
    )

    try:
        train_rllib.main()
    except Exception as exc:
        # Ray cluster startup can fail on resource-constrained machines
        # (e.g. when SB3 training workers are already consuming CPUs).
        if "timed out during startup" in str(exc) or "raylet" in str(exc).lower():
            pytest.skip(f"Ray cluster failed to start on this machine: {exc}")
        raise
    finally:
        # Always shut down Ray to avoid leaking cluster state between tests
        if ray.is_initialized():
            ray.shutdown()

