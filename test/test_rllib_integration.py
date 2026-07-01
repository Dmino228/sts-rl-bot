import os
import sys
import argparse
import json
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process_manager import GameProcessManager
from rllib.env_wrapper import (
    RLLibActionMaskEnv,
    make_sts_rllib_env,
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


class StubStrategicEnv(StubMaskedEnv):
    def __init__(self) -> None:
        super().__init__()
        self._mask = np.zeros(100, dtype=np.int8)
        self._mask[68] = 1
        self._mask[69] = 1
        self.current_state = {
            "type": "decision",
            "decision": "map_select",
            "choices": [
                {"col": 0, "row": 1, "type": "Elite"},
                {"col": 1, "row": 1, "type": "Monster"},
            ],
            "player": {"hp": 25, "max_hp": 80},
        }


class FakeEpisode:
    def __init__(self) -> None:
        self.user_data: dict[str, Any] = {}
        self.custom_metrics: dict[str, float] = {}
        self._info: dict[str, Any] | None = None

    def last_info_for(self):
        return self._info


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


def test_rllib_wrapper_can_hard_control_noncombat_with_heuristic():
    from sts2.heuristics import StS2StrategicHeuristic

    base_env = StubStrategicEnv()
    env = RLLibActionMaskEnv(
        base_env,
        heuristic_policy=StS2StrategicHeuristic(),
        heuristic_mode="hard",
    )

    obs, info = env.reset()

    assert np.flatnonzero(obs["action_mask"]).tolist() == [69]
    assert info["heuristic_action"]["action_id"] == 69

    _, _, _, _, info = env.step(68)

    assert base_env.last_action == 69
    assert info["invalid_action_remapped"] == {"requested": 68, "used": 69}
    assert info["heuristic_action"]["phase"] == "map_select"


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


def test_train_rllib_uses_game_scoped_default_checkpoint_dir(tmp_path, monkeypatch):
    from rllib import train_rllib

    monkeypatch.setattr(train_rllib, "MODELS_DIR", str(tmp_path / "models"))
    args = argparse.Namespace(
        smoke_test=False,
        game_version="2",
        checkpoint_dir="",
    )

    game_key = train_rllib._checkpoint_game_key(args)

    assert game_key == "sts2"
    assert train_rllib._resolve_checkpoint_dir(args, game_key) == os.path.join(
        str(tmp_path / "models"),
        "rllib",
        "sts2",
    )


def test_train_rllib_uses_stage_scoped_checkpoint_dir(tmp_path, monkeypatch):
    from rllib import train_rllib

    monkeypatch.setattr(train_rllib, "MODELS_DIR", str(tmp_path / "models"))
    args = argparse.Namespace(
        smoke_test=False,
        game_version="2",
        checkpoint_dir="",
        training_stage="combat_c0_ironclad_starter_act1",
    )

    game_key = train_rllib._checkpoint_game_key(args)

    assert train_rllib._resolve_checkpoint_dir(args, game_key) == os.path.join(
        str(tmp_path / "models"),
        "rllib",
        "sts2",
        "combat_c0_ironclad_starter_act1",
    )


def test_checkpoint_metadata_payload_records_curriculum_fields(tmp_path):
    from rllib import train_rllib

    args = argparse.Namespace(
        training_stage="combat_c0_ironclad_starter_act1",
        character="Ironclad",
        multi_character=False,
        deck_mode="starter",
        enemy_pool="act1",
        run_notes="Starter combat-only baseline",
        heuristic_mode="hard",
        heuristic_top_k=1,
        workers=8,
        envs_per_worker=1,
        train_batch_size=1024,
        minibatch_size=256,
        num_epochs=4,
        rollout_fragment_length=128,
        sts2_cli_path="Sts2Headless.exe",
        sts2_cli_cwd=r"C:\dev\sts2-cli",
        sts2_cli_args=["--lang", "en"],
        sts2_curriculum_mode="combat",
        sts2_combat_room_type="combat",
        sts2_combat_encounter="SHRINKER_BEETLE_WEAK",
        sts2_recycle_every_episodes=250,
        sts2_recycle_every_steps=0,
        sts2_recycle_rss_mb=768.0,
    )

    payload = train_rllib._checkpoint_metadata_payload(
        args,
        "sts2",
        total_steps=2_000_000,
        checkpoint_path=str(tmp_path / "checkpoint_000001"),
        source_checkpoint="",
    )

    assert payload["schema_version"] == train_rllib.CHECKPOINT_METADATA_SCHEMA
    assert payload["training_stage"] == "combat_c0_ironclad_starter_act1"
    assert payload["character"] == "Ironclad"
    assert payload["deck_mode"] == "starter"
    assert payload["enemy_pool"] == "act1"
    assert payload["total_steps"] == 2_000_000
    assert payload["source_checkpoint"] is None
    assert payload["notes"] == "Starter combat-only baseline"
    assert payload["heuristic_mode"] == "hard"
    assert payload["training"]["workers"] == 8
    assert payload["engine"]["sts2_curriculum_mode"] == "combat"
    assert payload["engine"]["sts2_combat_encounter"] == "SHRINKER_BEETLE_WEAK"
    assert payload["engine"]["sts2_recycle_rss_mb"] == 768.0


def test_write_checkpoint_metadata_places_json_next_to_checkpoint(tmp_path):
    from rllib import train_rllib

    checkpoint_dir = tmp_path / "checkpoint_000001"
    checkpoint_dir.mkdir()
    args = argparse.Namespace(
        training_stage="full_a1_ironclad_topkmask",
        character="Ironclad",
        multi_character=False,
        deck_mode="full_run",
        enemy_pool="act1",
        run_notes="",
        heuristic_mode="mask",
        heuristic_top_k=2,
        workers=1,
        envs_per_worker=1,
        train_batch_size=64,
        minibatch_size=32,
        num_epochs=1,
        rollout_fragment_length=16,
        sts2_cli_path="",
        sts2_cli_cwd="",
        sts2_cli_args=[],
        sts2_curriculum_mode="full_run",
        sts2_combat_room_type="combat",
        sts2_combat_encounter="SHRINKER_BEETLE_WEAK",
        sts2_recycle_every_episodes=0,
        sts2_recycle_every_steps=0,
        sts2_recycle_rss_mb=0.0,
    )

    metadata_path = train_rllib._write_checkpoint_metadata(
        str(checkpoint_dir),
        args,
        "sts2",
        total_steps=10,
        source_checkpoint=str(tmp_path / "source_checkpoint"),
    )

    with open(metadata_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert metadata_path == os.path.join(
        str(checkpoint_dir),
        train_rllib.CHECKPOINT_METADATA_FILENAME,
    )
    assert payload["training_stage"] == "full_a1_ironclad_topkmask"
    assert payload["source_checkpoint"] == os.path.abspath(
        str(tmp_path / "source_checkpoint")
    )


def test_train_rllib_resolves_sts2_timeout_defaults():
    from rllib import train_rllib

    args = argparse.Namespace(process_timeout_s=None, sample_timeout_s=None)

    args.process_timeout_s = train_rllib._resolve_process_timeout(args, "sts2")

    assert args.process_timeout_s == 30.0
    assert train_rllib._resolve_sample_timeout(args, "sts2") == 15.0


def test_train_rllib_resolves_sts2_recycle_defaults():
    from rllib import train_rllib

    args = argparse.Namespace(
        sts2_recycle_every_episodes=None,
        sts2_recycle_rss_mb=None,
    )

    assert train_rllib._resolve_sts2_recycle_every_episodes(args, "sts2") == 250
    assert train_rllib._resolve_sts2_recycle_rss_mb(args, "sts2") == 768.0
    assert train_rllib._resolve_sts2_recycle_every_episodes(args, "sts1") == 0
    assert train_rllib._resolve_sts2_recycle_rss_mb(args, "sts1") == 0.0


def test_train_rllib_allows_disabling_sts2_recycle_defaults():
    from rllib import train_rllib

    args = argparse.Namespace(
        sts2_recycle_every_episodes=0,
        sts2_recycle_rss_mb=0.0,
    )

    assert train_rllib._resolve_sts2_recycle_every_episodes(args, "sts2") == 0
    assert train_rllib._resolve_sts2_recycle_rss_mb(args, "sts2") == 0.0


def test_make_heuristic_policy_only_enables_sts2_non_none_modes():
    from rllib.env_wrapper import _make_heuristic_policy
    from sts2.heuristics import StS2StrategicHeuristic

    assert _make_heuristic_policy({"heuristic_mode": "none"}, "sts2") is None
    assert _make_heuristic_policy({"heuristic_mode": "hard"}, "sts1") is None
    assert isinstance(
        _make_heuristic_policy({"heuristic_mode": "hard"}, "sts2"),
        StS2StrategicHeuristic,
    )


def test_train_rllib_configures_env_runner_fault_tolerance():
    from rllib import train_rllib

    class FakeConfig:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def fault_tolerance(self, **kwargs: Any) -> "FakeConfig":
            self.kwargs = kwargs
            return self

    args = argparse.Namespace(
        disable_env_runner_fault_tolerance=False,
        env_runner_health_timeout_s=7.0,
        env_runner_restore_timeout_s=21.0,
        workers=3,
    )
    config = FakeConfig()

    assert train_rllib._configure_fault_tolerance(config, args) is config
    assert config.kwargs["restart_failed_env_runners"] is True
    assert config.kwargs["ignore_env_runner_failures"] is True
    assert config.kwargs["restart_failed_sub_environments"] is True
    assert config.kwargs["env_runner_health_probe_timeout_s"] == 7.0
    assert config.kwargs["env_runner_restore_timeout_s"] == 21.0
    assert config.kwargs["num_consecutive_env_runner_failures_tolerance"] == 12


def test_train_rllib_configures_progress_callback():
    from rllib import train_rllib
    from rllib.progress_metrics import ProgressMetricsCallback

    class FakeConfig:
        def __init__(self) -> None:
            self.callback_cls: Any = None

        def callbacks(self, callbacks_class: Any) -> "FakeConfig":
            self.callback_cls = callbacks_class
            return self

    config = FakeConfig()

    assert train_rllib._configure_callbacks(config, ProgressMetricsCallback) is config
    assert config.callback_cls is ProgressMetricsCallback


def test_progress_metrics_callback_aggregates_episode_info():
    from rllib.progress_metrics import ProgressMetricsCallback

    callback = ProgressMetricsCallback()
    episode = FakeEpisode()
    callback.on_episode_start(episode=episode)
    episode._info = {
        "progress_metrics": {
            "floor": 16,
            "boss_reached": 1.0,
            "boss_killed": 0.0,
            "act2": 0.0,
        }
    }
    callback.on_episode_step(episode=episode)
    episode._info = {
        "progress_metrics": {
            "floor": 17,
            "boss_reached": 1.0,
            "boss_killed": 1.0,
            "act2": 1.0,
        }
    }
    callback.on_episode_end(episode=episode)

    assert episode.custom_metrics["floor"] == 17.0
    assert episode.custom_metrics["max_floor"] == 17.0
    assert episode.custom_metrics["boss_reached_pct"] == 100.0
    assert episode.custom_metrics["boss_killed_pct"] == 100.0
    assert episode.custom_metrics["act2_pct"] == 100.0


def test_train_rllib_progress_log_metrics_reads_custom_metrics():
    from rllib import train_rllib

    metrics = train_rllib._progress_log_metrics(
        {
            "custom_metrics": {
                "floor_mean": 8.125,
                "max_floor_max": 16,
                "boss_reached_pct_mean": 25,
                "boss_killed_pct_mean": 12.5,
                "act2_pct_mean": 12.5,
            }
        }
    )

    assert metrics == {
        "floor_mean": "8.12",
        "max_floor": "16.00",
        "boss_reached_pct": "25.00",
        "boss_killed_pct": "12.50",
        "act2_pct": "12.50",
    }


def test_result_env_step_delta_prefers_this_iter_metric():
    from rllib import train_rllib

    assert (
        train_rllib._result_env_step_delta(
            {"num_env_steps_sampled_this_iter": 128},
            previous_steps=1000,
            current_steps=2000,
        )
        == 128
    )


def test_make_sts_rllib_env_passes_process_timeout(tmp_path, monkeypatch):
    captured_kwargs: dict[str, Any] = {}

    def fake_env(**kwargs: Any) -> StubMaskedEnv:
        captured_kwargs.update(kwargs)
        return StubMaskedEnv()

    monkeypatch.setattr("rllib.env_wrapper.SlayTheSpireEnv", fake_env)

    env = make_sts_rllib_env(
        {
            "workspace_dir": str(tmp_path),
            "game_version": "2",
            "character_class": "Ironclad",
            "process_timeout": 12.5,
            "sts2_recycle_every_episodes": 123,
            "sts2_recycle_every_steps": 4567,
            "sts2_recycle_rss_mb": 512.5,
            "sts2_curriculum_mode": "combat",
            "sts2_combat_room_type": "elite",
            "sts2_combat_encounter": "SHRINKER_BEETLE_WEAK",
        }
    )

    assert isinstance(env, RLLibActionMaskEnv)
    assert captured_kwargs["process_timeout"] == 12.5
    assert captured_kwargs["sts2_recycle_every_episodes"] == 123
    assert captured_kwargs["sts2_recycle_every_steps"] == 4567
    assert captured_kwargs["sts2_recycle_rss_mb"] == 512.5
    assert captured_kwargs["sts2_curriculum_mode"] == "combat"
    assert captured_kwargs["sts2_combat_room_type"] == "elite"
    assert captured_kwargs["sts2_combat_encounter"] == "SHRINKER_BEETLE_WEAK"


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

