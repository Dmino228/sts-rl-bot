"""Tests for the preset and config resolution system."""

from __future__ import annotations

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rllib.presets import (
    PRESETS,
    list_presets,
    load_preset,
    merge_configs,
)
from rllib.progress_metrics import classify_encounter
from rllib.config import resolve_config, parse_args


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------


class TestPresets:
    def test_list_presets_returns_sorted(self) -> None:
        names = list_presets()
        assert names == sorted(names)
        assert len(names) >= 6

    def test_load_preset_known(self) -> None:
        for name in list_presets():
            preset = load_preset(name)
            assert isinstance(preset, dict)
            assert "game_version" in preset

    def test_load_preset_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent_preset")

    def test_load_preset_returns_copy(self) -> None:
        a = load_preset("combat_smoke_fixed")
        b = load_preset("combat_smoke_fixed")
        a["workers"] = 999
        assert b["workers"] != 999

    def test_combat_presets_use_combat_mode(self) -> None:
        for name in list_presets():
            if name.startswith("combat"):
                preset = load_preset(name)
                assert preset["sts2_curriculum_mode"] == "combat"
                assert preset["sts2_reward_mode"] in ("combat_sparse", "combat_dense")

    def test_fullrun_presets_use_full_run_mode(self) -> None:
        for name in list_presets():
            if name.startswith("fullrun"):
                preset = load_preset(name)
                assert preset["sts2_curriculum_mode"] == "full_run"

    def test_combat_train_act1_mixed_random_deck_preset_properties(self) -> None:
        preset = load_preset("combat_train_act1_mixed_random_deck")
        assert preset["sts2_combat_enemy_pool"] == "act1_mixed"
        assert preset["deck_mode"] == "random_controlled"
        assert preset["enemy_pool"] == "act1"

    def test_combat_train_all_mixed_random_deck_preset_properties(self) -> None:
        preset = load_preset("combat_train_all_mixed_random_deck")
        assert preset["sts2_combat_enemy_pool"] == "all_mixed"
        assert preset["deck_mode"] == "random_controlled"
        assert preset["enemy_pool"] == "all"




# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_cli_overrides_preset(self) -> None:
        merged = merge_configs(
            {"workers": 8, "timesteps": 1_000_000},
            {},
            {"workers": 4},
        )
        assert merged["workers"] == 4
        assert merged["timesteps"] == 1_000_000

    def test_yaml_overrides_preset(self) -> None:
        merged = merge_configs(
            {"workers": 8},
            {"workers": 6},
            {},
        )
        assert merged["workers"] == 6

    def test_cli_overrides_yaml(self) -> None:
        merged = merge_configs(
            {"workers": 8},
            {"workers": 6},
            {"workers": 2},
        )
        assert merged["workers"] == 2

    def test_empty_merge(self) -> None:
        merged = merge_configs({}, {}, {})
        assert merged == {}


# ---------------------------------------------------------------------------
# Classify encounter tests
# ---------------------------------------------------------------------------


class TestClassifyEncounter:
    def test_weak(self) -> None:
        assert classify_encounter("SHRINKER_BEETLE_WEAK") == "weak"
        assert classify_encounter("TOADPOLES_WEAK") == "weak"

    def test_normal(self) -> None:
        assert classify_encounter("FLYCONID_NORMAL") == "normal"
        assert classify_encounter("CHOMPERS_NORMAL") == "normal"

    def test_elite(self) -> None:
        assert classify_encounter("BYRDONIS_ELITE") == "elite"
        assert classify_encounter("ENTOMANCER_ELITE") == "elite"

    def test_boss(self) -> None:
        assert classify_encounter("CEREMONIAL_BEAST_BOSS") == "boss"
        assert classify_encounter("THE_KIN_BOSS") == "boss"

    def test_other(self) -> None:
        assert classify_encounter("UNKNOWN_THING") == "other"
        assert classify_encounter("") == "other"

    def test_case_insensitive(self) -> None:
        assert classify_encounter("shrinker_beetle_weak") == "weak"
        assert classify_encounter("Byrdonis_Elite") == "elite"


# ---------------------------------------------------------------------------
# Config resolution tests
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_preset_resolves_game_key(self) -> None:
        args = parse_args(["--preset", "combat_smoke_fixed", "--dry-run"])
        config = resolve_config(args)
        assert config["_game_key"] == "sts2"
        assert config["sts2_curriculum_mode"] == "combat"

    def test_cli_override_workers(self) -> None:
        args = parse_args(["--preset", "combat_smoke_fixed", "--workers", "4", "--dry-run"])
        config = resolve_config(args)
        assert config["workers"] == 4

    def test_no_preset_uses_defaults(self) -> None:
        args = parse_args(["--dry-run"])
        config = resolve_config(args)
        assert config["_game_key"] in ("sts1", "smoke")
        assert config["workers"] == 2  # default

    def test_training_stage_resolves(self) -> None:
        args = parse_args([
            "--preset", "combat_train_act1_mixed",
            "--training-stage", "my_custom_stage",
            "--dry-run",
        ])
        config = resolve_config(args)
        assert config["_training_stage"] == "my_custom_stage"
        assert "my_custom_stage" in config["_checkpoint_dir"]
