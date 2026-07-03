"""Built-in experiment presets and YAML config loading for train_rllib.

Presets are plain dicts keyed by CLI flag names (underscored, matching
``argparse`` dest).  Resolution order:

    preset defaults  <  YAML config file  <  CLI flags

CLI always wins.
"""

from __future__ import annotations

import os
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    # ---- combat: quick smoke test on a single fixed encounter ----
    "combat_smoke_fixed": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "fixed",
        "training_stage": "smoke_combat_fixed",
        "workers": 2,
        "envs_per_worker": 1,
        "timesteps": 50_000,
        "eval_combat_episodes": 20,
        "eval_combat_freq": 5,
        "eval_random_baseline": 20,
        "eval_random_baseline_freq": 0,
        "console_mode": "compact",
        "character": "Ironclad",
        "deck_mode": "starter",
    },
    # ---- combat: debug with verbose logging + debug episodes ----
    "combat_debug_fixed": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "fixed",
        "training_stage": "debug_combat_fixed",
        "workers": 1,
        "envs_per_worker": 1,
        "timesteps": 10_000,
        "eval_combat_episodes": 10,
        "eval_combat_freq": 3,
        "eval_random_baseline": 10,
        "eval_random_baseline_freq": 0,
        "sts2_debug_episodes": 3,
        "console_mode": "verbose",
        "character": "Ironclad",
        "deck_mode": "starter",
    },
    # ---- combat: debug with verbose logging + debug episodes on act1 mixed pool with starter deck ----
    "combat_debug_act1_mixed_starter_deck": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "act1_mixed",
        "training_stage": "debug_combat_act1_mixed_starter_deck",
        "workers": 1,
        "envs_per_worker": 1,
        "timesteps": 10_000,
        "eval_combat_episodes": 10,
        "eval_combat_freq": 3,
        "eval_random_baseline": 10,
        "eval_random_baseline_freq": 0,
        "sts2_debug_episodes": 3,
        "console_mode": "verbose",
        "character": "Ironclad",
        "deck_mode": "starter",
        "enemy_pool": "act1",
    },
    # ---- combat: debug with verbose logging + debug episodes on all acts mixed pool with starter deck ----
    "combat_debug_all_mixed_starter_deck": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "all_mixed",
        "training_stage": "debug_combat_all_mixed_starter_deck",
        "workers": 1,
        "envs_per_worker": 1,
        "timesteps": 10_000,
        "eval_combat_episodes": 10,
        "eval_combat_freq": 3,
        "eval_random_baseline": 10,
        "eval_random_baseline_freq": 0,
        "sts2_debug_episodes": 3,
        "console_mode": "verbose",
        "character": "Ironclad",
        "deck_mode": "starter",
        "enemy_pool": "all",
    },
    # ---- combat: real training on act1 mixed pool (starter deck) ----
    "combat_train_act1_mixed": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "act1_mixed",
        "training_stage": "combat_c1_ironclad_starter_act1_mixed",
        "workers": 8,
        "envs_per_worker": 1,
        "timesteps": 1_000_000,
        "eval_combat_episodes": 0,
        "eval_combat_freq": 10,
        "eval_combat_deterministic": True,
        "eval_random_baseline": 0,
        "eval_random_baseline_freq": 0,
        "checkpoint_freq": 10,
        "console_mode": "compact",
        "character": "Ironclad",
        "deck_mode": "starter",
    },
    # ---- combat: real training on act1 mixed pool with explicit starter deck metadata ----
    "combat_train_act1_mixed_starter_deck": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "act1_mixed",
        "training_stage": "combat_c1_ironclad_starter_act1_mixed",
        "workers": 8,
        "envs_per_worker": 1,
        "timesteps": 1_000_000,
        "eval_combat_episodes": 0,
        "eval_combat_freq": 10,
        "eval_combat_deterministic": True,
        "eval_random_baseline": 0,
        "eval_random_baseline_freq": 0,
        "checkpoint_freq": 10,
        "console_mode": "compact",
        "character": "Ironclad",
        "deck_mode": "starter",
        "enemy_pool": "act1",
    },
    # ---- combat: real training on all acts mixed pool with starter deck ----
    "combat_train_all_mixed_starter_deck": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "all_mixed",
        "training_stage": "combat_c1_ironclad_starter_all_mixed",
        "workers": 8,
        "envs_per_worker": 1,
        "timesteps": 1_000_000,
        "eval_combat_episodes": 0,
        "eval_combat_freq": 10,
        "eval_combat_deterministic": True,
        "eval_random_baseline": 0,
        "eval_random_baseline_freq": 0,
        "checkpoint_freq": 10,
        "console_mode": "compact",
        "character": "Ironclad",
        "deck_mode": "starter",
        "enemy_pool": "all",
    },
    # ---- combat: eval-only on act1 mixed (no training steps) ----
    "combat_eval_act1_mixed": {
        "game_version": "2",
        "sts2_curriculum_mode": "combat",
        "sts2_reward_mode": "combat_sparse",
        "sts2_combat_enemy_pool": "act1_mixed",
        "training_stage": "eval_combat_act1_mixed",
        "workers": 1,
        "envs_per_worker": 1,
        "timesteps": 0,
        "eval_combat_episodes": 1000,
        "eval_combat_freq": 1,
        "eval_combat_deterministic": True,
        "eval_random_baseline": 1000,
        "eval_random_baseline_freq": 0,
        "console_mode": "compact",
        "character": "Ironclad",
        "deck_mode": "starter",
    },
    # ---- full run: Ironclad, no heuristic ----
    "fullrun_ironclad": {
        "game_version": "2",
        "sts2_curriculum_mode": "full_run",
        "sts2_reward_mode": "full_v3_2",
        "training_stage": "fullrun_ironclad",
        "workers": 8,
        "envs_per_worker": 1,
        "timesteps": 10_000_000,
        "heuristic_mode": "none",
        "console_mode": "compact",
        "character": "Ironclad",
    },
    # ---- full run: Ironclad with hard non-combat heuristic ----
    "fullrun_ironclad_heuristic_hard": {
        "game_version": "2",
        "sts2_curriculum_mode": "full_run",
        "sts2_reward_mode": "full_v3_2",
        "training_stage": "fullrun_ironclad_heuristic_hard",
        "workers": 8,
        "envs_per_worker": 1,
        "timesteps": 10_000_000,
        "heuristic_mode": "hard",
        "console_mode": "compact",
        "character": "Ironclad",
    },
}


def list_presets() -> list[str]:
    """Return sorted preset names."""
    return sorted(PRESETS)


def load_preset(name: str) -> dict[str, Any]:
    """Return a copy of a built-in preset dict.

    Raises ``ValueError`` if the preset is not found.
    """
    if name not in PRESETS:
        available = ", ".join(list_presets())
        raise ValueError(
            f"Unknown preset: {name!r}. Available presets: {available}"
        )
    return dict(PRESETS[name])


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a flat-key YAML config file.

    Keys should use underscores matching argparse dest names, e.g.
    ``sts2_reward_mode`` not ``--sts2-reward-mode``.

    Returns an empty dict for missing or empty files.
    """
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items()}


def merge_configs(
    preset: dict[str, Any],
    yaml_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge preset ← yaml ← CLI.  CLI always wins.

    ``cli_overrides`` should only contain keys that were **explicitly set**
    on the command line (not argparse defaults).
    """
    merged = dict(preset)
    merged.update(yaml_config)
    merged.update(cli_overrides)
    return merged
