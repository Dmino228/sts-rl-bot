"""Argument parsing, config resolution, and the ResolvedConfig container.

This module owns ``parse_args``, all ``_resolve_*`` helpers, and the final
``resolve_config`` entry-point that returns a fully-resolved dict ready
for use by the training loop, preflight, and console.

Resolution order:  preset defaults  <  YAML file  <  CLI flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Import these lazily to avoid pulling in sts2 at import time when the
# module is just being inspected (e.g. tests).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# CLI argument spec
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for ``train_rllib.py``."""
    parser = argparse.ArgumentParser(
        description="Ray RLlib training for STS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )

    # -- New UX flags -------------------------------------------------------
    g_ux = parser.add_argument_group("UX / preset")
    g_ux.add_argument(
        "--preset",
        default="",
        help="Built-in experiment preset (e.g. combat_train_act1_mixed). Use --list-presets to see all.",
    )
    g_ux.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    g_ux.add_argument(
        "--config",
        default="",
        dest="config_file",
        help="Path to a YAML config file. Keys match CLI flag names (underscored).",
    )
    g_ux.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without starting Ray.",
    )
    g_ux.add_argument(
        "--console",
        choices=["compact", "verbose", "quiet"],
        default=None,
        dest="console_mode",
        help="Console output mode: compact (rich live), verbose (full logs), quiet (errors only).",
    )

    # -- Core ---------------------------------------------------------------
    g_core = parser.add_argument_group("core")
    g_core.add_argument("--workers", type=int, default=None, help="Ray rollout workers.")
    g_core.add_argument("--envs-per-worker", type=int, default=None, help="Vector envs per worker.")
    g_core.add_argument("--timesteps", type=int, default=None, help="Additional env steps to train.")
    g_core.add_argument("--base-env-dir", default=None)
    g_core.add_argument("--workspace-dir", default=None)
    g_core.add_argument("--game-version", default=None, choices=["1", "2", "sts1", "sts2"])
    g_core.add_argument("--character", default=None)
    g_core.add_argument("--ascension", type=int, default=None)
    g_core.add_argument("--seed", type=int, default=None)
    g_core.add_argument("--num-gpus", type=float, default=None)

    # -- STS2 engine --------------------------------------------------------
    g_sts2 = parser.add_argument_group("STS2 engine")
    g_sts2.add_argument("--sts2-cli-path", default=None)
    g_sts2.add_argument("--sts2-cli-cwd", default=None)
    g_sts2.add_argument(
        "--sts2-cli-arg",
        action="append",
        default=None,
        dest="sts2_cli_args",
    )
    g_sts2.add_argument("--sts2-lang", default=None)
    g_sts2.add_argument(
        "--sts2-curriculum-mode",
        choices=["full_run", "combat"],
        default=None,
    )
    g_sts2.add_argument(
        "--sts2-reward-mode",
        choices=["full_v3_2", "combat_sparse", "combat_dense"],
        default=None,
    )
    g_sts2.add_argument("--sts2-combat-room-type", default=None)
    g_sts2.add_argument("--sts2-combat-encounter", default=None)
    g_sts2.add_argument("--sts2-combat-enemy-pool", default=None)
    g_sts2.add_argument("--sts2-combat-damage-reward-scale", type=float, default=None)
    g_sts2.add_argument("--sts2-combat-hp-loss-reward-scale", type=float, default=None)
    g_sts2.add_argument("--sts2-combat-action-penalty", type=float, default=None)
    g_sts2.add_argument("--sts2-debug-episodes", type=int, default=None)
    g_sts2.add_argument("--sts2-capture-stderr", action="store_true", default=None)
    g_sts2.add_argument("--sts2-recycle-every-episodes", type=int, default=None)
    g_sts2.add_argument("--sts2-recycle-every-steps", type=int, default=None)
    g_sts2.add_argument("--sts2-recycle-rss-mb", type=float, default=None)

    # -- Multi-character / heuristic ----------------------------------------
    g_heur = parser.add_argument_group("heuristic")
    g_heur.add_argument("--multi-character", action="store_true", default=None)
    g_heur.add_argument(
        "--heuristic-mode",
        choices=["none", "hard", "mask"],
        default=None,
    )
    g_heur.add_argument("--heuristic-top-k", type=int, default=None)

    # -- Training hyperparams -----------------------------------------------
    g_train = parser.add_argument_group("training")
    g_train.add_argument("--train-batch-size", type=int, default=None)
    g_train.add_argument("--minibatch-size", type=int, default=None)
    g_train.add_argument("--num-epochs", type=int, default=None)
    g_train.add_argument("--rollout-fragment-length", type=int, default=None)
    g_train.add_argument("--ram-usage", choices=["low", "default", "safe"], default=None)

    # -- Checkpointing ------------------------------------------------------
    g_ckpt = parser.add_argument_group("checkpointing")
    g_ckpt.add_argument("--checkpoint-freq", type=int, default=None)
    g_ckpt.add_argument("--checkpoint-dir", default=None)
    g_ckpt.add_argument("--training-stage", default=None)
    g_ckpt.add_argument("--deck-mode", default=None)
    g_ckpt.add_argument("--enemy-pool", default=None)
    g_ckpt.add_argument("--run-notes", default=None)
    g_ckpt.add_argument("--resume-from", default=None)
    g_ckpt.add_argument("--no-auto-resume", action="store_true", default=None)
    g_ckpt.add_argument("--init-from-sb3", default=None)

    # -- Evaluation ---------------------------------------------------------
    g_eval = parser.add_argument_group("evaluation")
    g_eval.add_argument("--eval-random-baseline", type=int, default=None)
    g_eval.add_argument("--eval-random-baseline-freq", type=int, default=None)
    g_eval.add_argument("--eval-combat-episodes", type=int, default=None)
    g_eval.add_argument("--eval-combat-freq", type=int, default=None)
    g_eval.add_argument("--eval-combat-deterministic", action="store_true", default=None)
    g_eval.add_argument("--eval-sts2-recycle-every-episodes", type=int, default=None)

    # -- Timeouts / fault tolerance -----------------------------------------
    g_ft = parser.add_argument_group("timeouts / fault tolerance")
    g_ft.add_argument("--process-timeout-s", type=float, default=None)
    g_ft.add_argument("--sample-timeout-s", type=float, default=None)
    g_ft.add_argument("--train-heartbeat-s", type=float, default=None)
    g_ft.add_argument("--slow-iteration-s", type=float, default=None)
    g_ft.add_argument("--cpus-per-worker", type=float, default=None)
    g_ft.add_argument("--disable-env-runner-fault-tolerance", action="store_true", default=None)
    g_ft.add_argument("--env-runner-health-timeout-s", type=float, default=None)
    g_ft.add_argument("--env-runner-restore-timeout-s", type=float, default=None)

    # -- Misc ---------------------------------------------------------------
    g_misc = parser.add_argument_group("misc")
    g_misc.add_argument("--base-port", type=int, default=None)
    g_misc.add_argument("--use-xvfb", action="store_true", default=None)
    g_misc.add_argument("--force-rebuild", action="store_true", default=None)
    g_misc.add_argument("--debug-env-info", action="store_true", default=None)
    g_misc.add_argument("--smoke-test", action="store_true", default=None)
    g_misc.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge preset + YAML + CLI into a fully-resolved config dict.

    Resolved internal keys (prefixed with ``_``) are added for downstream
    use: ``_game_key``, ``_checkpoint_dir``, ``_training_stage``, etc.
    """
    from rllib.presets import list_presets, load_preset, load_yaml_config, merge_configs

    # Handle --list-presets early exit
    if getattr(args, "list_presets", False):
        _print_presets_and_exit()

    # Load preset defaults
    preset_name = str(getattr(args, "preset", "") or "").strip()
    preset: dict[str, Any] = {}
    if preset_name:
        preset = load_preset(preset_name)

    # Load YAML config
    yaml_config: dict[str, Any] = {}
    config_file = str(getattr(args, "config_file", "") or "").strip()
    if config_file:
        yaml_config = load_yaml_config(config_file)

    # Collect CLI overrides (only explicitly-set values, not None defaults)
    cli_overrides = _collect_cli_overrides(args)

    # Merge: preset < yaml < cli
    merged = merge_configs(preset, yaml_config, cli_overrides)

    # Apply global defaults for anything still missing
    config = _apply_defaults(merged)

    # Resolve derived values
    game_key = _resolve_game_key(config)
    config["_game_key"] = game_key
    config["_preset_name"] = preset_name or "none"
    config["process_timeout_s"] = _resolve_process_timeout(config, game_key)
    config["sample_timeout_s"] = _resolve_sample_timeout(config, game_key)
    config["sts2_recycle_every_episodes"] = _resolve_sts2_recycle_every_episodes(
        config, game_key
    )
    config["sts2_recycle_every_steps"] = max(0, int(config.get("sts2_recycle_every_steps", 0) or 0))
    config["sts2_recycle_rss_mb"] = _resolve_sts2_recycle_rss_mb(config, game_key)
    config["checkpoint_freq"] = _resolve_checkpoint_freq(config, game_key)
    config["eval_combat_freq"] = _resolve_eval_combat_freq(config, game_key)
    config["eval_random_baseline_freq"] = max(
        0, int(config.get("eval_random_baseline_freq", 0) or 0)
    )
    config["eval_sts2_recycle_every_episodes"] = _resolve_eval_sts2_recycle_every_episodes(
        config, game_key
    )
    config["_checkpoint_dir"] = _resolve_checkpoint_dir(config, game_key)
    config["_training_stage"] = _resolve_training_stage(config, game_key)

    return config


def handle_dry_run(config: dict[str, Any]) -> None:
    """If ``--dry-run`` was set, print resolved config and exit."""
    if not config.get("dry_run", False):
        return

    import yaml

    # Filter out internal keys for display
    display = {k: v for k, v in sorted(config.items()) if not k.startswith("_")}
    internal = {k: v for k, v in sorted(config.items()) if k.startswith("_")}
    print("=" * 60)
    print("  RESOLVED CONFIGURATION (--dry-run)")
    print("=" * 60)
    print(yaml.dump(display, default_flow_style=False, sort_keys=True))
    print("--- Internal resolved values ---")
    print(yaml.dump(internal, default_flow_style=False, sort_keys=True))
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Helpers for building env_config dict (used by train_rllib.py)
# ---------------------------------------------------------------------------

def build_env_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the env_config dict passed to Ray from the resolved config."""
    from rllib.env_wrapper import DEFAULT_RLLIB_BASE_PORT

    return {
        "base_env_dir": config.get("base_env_dir", os.path.join(_PROJECT_ROOT, "SlayTheSpire")),
        "workspace_dir": config.get("workspace_dir", os.path.join(_PROJECT_ROOT, "rllib_workers")),
        "game_version": config.get("game_version", "1"),
        "character_class": config.get("character", "IRONCLAD"),
        "multi_character": bool(config.get("multi_character", False)),
        "heuristic_mode": config.get("heuristic_mode", "none"),
        "heuristic_top_k": int(config.get("heuristic_top_k", 1) or 1),
        "ram_usage": config.get("ram_usage", "default"),
        "base_port": int(config.get("base_port", DEFAULT_RLLIB_BASE_PORT) or DEFAULT_RLLIB_BASE_PORT),
        "use_xvfb": bool(config.get("use_xvfb", False)),
        "force_rebuild": bool(config.get("force_rebuild", False)),
        "debug_env_info": bool(config.get("debug_env_info", False)),
        "num_envs_per_env_runner": int(config.get("envs_per_worker", 1) or 1),
        "sts2_cli_path": config.get("sts2_cli_path", "sts2-cli"),
        "sts2_cli_args": list(config.get("sts2_cli_args") or []),
        "sts2_cli_cwd": config.get("sts2_cli_cwd", ""),
        "sts2_capture_stderr": bool(config.get("sts2_capture_stderr", False)),
        "sts2_recycle_every_episodes": int(config.get("sts2_recycle_every_episodes", 0) or 0),
        "sts2_recycle_every_steps": int(config.get("sts2_recycle_every_steps", 0) or 0),
        "sts2_recycle_rss_mb": float(config.get("sts2_recycle_rss_mb", 0.0) or 0.0),
        "sts2_curriculum_mode": config.get("sts2_curriculum_mode", "full_run"),
        "sts2_reward_mode": config.get("sts2_reward_mode", "full_v3_2"),
        "sts2_combat_room_type": config.get("sts2_combat_room_type", "combat"),
        "sts2_combat_encounter": config.get("sts2_combat_encounter", "SHRINKER_BEETLE_WEAK"),
        "sts2_combat_enemy_pool": config.get("sts2_combat_enemy_pool", "fixed"),
        "sts2_combat_damage_reward_scale": float(
            config.get("sts2_combat_damage_reward_scale", 0.01) or 0.01
        ),
        "sts2_combat_hp_loss_reward_scale": float(
            config.get("sts2_combat_hp_loss_reward_scale", 0.01) or 0.01
        ),
        "sts2_combat_action_penalty": float(
            config.get("sts2_combat_action_penalty", 0.001) or 0.001
        ),
        "sts2_debug_episodes": int(config.get("sts2_debug_episodes", 0) or 0),
        "process_timeout": float(config.get("process_timeout_s", 120.0) or 120.0),
        "ascension": int(config.get("ascension", 0) or 0),
        "sts2_lang": config.get("sts2_lang", "en"),
        "sts2_seed": config.get("seed"),
        "eval_sts2_recycle_every_episodes": int(
            config.get("eval_sts2_recycle_every_episodes", 0) or 0
        ),
    }


# ---------------------------------------------------------------------------
# CLI override collection
# ---------------------------------------------------------------------------

def _collect_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return only args explicitly set by the user (not None defaults).

    For ``store_true`` actions, we include the key only when the value is
    True, so that a preset's ``False`` is not overridden by the argparse
    default ``None``.
    """
    overrides: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in ("preset", "config_file", "list_presets"):
            continue
        if value is None:
            continue
        overrides[key] = value
    return overrides


# ---------------------------------------------------------------------------
# Defaults for values not set by any source
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "game_version": "1",
    "workers": 2,
    "envs_per_worker": 1,
    "timesteps": 1_000_000,
    "base_env_dir": os.path.join(_PROJECT_ROOT, "SlayTheSpire"),
    "workspace_dir": os.path.join(_PROJECT_ROOT, "rllib_workers"),
    "character": "IRONCLAD",
    "ascension": 0,
    "sts2_cli_path": "sts2-cli",
    "sts2_cli_cwd": "",
    "sts2_cli_args": [],
    "sts2_lang": "en",
    "sts2_curriculum_mode": "full_run",
    "sts2_reward_mode": "full_v3_2",
    "sts2_combat_room_type": "combat",
    "sts2_combat_encounter": "SHRINKER_BEETLE_WEAK",
    "sts2_combat_enemy_pool": "fixed",
    "sts2_combat_damage_reward_scale": 0.01,
    "sts2_combat_hp_loss_reward_scale": 0.01,
    "sts2_combat_action_penalty": 0.001,
    "sts2_debug_episodes": 0,
    "sts2_capture_stderr": False,
    "sts2_recycle_every_episodes": None,
    "sts2_recycle_every_steps": 0,
    "sts2_recycle_rss_mb": None,
    "multi_character": False,
    "heuristic_mode": "none",
    "heuristic_top_k": 1,
    "train_batch_size": 1024,
    "minibatch_size": 256,
    "num_epochs": 4,
    "rollout_fragment_length": 128,
    "ram_usage": "default",
    "checkpoint_freq": None,
    "checkpoint_dir": "",
    "training_stage": "",
    "deck_mode": "",
    "enemy_pool": "",
    "run_notes": "",
    "resume_from": "",
    "no_auto_resume": False,
    "init_from_sb3": "",
    "eval_random_baseline": 0,
    "eval_random_baseline_freq": 0,
    "eval_combat_episodes": 0,
    "eval_combat_freq": None,
    "eval_combat_deterministic": False,
    "eval_sts2_recycle_every_episodes": None,
    "process_timeout_s": None,
    "sample_timeout_s": None,
    "train_heartbeat_s": 30.0,
    "slow_iteration_s": 60.0,
    "cpus_per_worker": 1.0,
    "disable_env_runner_fault_tolerance": False,
    "env_runner_health_timeout_s": 10.0,
    "env_runner_restore_timeout_s": 60.0,
    "base_port": None,
    "use_xvfb": False,
    "force_rebuild": False,
    "debug_env_info": False,
    "num_gpus": 0.0,
    "seed": None,
    "smoke_test": False,
    "log_level": "INFO",
    "console_mode": "compact",
    "dry_run": False,
}


def _apply_defaults(merged: dict[str, Any]) -> dict[str, Any]:
    """Fill in any keys that are still missing with global defaults."""
    result = dict(_DEFAULTS)
    result.update({k: v for k, v in merged.items() if v is not None})
    # Carry over explicitly-set None values from CLI for seed, etc.
    for k, v in merged.items():
        if k not in result:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Derived-value resolution (moved from train_rllib.py)
# ---------------------------------------------------------------------------

def _resolve_game_key(config: dict[str, Any]) -> str:
    if config.get("smoke_test", False):
        return "smoke"
    from engine_factory import normalize_game_version

    return normalize_game_version(config.get("game_version", "1"))


def _resolve_process_timeout(config: dict[str, Any], game_key: str) -> float:
    val = config.get("process_timeout_s")
    if val is not None:
        return float(val)
    if game_key == "sts2":
        return 30.0
    return 120.0


def _resolve_sample_timeout(config: dict[str, Any], game_key: str) -> float:
    val = config.get("sample_timeout_s")
    if val is not None:
        return float(val)
    if game_key == "sts2":
        return 15.0
    if game_key == "smoke":
        return 60.0
    return 600.0


def _resolve_sts2_recycle_every_episodes(config: dict[str, Any], game_key: str) -> int:
    val = config.get("sts2_recycle_every_episodes")
    if val is not None:
        return max(0, int(val))
    if game_key == "sts2":
        return 250
    return 0


def _resolve_sts2_recycle_rss_mb(config: dict[str, Any], game_key: str) -> float:
    val = config.get("sts2_recycle_rss_mb")
    if val is not None:
        return max(0.0, float(val))
    if game_key == "sts2":
        return 768.0
    return 0.0


def _resolve_checkpoint_freq(config: dict[str, Any], game_key: str) -> int:
    val = config.get("checkpoint_freq")
    if val is not None:
        return max(0, int(val))
    if _is_sts2_combat(config, game_key):
        return 10
    return 1


def _resolve_eval_combat_freq(config: dict[str, Any], game_key: str) -> int:
    if int(config.get("eval_combat_episodes", 0) or 0) <= 0:
        return 0
    val = config.get("eval_combat_freq")
    if val is not None:
        return max(0, int(val))
    if _is_sts2_combat(config, game_key):
        return 10
    return 1


def _resolve_eval_sts2_recycle_every_episodes(config: dict[str, Any], game_key: str) -> int:
    if (
        int(config.get("eval_combat_episodes", 0) or 0) <= 0
        and int(config.get("eval_random_baseline", 0) or 0) <= 0
    ):
        return 0
    val = config.get("eval_sts2_recycle_every_episodes")
    if val is not None:
        return max(0, int(val))
    if _is_sts2_combat(config, game_key):
        return 1000
    return 0


def _resolve_checkpoint_dir(config: dict[str, Any], game_key: str) -> str:
    explicit = config.get("checkpoint_dir", "")
    if explicit:
        return os.path.abspath(str(explicit))
    stage = _safe_path_component(config.get("training_stage", ""))
    if stage:
        return os.path.join(_MODELS_DIR, "rllib", game_key, stage)
    return os.path.join(_MODELS_DIR, "rllib", game_key)


def _resolve_training_stage(config: dict[str, Any], game_key: str) -> str:
    stage = str(config.get("training_stage", "") or "").strip()
    if stage:
        return stage
    if game_key == "smoke":
        return "smoke"
    return "full_run"


def _is_sts2_combat(config: dict[str, Any], game_key: str) -> bool:
    return (
        game_key == "sts2"
        and str(config.get("sts2_curriculum_mode", "")).strip().lower() == "combat"
    )


def _safe_path_component(value: Any) -> str:
    text = str(value or "").strip().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in text).strip("._")


def _print_presets_and_exit() -> None:
    from rllib.presets import PRESETS

    print("\nAvailable presets:\n")
    for name in sorted(PRESETS):
        preset = PRESETS[name]
        curriculum = preset.get("sts2_curriculum_mode", "full_run")
        pool = preset.get("sts2_combat_enemy_pool", "n/a")
        workers = preset.get("workers", "?")
        steps = preset.get("timesteps", "?")
        print(
            f"  {name:<40s}  mode={curriculum:<10s}  pool={pool:<16s}  "
            f"workers={workers}  steps={steps:>12,}"
            if isinstance(steps, int)
            else f"  {name:<40s}  mode={curriculum:<10s}  pool={pool:<16s}  "
            f"workers={workers}  steps={steps}"
        )
    print()
    raise SystemExit(0)


def _epilog() -> str:
    return (
        "Examples:\n"
        "  python rllib/train_rllib.py --preset combat_smoke_fixed --sts2-cli-path <path>\n"
        "  python rllib/train_rllib.py --preset combat_train_act1_mixed --sts2-cli-path <path>\n"
        "  python rllib/train_rllib.py --preset combat_train_act1_mixed --workers 4  # override\n"
        "  python rllib/train_rllib.py --config my_experiment.yaml --sts2-cli-path <path>\n"
        "  python rllib/train_rllib.py --list-presets\n"
        "  python rllib/train_rllib.py --preset combat_smoke_fixed --dry-run\n"
    )
