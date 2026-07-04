"""Preflight validation checks run before ray.init().

All checks print warnings or raise SystemExit before any Ray overhead.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any


def run_preflight(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Run all preflight checks.  Raises SystemExit on fatal errors."""
    game_key = str(config.get("_game_key", ""))

    if game_key == "sts2":
        _validate_sts2_executable(config, logger)
        _validate_enemy_pool_non_empty(config, logger)
        _warn_combat_mode_with_full_reward(config, logger)
        _validate_sts2_deck_mode(config, logger)
        _warn_expensive_eval(config, logger)

    _warn_worker_count(config, game_key, logger)
    _print_experiment_summary(config, game_key, logger)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _validate_sts2_executable(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Validate the STS2 executable can be found."""
    cli_path = str(config.get("sts2_cli_path", "") or "")
    cli_cwd = str(config.get("sts2_cli_cwd", "") or "")
    resolved = _resolve_executable_path(cli_path, cli_cwd)
    if resolved:
        logger.info("STS2 executable resolved: %s", resolved)
        return

    hint = (
        "StS2 executable not found before starting Ray workers. "
        f"--sts2-cli-path={cli_path!r} cwd={cli_cwd!r}. "
        "Use either an absolute Sts2Headless.exe path, or run through dotnet, e.g. "
        "--sts2-cli-path dotnet --sts2-cli-cwd C:\\dev\\sts2-cli "
        "--sts2-cli-arg=run --sts2-cli-arg=--no-build --sts2-cli-arg=--project "
        "--sts2-cli-arg=C:\\dev\\sts2-cli\\src\\Sts2Headless\\Sts2Headless.csproj"
    )
    raise SystemExit(hint)


def _validate_enemy_pool_non_empty(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Check that the combat enemy pool actually has encounters."""
    curriculum_mode = str(config.get("sts2_curriculum_mode", "full_run"))
    if curriculum_mode != "combat":
        return

    pool = str(config.get("sts2_combat_enemy_pool", "fixed"))
    try:
        from sts2.encounters import combat_pool_ids

        ids = combat_pool_ids(
            pool,
            fixed_encounter=str(config.get("sts2_combat_encounter", "")),
        )
        if not ids:
            logger.warning("Combat enemy pool %r resolved to zero encounters!", pool)
        else:
            logger.info("Combat enemy pool %r: %d encounters", pool, len(ids))
    except Exception as exc:
        logger.warning("Could not validate enemy pool %r: %s", pool, exc)


def _warn_combat_mode_with_full_reward(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Warn if combat curriculum uses full_v3_2 reward (likely misconfigured)."""
    curriculum_mode = str(config.get("sts2_curriculum_mode", "full_run"))
    reward_mode = str(config.get("sts2_reward_mode", "full_v3_2"))
    if curriculum_mode == "combat" and reward_mode == "full_v3_2":
        logger.warning(
            "Combat curriculum mode is using full_v3_2 reward. "
            "This likely includes irrelevant floor/relic/card rewards. "
            "Consider --sts2-reward-mode combat_sparse or combat_dense."
        )


def _validate_sts2_deck_mode(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Validate deck-mode configuration and probe non-starter Headless support."""
    from sts2.deck_generator import normalize_deck_mode

    preset_name = str(config.get("_preset_name", "") or "")
    deck_mode = str(config.get("deck_mode", "") or "").strip().lower()
    try:
        deck_mode = normalize_deck_mode(deck_mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if "random_deck" in preset_name and deck_mode == "starter":
        raise SystemExit(
            f"Preset {preset_name!r} contains 'random_deck' but resolves to "
            "deck_mode=starter. Rename the preset or set a real random deck mode."
        )

    logger.info(
        "STS2 deck mode %r: duplicate_cap=%s allow_problematic_cards=%s",
        deck_mode,
        config.get("sts2_deck_duplicate_cap", 2),
        bool(config.get("sts2_deck_allow_problematic_cards", False)),
    )

    curriculum_mode = str(config.get("sts2_curriculum_mode", "full_run")).strip().lower()
    if curriculum_mode != "combat" or deck_mode == "starter":
        return
    _probe_sts2_deck_support(config, logger)


def _probe_sts2_deck_support(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Start a tiny Headless process and verify set_player can replace/upgrade deck."""
    from sts2.process_manager import StS2CliProcessManager

    manager = StS2CliProcessManager(
        timeout=min(20.0, float(config.get("process_timeout_s", 30.0) or 30.0)),
        cli_path=str(config.get("sts2_cli_path", "sts2-cli") or "sts2-cli"),
        cli_args=list(config.get("sts2_cli_args") or []),
        cli_cwd=str(config.get("sts2_cli_cwd", "") or "") or None,
        capture_stderr=bool(config.get("sts2_capture_stderr", False)),
    )
    try:
        manager.launch_game()
        manager.signal_ready()
        manager.send_command(
            {
                "cmd": "start_run",
                "character": "Ironclad",
                "ascension": 0,
                "seed": "__deck_probe__",
                "lang": str(config.get("sts2_lang", "en") or "en"),
            }
        )
        state = manager.read_state()
        if state.get("type") == "error":
            raise RuntimeError(state.get("message", state))
        manager.send_command(
            {
                "cmd": "set_player",
                "deck": [
                    {"id": "CARD.BASH", "upgraded": True},
                    {"id": "CARD.POMMEL_STRIKE", "upgraded": False},
                    {"id": "CARD.DEFEND_IRONCLAD", "upgraded": False},
                ],
                "hp": 68,
                "max_hp": 80,
                "relics": ["RELIC.BURNING_BLOOD"],
                "potions": [],
            }
        )
        state = manager.read_state()
        if state.get("type") == "error":
            raise RuntimeError(state.get("message", state))
        player = state.get("player") if isinstance(state, dict) else None
        deck = player.get("deck") if isinstance(player, dict) else None
        if not isinstance(deck, list) or len(deck) != 3:
            raise RuntimeError(f"set_player did not return the expected deck: {deck!r}")
        first = deck[0] if isinstance(deck[0], dict) else {}
        if first.get("id") != "CARD.BASH" or first.get("upgraded") is not True:
            raise RuntimeError(
                "set_player deck probe did not preserve CARD.BASH upgrade: "
                f"{first!r}"
            )
        logger.info("STS2 deck replacement probe passed.")
    except Exception as exc:
        raise SystemExit(
            "StS2 Headless cannot apply generated deck specs. "
            "Rebuild/update C:\\dev\\sts2-cli so set_player accepts deck entries "
            f"with id/upgraded before using random deck modes. Probe error: {exc}"
        ) from exc
    finally:
        try:
            manager.stop()
        except Exception:
            pass


def _warn_expensive_eval(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Warn if eval is large and runs every iteration."""
    eval_episodes = int(config.get("eval_combat_episodes", 0) or 0)
    eval_freq = int(config.get("eval_combat_freq", 0) or 0)
    random_episodes = int(config.get("eval_random_baseline", 0) or 0)
    random_freq = int(config.get("eval_random_baseline_freq", 0) or 0)

    if eval_episodes >= 200 and eval_freq == 1:
        logger.warning(
            "Large eval (%d episodes) runs every iteration (eval_combat_freq=1). "
            "This will significantly slow training. Consider --eval-combat-freq 5 or 10.",
            eval_episodes,
        )
    if random_episodes >= 200 and random_freq >= 1 and random_freq <= 2:
        logger.warning(
            "Large random baseline (%d episodes) reruns frequently (freq=%d). "
            "Random baseline usually only needs to run at startup (freq=0).",
            random_episodes,
            random_freq,
        )


def _warn_worker_count(
    config: dict[str, Any],
    game_key: str,
    logger: logging.Logger,
) -> None:
    """Warn if worker count is at or above logical CPU count for StS2."""
    if game_key != "sts2":
        return
    logical_cpus = os.cpu_count() or 1
    workers = max(int(config.get("workers", 0) or 0), 0)
    envs_per_worker = max(int(config.get("envs_per_worker", 1) or 1), 1)
    env_count = workers * envs_per_worker
    if env_count >= logical_cpus:
        logger.warning(
            "StS2 worker count is at or above logical CPU count: %d envs on %d CPUs. "
            "Each env also owns an external C# process, so 16 workers on an 8C/16T "
            "CPU can expose scheduler stalls. Treat 8-12 workers as the first "
            "performance sweep before trying 16 again.",
            env_count,
            logical_cpus,
        )


def _print_experiment_summary(
    config: dict[str, Any],
    game_key: str,
    logger: logging.Logger,
) -> None:
    """Print a clean summary of the resolved experiment configuration."""
    lines = [
        "=" * 60,
        "  EXPERIMENT SUMMARY",
        "=" * 60,
        f"  Game version:        {game_key}",
        f"  Training stage:      {config.get('_training_stage', 'unset')}",
        f"  Character:           {config.get('character', 'unset')}",
        f"  Preset:              {config.get('_preset_name', 'none')}",
    ]

    if game_key == "sts2":
        lines.extend([
            f"  Curriculum mode:     {config.get('sts2_curriculum_mode', 'full_run')}",
            f"  Reward mode:         {config.get('sts2_reward_mode', 'full_v3_2')}",
            f"  Enemy pool:          {config.get('sts2_combat_enemy_pool', 'fixed')}",
            f"  Deck mode:           {config.get('deck_mode', 'unspecified') or 'unspecified'}",
            f"  Heuristic mode:      {config.get('heuristic_mode', 'none')}",
        ])

    lines.extend([
        f"  Workers:             {config.get('workers', 0)} x {config.get('envs_per_worker', 1)}",
        f"  Timesteps:           {config.get('timesteps', 0):,}",
        f"  Checkpoint dir:      {config.get('_checkpoint_dir', 'unset')}",
        f"  Run folder:          {config.get('_run_folder_path', 'unset')}",
        f"  Console mode:        {config.get('console_mode', 'compact')}",
    ])

    eval_episodes = int(config.get("eval_combat_episodes", 0) or 0)
    eval_freq = int(config.get("eval_combat_freq", 0) or 0)
    random_episodes = int(config.get("eval_random_baseline", 0) or 0)
    if eval_episodes > 0 or random_episodes > 0:
        lines.append(
            f"  Eval:                combat={eval_episodes}@every{eval_freq} "
            f"random={random_episodes}@startup"
        )

    lines.append("=" * 60)
    logger.info("\n%s", "\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers (moved from train_rllib.py)
# ---------------------------------------------------------------------------

def _resolve_executable_path(cli_path: str, cli_cwd: str = "") -> str:
    """Attempt to resolve the STS2 executable path."""
    candidate = str(cli_path or "").strip()
    if not candidate:
        return ""

    if os.path.isabs(candidate) or os.path.dirname(candidate):
        paths = [candidate]
        if cli_cwd and not os.path.isabs(candidate):
            paths.insert(0, os.path.join(cli_cwd, candidate))
        for path in paths:
            absolute = os.path.abspath(path)
            if os.path.isfile(absolute):
                return absolute
        return ""

    if cli_cwd:
        cwd_candidate = os.path.abspath(os.path.join(cli_cwd, candidate))
        if os.path.isfile(cwd_candidate):
            return cwd_candidate

    found = shutil.which(candidate)
    return found or ""
