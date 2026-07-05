"""Factory for game-version-specific engine strategies."""

from __future__ import annotations

from typing import Any

from engine import GameEngine


def normalize_game_version(game_version: Any) -> str:
    """Normalize CLI/config game-version values to stable engine keys."""
    raw = str(game_version).strip().lower().replace("-", "").replace("_", "")
    if raw in {"1", "sts1", "slaythespire1", "slaythespire"}:
        return "sts1"
    if raw in {"2", "sts2", "slaythespire2"}:
        return "sts2"
    raise ValueError(
        f"Unsupported game_version '{game_version}'. "
        "Expected one of: 1, 2, sts1, sts2."
    )


def create_game_engine(game_version: Any = 1, **options: Any) -> GameEngine:
    """Instantiate the engine strategy for the requested game version."""
    normalized = normalize_game_version(game_version)
    if normalized == "sts1":
        from sts1.engine import StS1Engine

        return StS1Engine()
    if normalized == "sts2":
        from sts2.engine import StS2Engine

        return StS2Engine(encoder_mode=str(options.get("sts2_encoder_mode", "compact")))
    raise AssertionError(f"Unhandled normalized game version: {normalized}")
