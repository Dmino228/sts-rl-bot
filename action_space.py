"""Compatibility exports for the default STS1 action implementation.

Game-specific action logic lives under ``sts1/`` and ``sts2/``. New code should
prefer ``engine_factory.create_game_engine(...).create_action_mapper()`` and
``create_action_masker()``. These names remain for older scripts and tests.
"""

from __future__ import annotations

from engine import ActionMapperProtocol, ActionMaskerProtocol
from engine_factory import create_game_engine
from sts1.action_space import ActionMapper, ActionMasker


def create_action_mapper(game_version: int | str = 1) -> ActionMapperProtocol:
    """Create an action mapper for the requested game version."""
    return create_game_engine(game_version).create_action_mapper()


def create_action_masker(game_version: int | str = 1) -> ActionMaskerProtocol:
    """Create an action masker for the requested game version."""
    return create_game_engine(game_version).create_action_masker()


__all__ = [
    "ActionMapper",
    "ActionMasker",
    "ActionMapperProtocol",
    "ActionMaskerProtocol",
    "create_action_mapper",
    "create_action_masker",
]
