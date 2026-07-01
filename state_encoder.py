"""Compatibility exports for the default STS1 state encoder.

Game-specific encoders live under ``sts1/`` and ``sts2/``. New code should use
the engine factory so the selected game version controls the encoder.
"""

from __future__ import annotations

from engine import StateEncoderProtocol
from engine_factory import create_game_engine
from sts1.state_encoder import StateEncoder


def create_state_encoder(game_version: int | str = 1) -> StateEncoderProtocol:
    """Create a state encoder for the requested game version."""
    return create_game_engine(game_version).create_state_encoder()


__all__ = ["StateEncoder", "StateEncoderProtocol", "create_state_encoder"]
