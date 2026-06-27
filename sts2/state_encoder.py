"""Stub state encoder for Slay the Spire 2.

The final encoder will be based on sts2-cli/spire-codex data structures. Until
then this class preserves the shared observation contract and can consume an
engine-provided flat observation when sts2-cli emits one.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class StS2StateEncoder:
    """Temporary fixed-size STS2 encoder."""

    def __init__(self, size: int = 205) -> None:
        self.shape = (size,)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32,
        )

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        obs = np.zeros(self.shape, dtype=np.float32)
        raw_obs = state.get("observation")
        if raw_obs is None:
            raw_obs = state.get("game_state", {}).get("observation", [])

        if raw_obs is not None:
            values = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
            limit = min(values.size, obs.size)
            obs[:limit] = values[:limit]

        np.clip(obs, -1.0, 1.0, out=obs)
        return obs
