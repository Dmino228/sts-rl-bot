"""Small masked env used to verify RLlib without launching Slay the Spire."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rllib.env_wrapper import RLLibActionMaskEnv


RLLIB_SMOKE_ENV_NAME = "sts-rllib-smoke-mask-v0"


class MaskedCounterEnv(gym.Env):
    """A tiny deterministic masked-action env with STS-shaped spaces."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 8) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(205,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(100)
        self.episode_length = episode_length
        self.step_count = 0
        self._mask = np.ones(100, dtype=np.int8)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self._mask = self._make_mask()
        return self._observation(), {"action_mask": self._mask.copy()}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        valid = bool(self._mask[int(action)] > 0)
        reward = 1.0 if valid else -1.0
        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        self._mask = self._make_mask()
        return (
            self._observation(),
            reward,
            terminated,
            False,
            {"action_mask": self._mask.copy(), "valid_action": valid},
        )

    def get_action_mask(self) -> np.ndarray:
        return self._mask.copy()

    def _make_mask(self) -> np.ndarray:
        mask = np.zeros(100, dtype=np.int8)
        mask[65] = 1
        mask[66] = 1
        mask[68 + (self.step_count % 10)] = 1
        return mask

    def _observation(self) -> np.ndarray:
        obs = np.zeros(205, dtype=np.float32)
        obs[0] = self.step_count / max(self.episode_length, 1)
        return obs


def register_smoke_env() -> None:
    from ray.tune.registry import register_env

    register_env(RLLIB_SMOKE_ENV_NAME, lambda _config: RLLibActionMaskEnv(MaskedCounterEnv()))
