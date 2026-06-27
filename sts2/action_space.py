"""Stubs for the Slay the Spire 2 discrete action interface.

The concrete mapping will be filled once sts2-cli/spire-codex exposes the
complete command vocabulary. The class already satisfies the shared engine
contract so RLlib and the root env can be wired without knowing STS2 internals.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


STS2_ACTION_SPACE_SIZE = 100


class StS2ActionMapper:
    """Temporary STS2 action mapper with state-driven command fallback."""

    def __init__(self) -> None:
        self.action_space_size = STS2_ACTION_SPACE_SIZE

    def get_action_string(
        self,
        action_id: int,
        state: Optional[dict[str, Any]] = None,
    ) -> str:
        if not 0 <= action_id < self.action_space_size:
            raise ValueError(f"Invalid action ID: {action_id}")

        if state:
            commands = state.get("available_commands", [])
            if isinstance(commands, list) and action_id < len(commands):
                return str(commands[action_id]).upper()

        if action_id == 98:
            return "STATE"
        if action_id == 99:
            return "WAIT"
        return f"ACTION {action_id}"


class StS2ActionMasker:
    """Temporary STS2 action masker.

    Prefer an engine-provided mask when present. Otherwise expose only the first
    N raw available commands, plus STATE/WAIT as safe poll fallbacks.
    """

    def __init__(self) -> None:
        self.action_space_size = STS2_ACTION_SPACE_SIZE

    def get_mask(self, state: dict[str, Any]) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)

        raw_mask = state.get("action_mask")
        if raw_mask is not None:
            values = np.asarray(raw_mask, dtype=np.int8).reshape(-1)
            limit = min(values.size, self.action_space_size)
            mask[:limit] = values[:limit]

        commands = state.get("available_commands", [])
        if isinstance(commands, list):
            for idx in range(min(len(commands), self.action_space_size)):
                mask[idx] = 1

        if not np.any(mask):
            mask[98] = 1
            mask[99] = 1

        return mask
