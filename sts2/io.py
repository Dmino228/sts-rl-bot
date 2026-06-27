"""Thin stdin/stdout helpers for sts2-cli JSON-lines communication."""

from __future__ import annotations

import json
from typing import Any


class StS2StdIOOverlay:
    """Serialize commands and parse JSON state lines for sts2-cli."""

    def encode_command(self, command: str) -> str:
        return command.rstrip("\n") + "\n"

    def decode_state_line(self, line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped:
            return None
        state = json.loads(stripped)
        if not isinstance(state, dict):
            return None
        return state
