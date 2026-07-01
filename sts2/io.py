"""Thin stdin/stdout helpers for sts2-cli JSON-lines communication."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class StS2StdIOOverlay:
    """Serialize commands and parse JSON state lines for sts2-cli."""

    def encode_command(self, command: Any) -> str:
        if isinstance(command, Mapping):
            return json.dumps(command, ensure_ascii=True, separators=(",", ":")) + "\n"

        if isinstance(command, str):
            stripped = command.strip()
            if not stripped:
                raise ValueError("Cannot send an empty sts2-cli command.")
            if stripped.startswith("{"):
                json.loads(stripped)
                return stripped + "\n"
            if stripped == "quit":
                return json.dumps({"cmd": "quit"}, separators=(",", ":")) + "\n"

        raise TypeError(
            "sts2-cli commands must be JSON-like mappings or serialized JSON objects."
        )

    def decode_state_line(self, line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped:
            return None
        state = json.loads(stripped)
        if not isinstance(state, dict):
            return None
        return state
