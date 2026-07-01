"""Process-manager interfaces and compatibility exports.

The legacy Java/CommunicationMod implementation lives in ``sts1/``. STS2 uses
``sts2-cli`` and stdin/stdout under ``sts2/``. New code should go through the
engine factory; ``GameProcessManager`` remains as the STS1 compatibility name.
"""

from __future__ import annotations

from engine import ProcessManagerProtocol
from engine_factory import create_game_engine
from sts1.process_manager import GameProcessManager, StS1ProcessManager
from sts2.process_manager import StS2CliProcessManager


def create_process_manager(
    game_version: int | str = 1,
    **kwargs: object,
) -> ProcessManagerProtocol:
    """Create a process manager for the requested game version."""
    return create_game_engine(game_version).create_process_manager(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "GameProcessManager",
    "ProcessManagerProtocol",
    "StS1ProcessManager",
    "StS2CliProcessManager",
    "create_process_manager",
]
