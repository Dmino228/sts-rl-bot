"""Engine strategy for Slay the Spire 2 through sts2-cli."""

from __future__ import annotations

from typing import Optional

from engine import (
    ActionMapperProtocol,
    ActionMaskerProtocol,
    GameEngine,
    ProcessManagerProtocol,
    StateEncoderProtocol,
)
from sts2.action_space import StS2ActionMapper, StS2ActionMasker
from sts2.process_manager import StS2CliProcessManager
from sts2.state_encoder import StS2StateEncoder


class StS2Engine(GameEngine):
    """Concrete engine for the headless C#/.NET sts2-cli pipeline."""

    game_version = "sts2"
    valid_characters = frozenset()

    def create_process_manager(
        self,
        *,
        timeout: float,
        worker_dir: Optional[str],
        worker_id: Optional[int],
        base_port: int,
        use_xvfb: bool,
        ram_usage: str,
        sts2_cli_path: Optional[str] = None,
        sts2_cli_args: Optional[list[str]] = None,
    ) -> ProcessManagerProtocol:
        return StS2CliProcessManager(
            timeout=timeout,
            worker_dir=worker_dir,
            cli_path=sts2_cli_path or "sts2-cli",
            cli_args=sts2_cli_args or [],
        )

    def create_state_encoder(self) -> StateEncoderProtocol:
        return StS2StateEncoder()

    def create_action_mapper(self) -> ActionMapperProtocol:
        return StS2ActionMapper()

    def create_action_masker(self) -> ActionMaskerProtocol:
        return StS2ActionMasker()

    def normalize_character(self, character_class: str) -> str:
        return character_class.lower()

    def start_run_command(self, character_class: str) -> str:
        return f"START {character_class}"
