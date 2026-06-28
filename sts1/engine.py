"""Engine strategy for Slay the Spire 1 through ModTheSpire."""

from __future__ import annotations

from typing import Optional

from engine import (
    ActionMapperProtocol,
    ActionMaskerProtocol,
    GameEngine,
    ProcessManagerProtocol,
    StateEncoderProtocol,
)
from sts1.action_space import ActionMapper, ActionMasker
from sts1.process_manager import GameProcessManager
from sts1.state_encoder import StateEncoder


VALID_CHARACTERS = frozenset({"IRONCLAD", "SILENT", "DEFECT", "WATCHER"})


class StS1Engine(GameEngine):
    """Concrete engine for the legacy Java/CommunicationMod pipeline."""

    game_version = "sts1"
    valid_characters = VALID_CHARACTERS

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
        sts2_cli_cwd: Optional[str] = None,
        sts2_capture_stderr: bool = False,
    ) -> ProcessManagerProtocol:
        return GameProcessManager(
            timeout=timeout,
            worker_dir=worker_dir,
            worker_id=worker_id,
            base_port=base_port,
            use_xvfb=use_xvfb,
            ram_usage=ram_usage,
        )

    def create_state_encoder(self) -> StateEncoderProtocol:
        return StateEncoder()

    def create_action_mapper(self) -> ActionMapperProtocol:
        return ActionMapper()

    def create_action_masker(self) -> ActionMaskerProtocol:
        return ActionMasker()
