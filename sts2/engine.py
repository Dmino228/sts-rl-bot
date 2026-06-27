"""Engine strategy for Slay the Spire 2 through sts2-cli."""

from __future__ import annotations

from typing import Any, Optional

from engine import (
    ActionMapperProtocol,
    ActionMaskerProtocol,
    GameEngine,
    ProcessManagerProtocol,
    StateEncoderProtocol,
)
from sts2.action_space import StS2ActionMapper, StS2ActionMasker
from sts2.process_manager import StS2CliProcessManager
from sts2.state_encoder import StS2StateEncoder, normalize_sts2_state


VALID_STS2_CHARACTERS = frozenset(
    {"Ironclad", "Silent", "Defect", "Necrobinder", "Regent"}
)
CHARACTER_ALIASES = {name.lower(): name for name in VALID_STS2_CHARACTERS}


class StS2Engine(GameEngine):
    """Concrete engine for the headless C#/.NET sts2-cli pipeline."""

    game_version = "sts2"
    valid_characters = VALID_STS2_CHARACTERS

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
    ) -> ProcessManagerProtocol:
        return StS2CliProcessManager(
            timeout=timeout,
            worker_dir=worker_dir,
            cli_path=sts2_cli_path or "sts2-cli",
            cli_args=sts2_cli_args or [],
            cli_cwd=sts2_cli_cwd,
        )

    def create_state_encoder(self) -> StateEncoderProtocol:
        return StS2StateEncoder()

    def create_action_mapper(self) -> ActionMapperProtocol:
        return StS2ActionMapper()

    def create_action_masker(self) -> ActionMaskerProtocol:
        return StS2ActionMasker()

    def normalize_character(self, character_class: str) -> str:
        normalized = CHARACTER_ALIASES.get(character_class.strip().lower())
        if normalized:
            return normalized
        return character_class.strip()

    def start_run_command(self, character_class: str) -> dict[str, Any]:
        return {"cmd": "start_run", "character": character_class}

    def reset_run_state(
        self,
        *,
        process_manager: ProcessManagerProtocol,
        character_class: str,
        seed: Optional[int],
        options: Optional[dict[str, Any]],
        ascension: int = 0,
        lang: str = "en",
    ) -> dict[str, Any]:
        options = options or {}
        run_seed = options.get("seed", options.get("sts2_seed", seed))
        command = self.start_run_command(character_class)
        command["ascension"] = int(options.get("ascension", ascension))
        command["lang"] = str(options.get("lang", lang))
        if run_seed is not None:
            command["seed"] = str(run_seed)

        process_manager.send_command(command)
        state = process_manager.read_state()
        if state.get("type") == "error":
            raise RuntimeError(f"sts2-cli start_run failed: {state.get('message', state)}")
        return state

    def normalize_state(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        return normalize_sts2_state(raw_state)
