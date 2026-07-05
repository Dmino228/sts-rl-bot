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
from sts2.state_encoder import (
    StS2StateEncoder,
    StS2StateEncoderFlat,
    normalize_sts2_state,
)


VALID_STS2_CHARACTERS = frozenset(
    {"Ironclad", "Silent", "Defect", "Necrobinder", "Regent"}
)
CHARACTER_ALIASES = {name.lower(): name for name in VALID_STS2_CHARACTERS}


def _normalize_encoder_mode(raw: str | None) -> str:
    mode = str(raw or "compact").strip().lower().replace("-", "_")
    if mode in {"", "compact", "default"}:
        return "compact"
    if mode in {"flat", "identity", "card_id", "card_ids", "codex_flat"}:
        return "flat"
    raise ValueError(
        f"Unsupported STS2 encoder mode: {raw!r}. Expected 'compact' or 'flat'."
    )


class StS2Engine(GameEngine):
    """Concrete engine for the headless C#/.NET sts2-cli pipeline."""

    game_version = "sts2"
    valid_characters = VALID_STS2_CHARACTERS

    def __init__(self, *, encoder_mode: str = "compact") -> None:
        self.encoder_mode = _normalize_encoder_mode(encoder_mode)

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
        sts2_recycle_every_episodes: int = 0,
        sts2_recycle_every_steps: int = 0,
        sts2_recycle_rss_mb: float = 0.0,
    ) -> ProcessManagerProtocol:
        return StS2CliProcessManager(
            timeout=timeout,
            worker_dir=worker_dir,
            cli_path=sts2_cli_path or "sts2-cli",
            cli_args=sts2_cli_args or [],
            cli_cwd=sts2_cli_cwd,
            capture_stderr=sts2_capture_stderr,
            recycle_every_episodes=sts2_recycle_every_episodes,
            recycle_every_steps=sts2_recycle_every_steps,
            recycle_rss_mb=sts2_recycle_rss_mb,
        )

    def create_state_encoder(self) -> StateEncoderProtocol:
        if self.encoder_mode == "flat":
            return StS2StateEncoderFlat()
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

        curriculum_mode = str(options.get("curriculum_mode", "full_run")).strip().lower()
        if curriculum_mode == "combat":
            state = self._apply_curriculum_deck(
                process_manager=process_manager,
                options=options,
                current_state=state,
            )
            state = self._enter_curriculum_combat(
                process_manager=process_manager,
                options=options,
            )
        elif curriculum_mode not in {"", "full_run", "none"}:
            raise RuntimeError(f"Unsupported STS2 curriculum_mode: {curriculum_mode}")
        return state

    def normalize_state(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        return normalize_sts2_state(raw_state)

    def _enter_curriculum_combat(
        self,
        *,
        process_manager: ProcessManagerProtocol,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        room_type = str(options.get("combat_room_type", "combat") or "combat")
        encounter = str(
            options.get("combat_encounter", "SHRINKER_BEETLE_WEAK")
            or "SHRINKER_BEETLE_WEAK"
        )
        command: dict[str, Any] = {
            "cmd": "enter_room",
            "type": room_type,
            "encounter": encounter,
        }
        process_manager.send_command(command)
        state = process_manager.read_state()
        if state.get("type") == "error":
            raise RuntimeError(
                f"sts2-cli combat curriculum failed: {state.get('message', state)}"
            )
        return state

    def _apply_curriculum_deck(
        self,
        *,
        process_manager: ProcessManagerProtocol,
        options: dict[str, Any],
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        deck_spec = options.get("deck_spec")
        if not isinstance(deck_spec, dict):
            return current_state
        if not bool(deck_spec.get("apply_to_headless", False)):
            return current_state

        command: dict[str, Any] = {
            "cmd": "set_player",
            "deck": list(deck_spec.get("cards") or []),
            "hp": int(deck_spec.get("hp", 80) or 80),
            "max_hp": int(deck_spec.get("max_hp", 80) or 80),
            "relics": list(deck_spec.get("relics") or []),
            "potions": list(deck_spec.get("potions") or []),
        }
        process_manager.send_command(command)
        state = process_manager.read_state()
        if state.get("type") == "error":
            raise RuntimeError(
                f"sts2-cli deck setup failed: {state.get('message', state)}"
            )
        return state
