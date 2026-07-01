"""Game-engine contracts shared by STS1, STS2, SB3, and RLlib code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import gymnasium as gym
import numpy as np


class ProcessManagerProtocol(Protocol):
    """Minimal I/O lifecycle required by the Gymnasium environment."""

    auto_launch: bool
    _proc: Any

    def launch_game(self) -> None:
        """Start the backing game process when Python is the parent process."""

    def signal_ready(self) -> None:
        """Perform the engine-specific startup handshake, if any."""

    def read_state(self) -> dict[str, Any]:
        """Read and parse one raw JSON state from the game engine."""

    def send_command(self, command: Any) -> None:
        """Send one engine-specific command to the game engine."""

    def stop(self) -> None:
        """Stop the backing process and release I/O resources."""

    def terminate(self) -> None:
        """Hard-stop the process after a crash or timeout."""

    def is_process_alive(self) -> bool:
        """Return whether the managed process is currently running."""


class StateEncoderProtocol(Protocol):
    """Converts raw game JSON to a fixed float observation vector."""

    shape: tuple[int, ...]
    observation_space: gym.Space

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        """Encode one raw state."""


class ActionMapperProtocol(Protocol):
    """Maps a discrete action id to an engine command string."""

    action_space_size: int

    def get_action_string(
        self,
        action_id: int,
        state: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Return the command sent to the process manager."""


class ActionMaskerProtocol(Protocol):
    """Builds a legal-action mask for the current raw state."""

    action_space_size: int

    def get_mask(self, state: dict[str, Any]) -> np.ndarray:
        """Return a binary mask where 1 means the action is legal."""


class GameEngine(ABC):
    """Strategy object that hides game-version-specific implementation details."""

    game_version: str
    valid_characters: frozenset[str] = frozenset()

    @abstractmethod
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
        """Create the process manager for this game engine."""

    @abstractmethod
    def create_state_encoder(self) -> StateEncoderProtocol:
        """Create the state encoder for this game engine."""

    @abstractmethod
    def create_action_mapper(self) -> ActionMapperProtocol:
        """Create the discrete-action mapper for this game engine."""

    @abstractmethod
    def create_action_masker(self) -> ActionMaskerProtocol:
        """Create the legal-action masker for this game engine."""

    def normalize_character(self, character_class: str) -> str:
        """Normalize character ids before they are passed to an engine."""
        return character_class.upper()

    def validate_character(self, character_class: str) -> None:
        """Validate a character id when the engine has a fixed roster."""
        if self.valid_characters and character_class not in self.valid_characters:
            raise ValueError(
                f"Invalid character_class '{character_class}'. "
                f"Must be one of {sorted(self.valid_characters)}"
            )

    def start_run_command(self, character_class: str) -> str:
        """Build the command that starts a new run from an engine menu."""
        return f"START {character_class}"

    def reset_run_state(
        self,
        *,
        process_manager: ProcessManagerProtocol,
        character_class: str,
        seed: Optional[int],
        options: Optional[dict[str, Any]],
        ascension: int = 0,
        lang: str = "en",
    ) -> Optional[dict[str, Any]]:
        """Start/reset a run for engines with a native reset protocol.

        Returning ``None`` tells the shared environment to use the legacy StS1
        menu-cleanup flow.
        """
        return None

    def normalize_state(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        """Adapt an engine-native JSON response to the shared env shape."""
        return raw_state

    def should_launch_on_reset(
        self,
        process_manager: ProcessManagerProtocol,
    ) -> bool:
        """Return True when reset() should launch the process before reading."""
        if not getattr(process_manager, "auto_launch", False):
            return False
        is_alive = getattr(process_manager, "is_process_alive", None)
        if callable(is_alive) and getattr(process_manager, "_proc", None) is not None:
            return not bool(is_alive())
        return bool(
            getattr(process_manager, "_proc", None) is None
        )

    def can_soft_reset_at_act_boundary(
        self,
        current_state: dict[str, Any],
        episode_ended_by_act_completion: bool,
    ) -> bool:
        """Return True when a reset should continue from an act boundary."""
        game_state = current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            return False
        if not current_state.get("in_game", False):
            return False

        screen_type = game_state.get("screen_type", "NONE")
        if screen_type in {"GAME_OVER", "DEATH"}:
            return False

        current_act = game_state.get("act", 1)
        return episode_ended_by_act_completion or current_act > 1
