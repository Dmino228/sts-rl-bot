"""
SlayTheSpireEnv — Gymnasium wrapper for CommunicationMod.

This environment does NOT launch the game. CommunicationMod launches us.
We communicate via stdin (receive JSON state) and stdout (send commands).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from typing import Optional, Tuple, Dict, Any, List

from process_manager import GameProcessManager


class SlayTheSpireEnv(gym.Env):
    """
    Phase 1 MVP Gymnasium Environment for Slay the Spire.
    CommunicationMod launches this script. We read game state from stdin,
    send plain-text commands to stdout.
    """

    def __init__(self) -> None:
        super().__init__()

        self.process_manager = GameProcessManager(timeout=120.0)

        # Dummy spaces for MVP — will be replaced in Phase 2
        self.action_space = spaces.Discrete(100)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        self.current_state: Dict[str, Any] = {}
        self._first_reset: bool = True

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wait for the initial game state from CommunicationMod."""
        super().reset(seed=seed)

        if self._first_reset:
            # On first reset, signal ready and wait for game to send initial state
            self.process_manager.signal_ready()
            self._first_reset = False

        try:
            print("Waiting for game state from CommunicationMod...", file=sys.stderr)
            self.current_state = self.process_manager.read_state()
            print(
                f"Received state. game_state={self.current_state.get('game_state', {})}, "
                f"in_game={self.current_state.get('in_game', False)}",
                file=sys.stderr,
            )

            # If we land on the main menu, start a new run
            if not self.current_state.get("in_game", False):
                print("At main menu. Sending START ironclad...", file=sys.stderr)
                self.process_manager.send_command("START ironclad")
                self.current_state = self.process_manager.read_state()
                print("New run started.", file=sys.stderr)

        except Exception as e:
            print(f"Exception during reset: {e}", file=sys.stderr)
            raise

        dummy_obs = np.zeros(10, dtype=np.float32)
        return dummy_obs, {"raw_state": self.current_state}

    def step(
        self, action: str
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Send a plain-text command, read the next state.

        Args:
            action: A plain-text CommunicationMod command, e.g. "PLAY 1 0", "END", "PROCEED"

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        try:
            self.process_manager.send_command(action)
            self.current_state = self.process_manager.read_state()
        except Exception as e:
            print(f"Exception during step: {e}", file=sys.stderr)
            return (
                np.zeros(10, dtype=np.float32),
                0.0,
                True,
                False,
                {"error": str(e), "raw_state": self.current_state},
            )

        # Check for termination
        terminated = False
        game_state = self.current_state.get("game_state", {})

        # Game over (death or victory screen)
        if not self.current_state.get("in_game", True):
            terminated = True

        # Act 1 boundary
        if isinstance(game_state, dict):
            act = game_state.get("act", 1)
            if act > 1:
                terminated = True

        dummy_obs = np.zeros(10, dtype=np.float32)
        reward = 0.0
        truncated = False
        info: Dict[str, Any] = {"raw_state": self.current_state}

        return dummy_obs, reward, terminated, truncated, info

    def get_available_commands(self) -> List[str]:
        """Extract available_commands from the current state."""
        commands = self.current_state.get("available_commands", [])
        if isinstance(commands, list):
            return commands
        return []

    def close(self) -> None:
        """No-op — CommunicationMod manages our lifecycle."""
        self.process_manager.stop()
