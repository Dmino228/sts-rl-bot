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
from action_space import ActionMapper, ActionMasker
from state_encoder import StateEncoder


class SlayTheSpireEnv(gym.Env):
    """
    Phase 1 MVP Gymnasium Environment for Slay the Spire.
    CommunicationMod launches this script. We read game state from stdin,
    send plain-text commands to stdout.
    """

    def __init__(self) -> None:
        super().__init__()

        self.process_manager = GameProcessManager(timeout=120.0)

        self.action_mapper = ActionMapper()
        self.action_masker = ActionMasker()
        self.state_encoder = StateEncoder()

        self.action_space = gym.spaces.Discrete(self.action_mapper.action_space_size)
        self.observation_space = self.state_encoder.observation_space

        self.current_state: Dict[str, Any] = {}
        
        self.previous_hp: Optional[int] = None
        self.previous_floor: Optional[int] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wait for the initial game state from CommunicationMod."""
        super().reset(seed=seed)

        try:
            print("Waiting for game state from CommunicationMod...", file=sys.stderr)

            # If we already have state from step() (e.g. death/victory screen),
            # reuse it instead of blocking on a fresh read_state() — which would
            # deadlock because CommunicationMod is waiting for OUR command.
            if not self.current_state:
                self.current_state = self.process_manager.read_state()

            # Cleanup Loop: navigate through Game Over / Victory / Score screens
            # back to Main Menu where "start" is available.
            max_cleanup_steps = 30
            for cleanup_step in range(max_cleanup_steps):
                in_game = self.current_state.get("in_game", False)
                available_cmds = self.current_state.get("available_commands", [])

                print(
                    f"[reset cleanup #{cleanup_step}] in_game={in_game}, "
                    f"cmds={available_cmds}",
                    file=sys.stderr,
                )

                if not in_game and "start" in available_cmds:
                    print("At main menu. Sending START ironclad...", file=sys.stderr)
                    self.process_manager.send_command("START ironclad")
                    break
                elif "proceed" in available_cmds:
                    self.process_manager.send_command("PROCEED")
                elif "return" in available_cmds:
                    self.process_manager.send_command("RETURN")
                elif "confirm" in available_cmds:
                    self.process_manager.send_command("CONFIRM")
                else:
                    # Safe fallback to advance frame and re-poll state
                    self.process_manager.send_command("STATE")

                self.current_state = self.process_manager.read_state()
            else:
                raise RuntimeError(
                    f"Could not reach main menu after {max_cleanup_steps} cleanup steps. "
                    f"Last state: in_game={self.current_state.get('in_game')}, "
                    f"cmds={self.current_state.get('available_commands')}"
                )

            # Wait until NeowRoom is loaded (i.e. in_game == True)
            for wait_step in range(max_cleanup_steps):
                self.current_state = self.process_manager.read_state()
                if self.current_state.get("in_game", False):
                    print("New run started.", file=sys.stderr)
                    break
                print(
                    f"[reset wait #{wait_step}] Still transitioning...",
                    file=sys.stderr,
                )
                # Request next state if the game is still transitioning
                self.process_manager.send_command("STATE")
            else:
                raise RuntimeError(
                    f"Game did not enter in_game=True after START. "
                    f"Last state: {self.current_state.get('game_state', {}).get('screen_type', 'unknown')}"
                )

        except Exception as e:
            print(f"Exception during reset: {e}", file=sys.stderr)
            raise
            
        # Reset reward tracking
        self.previous_hp = None
        self.previous_floor = None

        obs = self.state_encoder.encode(self.current_state)
        mask = self.action_masker.get_mask(self.current_state)
        return obs, {"raw_state": self.current_state, "action_mask": mask}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Send a mapped text command, read the next state.

        Args:
            action: An integer mapped to a CommunicationMod command.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        try:
            action_str = self.action_mapper.get_action_string(action)
            self.process_manager.send_command(action_str)
            self.current_state = self.process_manager.read_state()
        except Exception as e:
            print(f"Exception during step: {e}", file=sys.stderr)
            mask = np.zeros(self.action_space.n, dtype=np.int8)
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,
                False,
                {"error": str(e), "raw_state": self.current_state, "action_mask": mask},
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

        obs = self.state_encoder.encode(self.current_state)
        reward = self._calculate_reward()
        truncated = False
        mask = self.action_masker.get_mask(self.current_state)
        info: Dict[str, Any] = {"raw_state": self.current_state, "action_mask": mask}

        return obs, reward, terminated, truncated, info

    def get_available_commands(self) -> List[str]:
        """Extract available_commands from the current state."""
        commands = self.current_state.get("available_commands", [])
        if isinstance(commands, list):
            return commands
        return []

    def get_action_mask(self) -> np.ndarray:
        """Returns the binary mask of valid actions for sb3-contrib ActionMasker."""
        return self.action_masker.get_mask(self.current_state)

    def _calculate_reward(self) -> float:
        """Placeholder reward: penalize HP loss, reward floor progress."""
        if not self.current_state:
            return 0.0
            
        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            return 0.0
            
        current_hp = game_state.get("current_hp")
        current_floor = game_state.get("floor")
        
        reward = 0.0
        
        if current_hp is not None and self.previous_hp is not None:
            hp_delta = current_hp - self.previous_hp
            reward += hp_delta * 0.1
        self.previous_hp = current_hp
            
        if current_floor is not None and self.previous_floor is not None:
            if current_floor > self.previous_floor:
                reward += 1.0
        self.previous_floor = current_floor
            
        return reward

    def close(self) -> None:
        """No-op — CommunicationMod manages our lifecycle."""
        self.process_manager.stop()
