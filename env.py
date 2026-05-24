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


# V3.2 reward constants
HP_DELTA_SCALE = 100.0
HP_DELTA_CAP = 100

FLOOR_REWARD = 3.0
COMBAT_VICTORY_REWARD = 2.0
RELIC_REWARD = 1.0
CARD_UPGRADE_REWARD = 0.5
CARD_REMOVE_REWARD = 0.5

DEATH_PENALTY = -25.0
ACT_COMPLETION_REWARD = 15.0

COMBAT_STEP_GRACE = 80
ANTI_STALL_PENALTY = -0.02


# Valid CommunicationMod character identifiers
VALID_CHARACTERS = {"IRONCLAD", "SILENT", "DEFECT", "WATCHER"}


class SlayTheSpireEnv(gym.Env):
    """
    Gymnasium Environment for Slay the Spire via CommunicationMod.

    Supports:
    - Multi-character generalization via `character_class`.
    - Directory-isolated parallel workers via `worker_dir`.
    - Headless xvfb launching via `use_xvfb`.
    """

    def __init__(
        self,
        character_class: str = "IRONCLAD",
        worker_dir: Optional[str] = None,
        use_xvfb: bool = False,
        include_raw_state_in_info: bool = True,
        include_action_mask_in_info: bool = True,
    ) -> None:
        super().__init__()

        # Validate character
        self.character_class = character_class.upper()
        if self.character_class not in VALID_CHARACTERS:
            raise ValueError(
                f"Invalid character_class '{character_class}'. "
                f"Must be one of {VALID_CHARACTERS}"
            )

        self.worker_dir = worker_dir
        self.use_xvfb = use_xvfb
        self.include_raw_state_in_info = include_raw_state_in_info
        self.include_action_mask_in_info = include_action_mask_in_info

        self.process_manager = GameProcessManager(
            timeout=120.0,
            worker_dir=worker_dir,
            use_xvfb=use_xvfb,
        )

        self.action_mapper = ActionMapper()
        self.action_masker = ActionMasker()
        self.state_encoder = StateEncoder()

        self.action_space = gym.spaces.Discrete(self.action_mapper.action_space_size)
        self.observation_space = self.state_encoder.observation_space

        self.current_state: Dict[str, Any] = {}
        self.current_action_mask: Optional[np.ndarray] = None
        self._mask_state_id: Optional[int] = None

        # Reward tracking - V3.2
        self.last_player_hp: Optional[int] = None
        self.last_max_hp: Optional[int] = None
        self.last_monster_total_hp: Optional[int] = None
        self.last_floor: int = 0
        self.last_screen_type: str = "NONE"
        self.last_relic_ids: Optional[set[str]] = None
        self.last_deck_size: Optional[int] = None
        self.last_upgraded_cards: Optional[int] = None
        self.step_count: int = 0
        self.combat_step_count: int = 0
        self.last_act: int = 1
        self.last_in_combat: bool = False
        self.terminal_reward_given: bool = False
        self.episode_ended_by_act_completion: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wait for the initial game state from CommunicationMod."""
        super().reset(seed=seed)

        max_reset_attempts = 3
        for attempt in range(max_reset_attempts):
            try:
                # Launch the game subprocess if running in Python-as-parent mode
                if self.worker_dir is not None and self.process_manager._proc is None:
                    self.process_manager.launch_game()
                    self.process_manager.signal_ready()

                print(f"Waiting for game state from CommunicationMod (attempt {attempt + 1})...", file=sys.stderr)

                # If we already have state from step() (e.g. death/victory screen),
                # reuse it instead of blocking on a fresh read_state() — which would
                # deadlock because CommunicationMod is waiting for OUR command.
                if not self.current_state:
                    self.current_state = self.process_manager.read_state()

                game_state = self.current_state.get("game_state", {})
                if self._can_soft_reset_at_act_boundary(game_state):
                    act = game_state.get("act", 1)
                    floor = game_state.get("floor", 0)
                    screen = game_state.get("screen_type", "NONE")
                    print(
                        f"Soft reset at act boundary: act={act}, "
                        f"floor={floor}, screen={screen}.",
                        file=sys.stderr,
                    )
                    self._reset_reward_tracking(bootstrap_current=True)
                    obs = self.state_encoder.encode(self.current_state)
                    mask = self._refresh_action_mask()
                    return obs, self._make_info(mask)

                # Cleanup Loop: navigate through Game Over / Victory / Score screens
                # back to Main Menu where "start" is available.
                max_cleanup_steps = 30
                for cleanup_step in range(max_cleanup_steps):
                    in_game = self.current_state.get("in_game", False)
                    available_cmds = self.current_state.get("available_commands", [])

                    if not in_game and "start" in available_cmds:
                        print(
                            f"At main menu. Sending START {self.character_class}...",
                            file=sys.stderr,
                        )
                        self.process_manager.send_command(f"START {self.character_class}")
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

                break
            except (ConnectionResetError, EOFError, TimeoutError, Exception) as e:
                print(f"Exception during reset (attempt {attempt + 1}/{max_reset_attempts}): {e}", file=sys.stderr)
                self.process_manager.terminate()
                self.current_state = {}
                if attempt == max_reset_attempts - 1:
                    raise

        self._reset_reward_tracking(bootstrap_current=False)

        obs = self.state_encoder.encode(self.current_state)
        mask = self._refresh_action_mask()
        return obs, self._make_info(mask)

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
            action_str = self.action_mapper.get_action_string(action, self.current_state)
            self.process_manager.send_command(action_str)
            self.current_state = self.process_manager.read_state()
        except (ConnectionResetError, EOFError, TimeoutError, Exception) as e:
            print(f"Watchdog: Java process/socket crashed during step: {e}", file=sys.stderr)
            try:
                self.process_manager.terminate()
                self.process_manager.launch_game()
                self.process_manager.signal_ready()
                self.current_state = {}
            except Exception as restart_err:
                print(f"Watchdog: Error during soft restart in step: {restart_err}", file=sys.stderr)
                self.current_state = {}

            mask = np.zeros(self.action_mapper.action_space_size, dtype=np.int8)
            self.current_action_mask = mask
            self._mask_state_id = id(self.current_state)
            info = self._make_info(mask, error=str(e))
            info["crashed"] = True
            return (
                np.zeros(self.state_encoder.shape, dtype=np.float32),
                0.0,
                True,
                False,
                info,
            )

        # Check for termination
        terminated = False
        game_state = self.current_state.get("game_state", {})

        # Game over (death or victory screen)
        if not self.current_state.get("in_game", True):
            terminated = True

        # Act boundary — terminate only on the transition. reset() will soft-start
        # the next episode from the new act instead of trying to return to menu.
        if isinstance(game_state, dict):
            act = game_state.get("act", 1)
            if act > self.last_act:
                terminated = True
                self.episode_ended_by_act_completion = True

        obs = self.state_encoder.encode(self.current_state)
        reward = self._calculate_reward()
        truncated = False
        mask = self._refresh_action_mask()
        info = self._make_info(mask)

        return obs, reward, terminated, truncated, info

    def get_available_commands(self) -> List[str]:
        """Extract available_commands from the current state."""
        commands = self.current_state.get("available_commands", [])
        if isinstance(commands, list):
            return commands
        return []

    def get_action_mask(self) -> np.ndarray:
        """Returns the binary mask of valid actions for sb3-contrib ActionMasker."""
        if self.current_action_mask is None or self._mask_state_id != id(self.current_state):
            return self._refresh_action_mask()
        return self.current_action_mask

    def _refresh_action_mask(self) -> np.ndarray:
        """Compute and cache the action mask for the current state."""
        self.current_action_mask = self.action_masker.get_mask(self.current_state)
        self._mask_state_id = id(self.current_state)
        return self.current_action_mask

    def _make_info(
        self,
        mask: Optional[np.ndarray] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build Gym info, keeping cluster IPC lean when raw state is not needed."""
        info: Dict[str, Any] = {}
        if error is not None:
            info["error"] = error
        if self.include_raw_state_in_info:
            info["raw_state"] = self.current_state
        if self.include_action_mask_in_info:
            info["action_mask"] = mask if mask is not None else self.get_action_mask()
        return info

    def _can_soft_reset_at_act_boundary(self, game_state: Dict[str, Any]) -> bool:
        """Return True when reset() should continue from a completed act."""
        if not isinstance(game_state, dict):
            return False
        if not self.current_state.get("in_game", False):
            return False

        screen_type = game_state.get("screen_type", "NONE")
        if screen_type in ["GAME_OVER", "DEATH"]:
            return False

        current_act = game_state.get("act", 1)
        return self.episode_ended_by_act_completion or current_act > 1

    def _reset_reward_tracking(self, bootstrap_current: bool = False) -> None:
        """Reset reward deltas, optionally seeding them from current_state."""
        self.last_player_hp = None
        self.last_max_hp = None
        self.last_monster_total_hp = None
        self.last_floor = 0
        self.last_screen_type = "NONE"
        self.last_relic_ids = None
        self.last_deck_size = None
        self.last_upgraded_cards = None
        self.step_count = 0
        self.combat_step_count = 0
        self.last_act = 1
        self.last_in_combat = False
        self.terminal_reward_given = False
        self.episode_ended_by_act_completion = False

        if not bootstrap_current or not self.current_state:
            return

        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            return

        in_combat = self._is_in_combat(self.current_state)
        combat_state = game_state.get("combat_state", {})

        current_hp = None
        current_max_hp = None
        if in_combat and combat_state:
            player_data = combat_state.get("player", {})
            current_hp = player_data.get("current_hp")
            current_max_hp = player_data.get("max_hp")
        else:
            current_hp = game_state.get("current_hp")
            current_max_hp = game_state.get("max_hp")

        self.last_player_hp = current_hp
        self.last_max_hp = current_max_hp
        self.last_monster_total_hp = self._get_total_monster_hp(game_state)
        self.last_floor = game_state.get("floor", 0)
        self.last_screen_type = game_state.get("screen_type", "NONE")
        self.last_relic_ids = self._get_relic_ids(game_state)
        self.last_deck_size = len(game_state.get("deck", []))
        self.last_upgraded_cards = self._count_upgraded_cards(game_state)
        self.last_act = game_state.get("act", 1)
        self.last_in_combat = in_combat

    # ──────────────────────────────────────────────────────────────
    # V3 REWARD HELPERS
    # ──────────────────────────────────────────────────────────────

    def _get_total_monster_hp(self, game_state: Dict[str, Any]) -> int:
        """Sum current_hp of all alive monsters."""
        combat_state = game_state.get("combat_state", {})
        if not combat_state:
            return 0
        monsters = combat_state.get("monsters", [])
        return sum(
            m.get("current_hp", 0)
            for m in monsters
            if not m.get("is_gone", False)
        )

    def _count_upgraded_cards(self, game_state: Dict[str, Any]) -> int:
        """Count total upgrade level across all cards in deck."""
        deck = game_state.get("deck", [])
        return sum(card.get("upgrades", 0) for card in deck)

    def _get_relic_ids(self, game_state: Dict[str, Any]) -> set[str]:
        """Extract stable relic identifiers for delta tracking."""
        relic_ids = set()
        for relic in game_state.get("relics", []):
            relic_id = relic.get("id") or relic.get("name")
            if relic_id:
                relic_ids.add(str(relic_id))
        return relic_ids

    def _bounded_hp_reward(self, delta: int) -> float:
        """Preserve /100 HP scale while clipping extreme state deltas."""
        clipped = max(-HP_DELTA_CAP, min(HP_DELTA_CAP, delta))
        return clipped / HP_DELTA_SCALE

    def _is_in_combat(self, state: Dict[str, Any]) -> bool:
        """Check if we're actively in combat (playing cards)."""
        screen_type = state.get("game_state", {}).get("screen_type", "NONE")
        available_cmds = state.get("available_commands", [])
        return screen_type == "NONE" and "end" in available_cmds

    # ──────────────────────────────────────────────────────────────
    # V3.2 REWARD FUNCTION
    # ──────────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        """V3.2 reward: clipped HP deltas, robust terminal handling."""
        if not self.current_state:
            return 0.0

        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            return 0.0

        reward = 0.0

        # ── Extract current values safely ──
        screen_type = game_state.get("screen_type", "NONE")
        state_in_game = self.current_state.get("in_game", True)
        current_floor = game_state.get("floor", 0)
        current_monster_hp = self._get_total_monster_hp(game_state)
        current_relic_ids = self._get_relic_ids(game_state)
        current_deck_size = len(game_state.get("deck", []))
        current_upgraded_cards = self._count_upgraded_cards(game_state)
        current_act = game_state.get("act", 1)

        # ── Robust HP extraction ──
        # Priority: combat_state.player > game_state > last known
        current_hp = self.last_player_hp        # default to last known
        current_max_hp = self.last_max_hp        # default to last known

        in_combat = self._is_in_combat(self.current_state)
        combat_state = game_state.get("combat_state", {})

        if in_combat and combat_state:
            # In combat — combat_state.player is the authoritative source
            player_data = combat_state.get("player", {})
            current_hp = player_data.get("current_hp", current_hp)
            current_max_hp = player_data.get("max_hp", current_max_hp)
        else:
            # Outside combat — use game_state top-level fields
            gs_hp = game_state.get("current_hp")
            gs_max = game_state.get("max_hp")
            if gs_hp is not None:
                current_hp = gs_hp
            if gs_max is not None:
                current_max_hp = gs_max

        hp_is_known = current_hp is not None
        if current_max_hp is None:
            current_max_hp = game_state.get("max_hp", 80)

        # ══════════════════════════════════════════
        # A. HP PROGRESS REWARDS
        # ══════════════════════════════════════════
        if (
            self.last_in_combat
            and self.last_monster_total_hp is not None
            and (in_combat or screen_type == "COMBAT_REWARD")
        ):
            monster_delta = self.last_monster_total_hp - current_monster_hp
            reward += self._bounded_hp_reward(monster_delta)

        if self.last_player_hp is not None and hp_is_known:
            hp_delta = current_hp - self.last_player_hp
            reward += self._bounded_hp_reward(hp_delta)

        if in_combat:
            # Anti-stall: action-step fallback until a reliable turn counter exists.
            self.combat_step_count += 1
            if self.combat_step_count > COMBAT_STEP_GRACE:
                reward += ANTI_STALL_PENALTY

        # ══════════════════════════════════════════
        # B. MACRO-ECONOMY
        # ══════════════════════════════════════════

        if state_in_game and screen_type not in ["GAME_OVER", "DEATH"]:
            # B1. Floor progress — primary signal
            if current_floor > self.last_floor:
                reward += FLOOR_REWARD * (current_floor - self.last_floor)

            # B2. Combat victory (screen transition guard)
            if screen_type == "COMBAT_REWARD" and self.last_screen_type != "COMBAT_REWARD":
                reward += COMBAT_VICTORY_REWARD
                self.combat_step_count = 0

            # B3. Relics acquired, tracked by ID so swaps are not missed
            if self.last_relic_ids is not None:
                reward += RELIC_REWARD * len(current_relic_ids - self.last_relic_ids)

            # B4. Card upgraded
            if self.last_upgraded_cards is not None:
                upgrade_delta = max(0, current_upgraded_cards - self.last_upgraded_cards)
                reward += CARD_UPGRADE_REWARD * upgrade_delta

            # B5. Card removed (outside combat only — filters exhaust noise)
            if self.last_deck_size is not None and not in_combat:
                removed = max(0, self.last_deck_size - current_deck_size)
                reward += CARD_REMOVE_REWARD * removed

        # ══════════════════════════════════════════
        # C. TERMINAL STATES
        # ══════════════════════════════════════════

        # C1. Death — include terminal no-longer-in-game states.
        dead_screen = screen_type in ["GAME_OVER", "DEATH"]
        dead_by_hp = hp_is_known and current_hp <= 0
        terminal_failure = dead_by_hp or dead_screen
        if terminal_failure and not self.terminal_reward_given:
            reward += DEATH_PENALTY
            self.terminal_reward_given = True

        # C2. Act completion — one-shot act transition bonus.
        if (
            current_act > self.last_act
            and screen_type not in ["GAME_OVER", "DEATH"]
            and not self.terminal_reward_given
        ):
            reward += ACT_COMPLETION_REWARD
            self.terminal_reward_given = True

        # Reset combat counter when leaving combat
        if not in_combat and self.last_screen_type == "NONE":
            self.combat_step_count = 0

        # ══════════════════════════════════════════
        # UPDATE TRACKING STATE (must be LAST)
        # ══════════════════════════════════════════
        self.last_player_hp = current_hp
        self.last_max_hp = current_max_hp
        self.last_monster_total_hp = current_monster_hp
        self.last_floor = current_floor
        self.last_screen_type = screen_type
        self.last_relic_ids = current_relic_ids
        self.last_deck_size = current_deck_size
        self.last_upgraded_cards = current_upgraded_cards
        self.last_act = current_act
        self.last_in_combat = in_combat

        # Periodic logging
        self.step_count += 1
        if self.step_count % 500 == 0:
            print(
                f"[REWARD V3.2 #{self.step_count}] r={reward:.3f} | "
                f"hp={current_hp}/{current_max_hp} floor={current_floor} "
                f"screen={screen_type} combat_steps={self.combat_step_count}",
                file=sys.stderr,
            )

        return reward

    def close(self) -> None:
        """No-op — CommunicationMod manages our lifecycle."""
        self.process_manager.stop()
