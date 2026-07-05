"""Game-version router exposed as a Gymnasium environment."""

import json
import os
import random
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from collections import Counter, deque
from typing import Optional, Tuple, Dict, Any, List

from engine_factory import create_game_engine
from sts2.curriculum_profiles import sample_curriculum_profile
from sts2.deck_generator import build_combat_deck_spec
from sts2.encounters import combat_pool_ids


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
MAX_COMBAT_STEPS = 250


# Valid CommunicationMod character identifiers
VALID_CHARACTERS = {"IRONCLAD", "SILENT", "DEFECT", "WATCHER"}


class SlayTheSpireEnv(gym.Env):
    """
    Gymnasium Environment for Slay the Spire engines.

    Supports:
    - Dynamic engine selection via `game_version` (1/sts1 or 2/sts2).
    - Multi-character generalization via `character_class`.
    - Directory-isolated parallel workers via `worker_dir`.
    - StS1 Java/CommunicationMod and StS2 sts2-cli process managers.
    """

    def __init__(
        self,
        character_class: str = "IRONCLAD",
        worker_dir: Optional[str] = None,
        worker_id: Optional[int] = None,
        base_port: int = 12340,
        use_xvfb: bool = False,
        include_raw_state_in_info: bool = True,
        include_action_mask_in_info: bool = True,
        ram_usage: str = "default",
        game_version: int | str = 1,
        process_timeout: float = 120.0,
        sts2_cli_path: Optional[str] = None,
        sts2_cli_args: Optional[List[str]] = None,
        sts2_cli_cwd: Optional[str] = None,
        sts2_capture_stderr: bool = False,
        sts2_recycle_every_episodes: int = 0,
        sts2_recycle_every_steps: int = 0,
        sts2_recycle_rss_mb: float = 0.0,
        sts2_ascension: int = 0,
        sts2_lang: str = "en",
        sts2_curriculum_mode: str = "full_run",
        sts2_reward_mode: str = "full_v3_2",
        sts2_combat_room_type: str = "combat",
        sts2_combat_encounter: str = "SHRINKER_BEETLE_WEAK",
        sts2_combat_enemy_pool: str = "fixed",
        sts2_combat_damage_reward_scale: float = 0.01,
        sts2_combat_hp_loss_reward_scale: float = 0.01,
        sts2_combat_action_penalty: float = 0.001,
        sts2_debug_episodes: int = 0,
        sts2_seed: Optional[int | str] = None,
        deck_mode: str = "",
        sts2_debug_jsonl_path: Optional[str] = None,
        sts2_deck_duplicate_cap: int = 2,
        sts2_deck_allow_problematic_cards: bool = False,
        sts2_encoder_mode: str = "compact",
        curriculum_mix: str = "",
    ) -> None:
        super().__init__()

        self.sts2_encoder_mode = str(sts2_encoder_mode or "compact").strip().lower()
        self.engine = create_game_engine(
            game_version,
            sts2_encoder_mode=self.sts2_encoder_mode,
        )
        self.game_version = self.engine.game_version
        self.character_class = self.engine.normalize_character(character_class)
        self.engine.validate_character(self.character_class)

        self.worker_dir = worker_dir
        self.worker_id = worker_id
        self.base_port = base_port
        self.use_xvfb = use_xvfb
        self.include_raw_state_in_info = include_raw_state_in_info
        self.include_action_mask_in_info = include_action_mask_in_info
        self.ram_usage = ram_usage.lower()

        self.process_manager = self.engine.create_process_manager(
            timeout=process_timeout,
            worker_dir=worker_dir,
            worker_id=worker_id,
            base_port=base_port,
            use_xvfb=use_xvfb,
            ram_usage=self.ram_usage,
            sts2_cli_path=sts2_cli_path,
            sts2_cli_args=sts2_cli_args,
            sts2_cli_cwd=sts2_cli_cwd,
            sts2_capture_stderr=sts2_capture_stderr,
            sts2_recycle_every_episodes=sts2_recycle_every_episodes,
            sts2_recycle_every_steps=sts2_recycle_every_steps,
            sts2_recycle_rss_mb=sts2_recycle_rss_mb,
        )

        self.action_mapper = self.engine.create_action_mapper()
        self.action_masker = self.engine.create_action_masker()
        self.state_encoder = self.engine.create_state_encoder()

        self.action_space = gym.spaces.Discrete(self.action_mapper.action_space_size)
        self.observation_space = self.state_encoder.observation_space

        self.current_state: Dict[str, Any] = {}
        self.current_action_mask: Optional[np.ndarray] = None
        self._mask_state_id: Optional[int] = None
        
        # Local tracking to prevent unselecting cards in loops
        self.current_selections: set[int] = set()

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
        self.combat_victory_reward_given: bool = False
        self.episode_ended_by_act_completion: bool = False
        self.sts2_ascension = int(sts2_ascension)
        self.sts2_lang = sts2_lang
        self.sts2_curriculum_mode = sts2_curriculum_mode.strip().lower()
        self.sts2_reward_mode = sts2_reward_mode.strip().lower()
        self.sts2_combat_room_type = sts2_combat_room_type
        self.sts2_combat_encounter = sts2_combat_encounter
        self.sts2_combat_enemy_pool = sts2_combat_enemy_pool.strip().lower()
        self._combat_encounter_pool = combat_pool_ids(
            self.sts2_combat_enemy_pool,
            fixed_encounter=self.sts2_combat_encounter,
        )
        self._current_combat_encounter = self._combat_encounter_pool[0]
        self._current_combat_encounter_pool = list(self._combat_encounter_pool)
        self._current_combat_enemy_pool = self.sts2_combat_enemy_pool
        self.sts2_combat_damage_reward_scale = float(sts2_combat_damage_reward_scale)
        self.sts2_combat_hp_loss_reward_scale = float(sts2_combat_hp_loss_reward_scale)
        self.sts2_combat_action_penalty = float(sts2_combat_action_penalty)
        self.sts2_debug_episodes = max(0, int(sts2_debug_episodes))
        self.sts2_seed = sts2_seed
        self.deck_mode = str(deck_mode or "").strip().lower() or "unspecified"
        self.curriculum_mix = str(curriculum_mix or "").strip()
        self._current_deck_mode = self.deck_mode
        self.sts2_deck_duplicate_cap = max(1, int(sts2_deck_duplicate_cap))
        self.sts2_deck_allow_problematic_cards = bool(sts2_deck_allow_problematic_cards)
        self.sts2_debug_jsonl_path = str(sts2_debug_jsonl_path or "").strip()
        self._curriculum_mix_rng = random.Random(
            f"{self.sts2_seed}|{self.worker_id}|{self.curriculum_mix}"
        )
        self._current_curriculum_profile = ""
        self._debug_jsonl_failed = False
        self._episode_index = 0
        self._combat_initial_hp: Optional[int] = None
        self._combat_initial_monster_hp: Optional[int] = None
        self._combat_last_monster_hp: Optional[int] = None
        self._combat_min_monster_hp: Optional[int] = None
        self._combat_damage_dealt_total: float = 0.0
        self._combat_initial_boss_hp: Optional[int] = None
        self._combat_last_boss_hp: Optional[int] = None
        self._combat_min_boss_hp: Optional[int] = None
        self._combat_boss_damage_dealt_total: float = 0.0
        self._combat_max_turn: int = 0
        self._combat_boss_milestones_awarded: set[float] = set()
        self._combat_end_turn_count: int = 0
        self._combat_end_turn_with_energy: int = 0
        self._combat_end_turn_with_playable_attack: int = 0
        self._combat_end_turn_with_playable_block_when_incoming: int = 0
        self._combat_incoming_steps: int = 0
        self._combat_block_when_incoming_count: int = 0
        self._combat_cards_played_count: int = 0
        self._combat_power_played_count: int = 0
        self._combat_cards_played_by_id: Counter[str] = Counter()
        self._last_combat_reset_debug: Dict[str, Any] = {}
        self._current_deck_spec: Dict[str, Any] = {}
        self._last_reward_parts: Dict[str, float] = {}
        self._last_done_reason = ""
        self._last_action_id: Optional[int] = None
        self._last_action_command: Any = None
        self._last_action_summary: Any = None
        self._recent_combat_trace = deque(maxlen=20)

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
                # Launch the game subprocess if this engine runs Python-as-parent.
                if self._should_launch_process_on_reset():
                    self.process_manager.launch_game()
                    self.process_manager.signal_ready()
                else:
                    self._maybe_recycle_process_on_reset()

                reset_options = self._engine_reset_options(options, seed=seed)
                native_reset_state = self.engine.reset_run_state(
                    process_manager=self.process_manager,
                    character_class=self.character_class,
                    seed=seed,
                    options=reset_options,
                    ascension=self.sts2_ascension,
                    lang=self.sts2_lang,
                )
                if native_reset_state is not None:
                    self._record_run_started()
                    self.current_state = self.engine.normalize_state(native_reset_state)
                    break

                print(f"Waiting for game state from CommunicationMod (attempt {attempt + 1})...", file=sys.stderr)

                # If we already have state from step() (e.g. death/victory screen),
                # reuse it instead of blocking on a fresh read_state() — which would
                # deadlock because CommunicationMod is waiting for OUR command.
                if not self.current_state:
                    self.current_state = self.engine.normalize_state(
                        self.process_manager.read_state()
                    )

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
                        start_command = self.engine.start_run_command(
                            self.character_class
                        )
                        print(
                            f"At main menu. Sending {start_command}...",
                            file=sys.stderr,
                        )
                        self.process_manager.send_command(start_command)
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

                    self.current_state = self.engine.normalize_state(
                        self.process_manager.read_state()
                    )
                else:
                    raise RuntimeError(
                        f"Could not reach main menu after {max_cleanup_steps} cleanup steps. "
                        f"Last state: in_game={self.current_state.get('in_game')}, "
                        f"cmds={self.current_state.get('available_commands')}"
                    )

                # Wait until NeowRoom is loaded (i.e. in_game == True)
                for wait_step in range(max_cleanup_steps):
                    self.current_state = self.engine.normalize_state(
                        self.process_manager.read_state()
                    )
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
        self.current_selections.clear()
        if self.current_state:
            self.current_state["_env_selections"] = self.current_selections.copy()
        self._episode_index += 1
        self._reset_combat_episode_tracking()
        self._recent_combat_trace.clear()
        self._last_combat_reset_debug = {}
        if self._is_combat_curriculum():
            self._last_combat_reset_debug = self._build_combat_reset_debug()
            self._maybe_log_combat_reset_debug()

        obs = self.state_encoder.encode(self.current_state)
        mask = self._refresh_action_mask()
        return obs, self._make_info(mask)

    def _engine_reset_options(
        self,
        options: Optional[Dict[str, Any]],
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Merge shared reset options with STS2 curriculum defaults."""
        reset_options = dict(options or {})
        if self.game_version == "sts2":
            profile_options: Dict[str, Any] = {}
            self._current_curriculum_profile = ""
            if self.curriculum_mix:
                profile = sample_curriculum_profile(
                    self.curriculum_mix,
                    rng=self._curriculum_mix_rng,
                )
                if profile is not None:
                    self._current_curriculum_profile = profile.name
                    profile_options = dict(profile.options)
            reset_options.setdefault("curriculum_mode", self.sts2_curriculum_mode)
            reset_options.setdefault(
                "combat_room_type",
                profile_options.get("combat_room_type", self.sts2_combat_room_type),
            )
            selected_pool = str(
                profile_options.get("combat_enemy_pool", self.sts2_combat_enemy_pool)
            )
            self._current_combat_enemy_pool = selected_pool
            self._current_combat_encounter_pool = combat_pool_ids(
                selected_pool,
                fixed_encounter=str(
                    profile_options.get("combat_encounter", self.sts2_combat_encounter)
                ),
            )
            if "combat_encounter" not in reset_options:
                if profile_options.get("combat_encounter"):
                    reset_options["combat_encounter"] = str(profile_options["combat_encounter"])
                else:
                    reset_options["combat_encounter"] = self._select_combat_encounter(
                        self._current_combat_encounter_pool,
                    )
            if profile_options.get("run_seed") is not None:
                reset_options.setdefault("seed", profile_options["run_seed"])
            if self.sts2_seed is not None:
                reset_options.setdefault("seed", self.sts2_seed)
            self._current_combat_encounter = str(reset_options["combat_encounter"])
            self._current_deck_spec = {}
            self._current_deck_mode = str(
                profile_options.get("deck_mode", self.deck_mode)
            ).strip().lower() or "unspecified"
            if self._is_combat_curriculum():
                deck_spec = build_combat_deck_spec(
                    mode=self._current_deck_mode,
                    character=self.character_class,
                    seed=profile_options.get("deck_seed", reset_options.get("seed", seed)),
                    worker_id=self.worker_id,
                    episode_id=int(
                        profile_options.get("deck_episode_id", self._episode_index + 1)
                    ),
                    duplicate_cap=self.sts2_deck_duplicate_cap,
                    allow_problematic_cards=self.sts2_deck_allow_problematic_cards,
                )
                self._current_deck_spec = deck_spec.to_debug()
                reset_options.setdefault("deck_spec", deck_spec.to_engine_options())
        return reset_options

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
            self._last_action_id = int(action)
            self._last_action_command = action_str
            self._record_combat_trace(action, action_str, self.current_state)
            self.process_manager.send_command(action_str)
            
            old_screen_type = "NONE"
            if self.current_state and "game_state" in self.current_state:
                old_screen_type = self.current_state["game_state"].get("screen_type", "NONE")
                
            self.current_state = self.engine.normalize_state(
                self.process_manager.read_state()
            )
            self._record_env_step()
            
            new_screen_type = "NONE"
            if self.current_state and "game_state" in self.current_state:
                new_screen_type = self.current_state["game_state"].get("screen_type", "NONE")
                
            if new_screen_type != old_screen_type:
                self.current_selections.clear()
            elif new_screen_type in ["GRID", "HAND_SELECT"]:
                if isinstance(action_str, str) and action_str.startswith("CHOOSE "):
                    try:
                        self.current_selections.add(int(action_str.split()[1]))
                    except Exception:
                        pass
                elif action_str in ["CONFIRM", "RETURN", "CANCEL"]:
                    self.current_selections.clear()
            else:
                self.current_selections.clear()
                
            if self.current_state:
                self.current_state["_env_selections"] = self.current_selections.copy()

        except (ConnectionResetError, EOFError, TimeoutError, Exception) as e:
            diagnostics = self._process_diagnostics()
            print(
                f"Watchdog: game process/socket crashed during step: {e} "
                f"diagnostics={diagnostics}",
                file=sys.stderr,
            )
            # Only clean up — do NOT restart here. ThreadedVecEnv auto-calls
            # reset() after terminated=True, and reset() already handles
            # launch_game(). Restarting here blocks Ctrl+C shutdown because
            # the new Java process launches before KeyboardInterrupt propagates.
            try:
                self.process_manager.terminate()
            except Exception as cleanup_err:
                print(f"Watchdog: Error during cleanup: {cleanup_err}", file=sys.stderr)
            self.current_state = {}

            mask = np.zeros(self.action_mapper.action_space_size, dtype=np.int8)
            self.current_action_mask = mask
            self._mask_state_id = id(self.current_state)
            info = self._make_info(mask, error=str(e))
            info["crashed"] = True
            if diagnostics:
                info["process_diagnostics"] = diagnostics
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
        if self._is_combat_curriculum():
            if not self._last_done_reason:
                game_state = self.current_state.get("game_state", {})
                if isinstance(game_state, dict):
                    self._last_done_reason = self._combat_done_reason(
                        game_state,
                        self._current_player_hp(game_state),
                    )
            done_reason = self._last_done_reason
            terminated = done_reason in {"win", "loss"}
            truncated = done_reason == "timeout"
        elif self.combat_step_count > MAX_COMBAT_STEPS:
            terminated = True

        mask = self._refresh_action_mask()
        info = self._make_info(mask)
        if self._is_combat_curriculum():
            info["combat_done_reason"] = self._last_done_reason or "ongoing"
            self._maybe_log_debug_episode(reward, terminated, truncated)

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

    def action_masks(self) -> np.ndarray:
        """Return the current legal-action mask for mask-aware RL libraries."""
        return self.get_action_mask()

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
        info["progress_metrics"] = self._progress_metrics()
        if self._last_combat_reset_debug:
            info["combat_reset"] = self._last_combat_reset_debug
        return info

    def _progress_metrics(self) -> Dict[str, Any]:
        """Expose run-progress counters for training dashboards and logs."""
        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            game_state = {}

        context = self.current_state.get("context", {})
        if not isinstance(context, dict):
            context = {}

        floor = self._safe_int(game_state.get("floor", context.get("floor", 0)), 0)
        act = self._safe_int(game_state.get("act", context.get("act", 1)), 1)
        room_type = str(context.get("room_type") or game_state.get("room_type") or "")
        screen_type = str(game_state.get("screen_type") or "")

        boss_reached = (
            act > 1
            or floor >= 16
            or room_type.lower() == "boss"
            or "boss" in screen_type.lower()
        )
        boss_killed = act > 1 or self.episode_ended_by_act_completion

        metrics = {
            "floor": floor,
            "act": act,
            "boss_reached": float(boss_reached),
            "boss_killed": float(boss_killed),
            "act2": float(act >= 2),
        }
        metrics.update(self._combat_progress_metrics())
        return metrics

    def _can_soft_reset_at_act_boundary(self, game_state: Dict[str, Any]) -> bool:
        """Return True when reset() should continue from a completed act."""
        return self.engine.can_soft_reset_at_act_boundary(
            self.current_state,
            self.episode_ended_by_act_completion,
        )

    def _should_launch_process_on_reset(self) -> bool:
        """Return True when reset should launch or relaunch the engine process."""
        if hasattr(self.process_manager, "auto_launch"):
            return self.engine.should_launch_on_reset(self.process_manager)
        return self.worker_dir is not None and getattr(self.process_manager, "_proc", None) is None

    def _maybe_recycle_process_on_reset(self) -> None:
        recycler = getattr(self.process_manager, "recycle_if_needed", None)
        if not callable(recycler):
            return
        if recycler():
            self.current_state = {}

    def _record_run_started(self) -> None:
        recorder = getattr(self.process_manager, "record_run_started", None)
        if callable(recorder):
            recorder()

    def _record_env_step(self) -> None:
        recorder = getattr(self.process_manager, "record_env_step", None)
        if callable(recorder):
            recorder()

    def _process_diagnostics(self) -> Dict[str, Any]:
        getter = getattr(self.process_manager, "diagnostic_snapshot", None)
        if not callable(getter):
            return {}
        try:
            snapshot = getter()
        except Exception:
            return {}
        if not isinstance(snapshot, dict):
            return {}
        snapshot["env"] = {
            "encounter_id": self._current_combat_encounter,
            "combat_step": self.combat_step_count,
            "turn": self._state_turn(self.current_state),
            "last_action_id": self._last_action_id,
            "last_action": self._last_action_summary,
            "hand": self._hand_debug(self.current_state),
            "enemies": self._enemy_debug(self.current_state),
            "recent_trace": list(self._recent_combat_trace),
        }
        return snapshot

    def _record_combat_trace(
        self,
        action_id: int,
        command: Any,
        state: Dict[str, Any],
    ) -> None:
        if not self._is_combat_curriculum():
            return
        action_summary = self._describe_action(command, state)
        self._last_action_summary = action_summary
        self._record_action_quality(command, state, action_summary)
        self._recent_combat_trace.append(
            {
                "encounter_id": self._current_combat_encounter,
                "combat_step": self.combat_step_count,
                "turn": self._state_turn(state),
                "action_id": int(action_id),
                "action": action_summary,
                "hand": self._hand_debug(state),
                "enemies": self._enemy_debug(state),
            }
        )

    def _record_action_quality(
        self,
        command: Any,
        state: Dict[str, Any],
        action_summary: Any,
    ) -> None:
        if not isinstance(command, dict):
            return
        if not self._is_in_combat(state):
            return
        incoming_damage = self._incoming_damage(state)
        if incoming_damage > 0:
            self._combat_incoming_steps += 1
        action = str(command.get("action") or "")
        if action == "end_turn":
            self._combat_end_turn_count += 1
            player = self._state_player(state)
            if self._safe_int(player.get("energy"), 0) > 0:
                self._combat_end_turn_with_energy += 1
            if self._has_playable_card_type(state, "attack"):
                self._combat_end_turn_with_playable_attack += 1
            player_block = self._safe_int(player.get("block"), 0)
            if (
                incoming_damage > player_block
                and self._has_playable_block_card(state)
            ):
                self._combat_end_turn_with_playable_block_when_incoming += 1
            return

        if action == "play_card" and isinstance(action_summary, dict):
            card = action_summary.get("chosen_card")
            if isinstance(card, dict):
                self._combat_cards_played_count += 1
                card_id = str(card.get("id") or card.get("name") or "UNKNOWN")
                self._combat_cards_played_by_id[card_id] += 1
                card_type = str(card.get("type") or "").lower()
                if card_type == "power":
                    self._combat_power_played_count += 1
                if incoming_damage > 0 and self._card_block_value(card) > 0:
                    self._combat_block_when_incoming_count += 1

    def _describe_action(self, command: Any, state: Dict[str, Any]) -> Any:
        if not isinstance(command, dict):
            return command
        summary: Dict[str, Any] = {
            "cmd": command.get("cmd"),
            "action": command.get("action"),
        }
        args = command.get("args")
        if isinstance(args, dict):
            summary["args"] = dict(args)
            if command.get("action") == "play_card":
                card_index = self._safe_int(args.get("card_index"), -1)
                target_index = self._safe_int(args.get("target_index"), -1)
                card = self._find_card_by_index(self._state_hand(state), card_index)
                target = self._find_enemy_by_index(self._state_enemies(state), target_index)
                summary["chosen_card"] = self._card_debug(card, card_index)
                if target_index >= 0:
                    summary["target"] = self._enemy_target_debug(target, target_index)
        return summary

    def _incoming_damage(self, state: Dict[str, Any]) -> int:
        total = 0
        for enemy in self._state_enemies(state):
            if not isinstance(enemy, dict):
                continue
            if enemy.get("is_gone", False):
                continue
            for intent in enemy.get("intents") or []:
                if not isinstance(intent, dict):
                    continue
                intent_type = str(intent.get("type") or "").lower()
                if "attack" in intent_type:
                    total += self._safe_int(intent.get("damage"), 0)
            for key in ("intent_damage", "damage"):
                if enemy.get(key) is not None:
                    total += self._safe_int(enemy.get(key), 0)
                    break
        return max(0, total)

    def _has_playable_card_type(self, state: Dict[str, Any], card_type: str) -> bool:
        wanted = card_type.strip().lower()
        for card in self._state_hand(state):
            if not isinstance(card, dict) or not bool(card.get("can_play", True)):
                continue
            if str(card.get("type") or "").lower() == wanted:
                return True
        return False

    def _has_playable_block_card(self, state: Dict[str, Any]) -> bool:
        for card in self._state_hand(state):
            if not isinstance(card, dict) or not bool(card.get("can_play", True)):
                continue
            if self._card_block_value(card) > 0:
                return True
        return False

    def _card_block_value(self, card: Dict[str, Any]) -> int:
        stats = card.get("stats")
        if isinstance(stats, dict) and stats.get("block") is not None:
            return self._safe_int(stats.get("block"), 0)
        return self._safe_int(card.get("block"), 0)

    def _state_turn(self, state: Dict[str, Any]) -> Any:
        if not isinstance(state, dict):
            return None
        for key in ("turn", "round"):
            if state.get(key) is not None:
                return state.get(key)
        game_state = state.get("game_state", {})
        if isinstance(game_state, dict):
            combat_state = game_state.get("combat_state", {})
            if isinstance(combat_state, dict):
                return combat_state.get("turn", combat_state.get("round"))
        return None

    def _state_hand(self, state: Dict[str, Any]) -> List[Any]:
        if isinstance(state.get("hand"), list):
            return state["hand"]
        game_state = state.get("game_state", {})
        if isinstance(game_state, dict):
            combat_state = game_state.get("combat_state", {})
            if isinstance(combat_state, dict) and isinstance(combat_state.get("hand"), list):
                return combat_state["hand"]
        return []

    def _state_enemies(self, state: Dict[str, Any]) -> List[Any]:
        if isinstance(state.get("enemies"), list):
            return state["enemies"]
        game_state = state.get("game_state", {})
        if isinstance(game_state, dict):
            combat_state = game_state.get("combat_state", {})
            if isinstance(combat_state, dict):
                monsters = combat_state.get("monsters")
                if isinstance(monsters, list):
                    return monsters
        return []

    def _state_player(self, state: Dict[str, Any]) -> Dict[str, Any]:
        player = state.get("player")
        if isinstance(player, dict):
            return player
        game_state = state.get("game_state", {})
        if isinstance(game_state, dict):
            combat_state = game_state.get("combat_state", {})
            if isinstance(combat_state, dict) and isinstance(combat_state.get("player"), dict):
                return combat_state["player"]
        return {}

    def _state_deck(self, state: Dict[str, Any]) -> List[Any]:
        player = self._state_player(state)
        if isinstance(player.get("deck"), list):
            return player["deck"]
        game_state = state.get("game_state", {})
        if isinstance(game_state, dict) and isinstance(game_state.get("deck"), list):
            return game_state["deck"]
        return []

    def _state_named_list(self, state: Dict[str, Any], key: str) -> List[Any]:
        player = self._state_player(state)
        if isinstance(player.get(key), list):
            return player[key]
        game_state = state.get("game_state", {})
        if isinstance(game_state, dict) and isinstance(game_state.get(key), list):
            return game_state[key]
        return []

    def _pile_size(self, state: Dict[str, Any], key: str) -> Optional[int]:
        found = self._find_nested_list(state, key)
        if found is not None:
            return len(found)
        count_key = f"{key}_count"
        found_count = self._find_nested_number(state, count_key)
        return int(found_count) if found_count is not None else None

    def _find_nested_list(self, value: Any, key: str, depth: int = 0) -> Optional[List[Any]]:
        if depth > 4:
            return None
        if isinstance(value, dict):
            direct = value.get(key)
            if isinstance(direct, list):
                return direct
            for nested_key in ("game_state", "combat_state", "player", "piles"):
                nested = value.get(nested_key)
                found = self._find_nested_list(nested, key, depth + 1)
                if found is not None:
                    return found
        return None

    def _find_nested_number(self, value: Any, key: str, depth: int = 0) -> Optional[float]:
        if depth > 4:
            return None
        if isinstance(value, dict):
            direct = value.get(key)
            if isinstance(direct, (int, float)):
                return float(direct)
            for nested_key in ("game_state", "combat_state", "player", "piles"):
                nested = value.get(nested_key)
                found = self._find_nested_number(nested, key, depth + 1)
                if found is not None:
                    return found
        return None

    def _hand_debug(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for slot, card in enumerate(self._state_hand(state)[:10]):
            if not isinstance(card, dict):
                continue
            cards.append(self._card_debug(card, self._safe_int(card.get("index"), slot), slot))
        return cards

    def _enemy_debug(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        enemies: List[Dict[str, Any]] = []
        for slot, enemy in enumerate(self._state_enemies(state)[:5]):
            if not isinstance(enemy, dict):
                continue
            enemies.append(self._enemy_target_debug(enemy, slot))
        return enemies

    def _card_debug(
        self,
        card: Any,
        fallback_index: int,
        slot: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not isinstance(card, dict):
            return {"index": fallback_index, "missing": True}
        result = {
            "index": self._safe_int(card.get("index"), fallback_index),
            "id": card.get("id"),
            "name": card.get("name"),
            "cost": card.get("cost"),
            "can_play": card.get("can_play"),
            "target_type": card.get("target_type"),
            "type": card.get("type"),
            "stats": card.get("stats"),
            "block": card.get("block"),
            "damage": card.get("damage"),
        }
        if slot is not None:
            result["slot"] = slot
        return result

    def _deck_card_debug(self, card: Any, slot: int) -> Dict[str, Any]:
        if not isinstance(card, dict):
            return {"slot": slot, "missing": True, "value": card}
        return {
            "slot": slot,
            "id": card.get("id"),
            "name": card.get("name"),
            "upgrades": self._safe_int(card.get("upgrades"), 1 if card.get("upgraded") else 0),
            "upgraded": bool(card.get("upgraded") or self._safe_int(card.get("upgrades"), 0) > 0),
            "cost": card.get("cost"),
            "type": card.get("type"),
            "rarity": card.get("rarity", card.get("rarity_key")),
        }

    def _named_object_debug(self, item: Any, slot: int) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {"slot": slot, "value": item}
        return {
            "slot": slot,
            "id": item.get("id"),
            "name": item.get("name"),
            "rarity": item.get("rarity", item.get("rarity_key")),
            "target_type": item.get("target_type"),
        }

    def _enemy_target_debug(self, enemy: Any, fallback_index: int) -> Dict[str, Any]:
        if not isinstance(enemy, dict):
            return {"index": fallback_index, "missing": True}
        return {
            "index": self._safe_int(enemy.get("index"), fallback_index),
            "name": enemy.get("name"),
            "hp": enemy.get("hp", enemy.get("current_hp")),
            "max_hp": enemy.get("max_hp"),
            "intent": enemy.get("intent"),
            "intents": enemy.get("intents"),
        }

    def _find_card_by_index(self, hand: List[Any], card_index: int) -> Any:
        for slot, card in enumerate(hand):
            if not isinstance(card, dict):
                continue
            if self._safe_int(card.get("index"), slot) == card_index:
                return card
        if 0 <= card_index < len(hand):
            return hand[card_index]
        return None

    def _find_enemy_by_index(self, enemies: List[Any], target_index: int) -> Any:
        if 0 <= target_index < len(enemies):
            return enemies[target_index]
        return None

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
        self.combat_victory_reward_given = False
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

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _get_total_monster_hp(self, game_state: Dict[str, Any]) -> int:
        """Sum current_hp of all alive monsters."""
        observed = self._get_total_monster_hp_observed(game_state)
        return int(observed) if observed is not None else 0

    def _get_total_monster_hp_observed(self, game_state: Dict[str, Any]) -> Optional[int]:
        """Return monster HP only when the current state still exposes monsters."""
        combat_state = game_state.get("combat_state", {})
        if not isinstance(combat_state, dict) or not combat_state:
            return None
        monsters = combat_state.get("monsters", [])
        if not isinstance(monsters, list):
            return None
        return sum(
            m.get("current_hp", 0)
            for m in monsters
            if isinstance(m, dict)
            if not m.get("is_gone", False)
        )

    def _get_primary_boss_hp_observed(self, game_state: Dict[str, Any]) -> Optional[int]:
        """Return HP for the likely primary boss monster when visible.

        Act 1 boss encounters can include adds. For diagnostics and boss-specific
        reward shaping we use the living monster with the largest max HP as the
        primary boss proxy.
        """
        combat_state = game_state.get("combat_state", {})
        if not isinstance(combat_state, dict) or not combat_state:
            return None
        monsters = combat_state.get("monsters", [])
        if not isinstance(monsters, list):
            return None
        candidates = [
            monster
            for monster in monsters
            if isinstance(monster, dict) and not monster.get("is_gone", False)
        ]
        if not candidates:
            return None
        boss = max(
            candidates,
            key=lambda monster: (
                self._safe_int(monster.get("max_hp"), 0),
                self._safe_int(monster.get("current_hp"), 0),
            ),
        )
        return self._safe_int(
            boss.get("current_hp", boss.get("hp")),
            0,
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
        if self._uses_combat_reward_mode():
            return self._calculate_combat_reward()
        return self._calculate_full_v3_2_reward()

    def _calculate_full_v3_2_reward(self) -> float:
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

        stall_failure = False
        if in_combat:
            self.combat_victory_reward_given = False
            # Anti-stall: action-step fallback until a reliable turn counter exists.
            self.combat_step_count += 1
            if self.combat_step_count > COMBAT_STEP_GRACE:
                reward += ANTI_STALL_PENALTY
                
            if self.combat_step_count > MAX_COMBAT_STEPS:
                stall_failure = True

        # ══════════════════════════════════════════
        # B. MACRO-ECONOMY
        # ══════════════════════════════════════════

        if state_in_game and screen_type not in ["GAME_OVER", "DEATH"]:
            # B1. Floor progress — primary signal
            if current_floor > self.last_floor:
                reward += FLOOR_REWARD * (current_floor - self.last_floor)
                self.combat_victory_reward_given = False

            # B2. Combat victory (screen transition guard & one-shot victory guard)
            if screen_type == "COMBAT_REWARD" and not self.combat_victory_reward_given:
                reward += COMBAT_VICTORY_REWARD
                self.combat_victory_reward_given = True
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
        terminal_failure = dead_by_hp or dead_screen or stall_failure
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

    def _is_combat_curriculum(self) -> bool:
        return self.game_version == "sts2" and self.sts2_curriculum_mode == "combat"

    def _select_combat_encounter(self, pool: Optional[List[str]] = None) -> str:
        if not self._is_combat_curriculum():
            return self.sts2_combat_encounter
        encounter_pool = list(pool or self._combat_encounter_pool)
        if len(encounter_pool) == 1:
            self._current_combat_encounter = encounter_pool[0]
            return self._current_combat_encounter
        index = int(self.np_random.integers(0, len(encounter_pool)))
        self._current_combat_encounter = encounter_pool[index]
        return self._current_combat_encounter

    def _uses_combat_reward_mode(self) -> bool:
        return self._is_combat_curriculum() and self.sts2_reward_mode in {
            "combat_sparse",
            "combat_dense",
            "combat_boss_potential",
        }

    def _reset_combat_episode_tracking(self) -> None:
        self._last_done_reason = ""
        self._last_reward_parts = {}
        self._last_action_summary = None
        self._combat_last_monster_hp = None
        self._combat_min_monster_hp = None
        self._combat_damage_dealt_total = 0.0
        self._combat_initial_boss_hp = None
        self._combat_last_boss_hp = None
        self._combat_min_boss_hp = None
        self._combat_boss_damage_dealt_total = 0.0
        self._combat_max_turn = 0
        self._combat_boss_milestones_awarded.clear()
        self._combat_end_turn_count = 0
        self._combat_end_turn_with_energy = 0
        self._combat_end_turn_with_playable_attack = 0
        self._combat_end_turn_with_playable_block_when_incoming = 0
        self._combat_incoming_steps = 0
        self._combat_block_when_incoming_count = 0
        self._combat_cards_played_count = 0
        self._combat_power_played_count = 0
        self._combat_cards_played_by_id.clear()
        if not self._is_combat_curriculum():
            self._combat_initial_hp = None
            self._combat_initial_monster_hp = None
            return

        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            self._combat_initial_hp = None
            self._combat_initial_monster_hp = None
            return

        self._combat_initial_hp = self._current_player_hp(game_state)
        self._combat_initial_monster_hp = self._get_total_monster_hp(game_state)
        self._combat_last_monster_hp = self._combat_initial_monster_hp
        self._combat_min_monster_hp = self._combat_initial_monster_hp
        self._combat_initial_boss_hp = (
            self._get_primary_boss_hp_observed(game_state)
            or self._combat_initial_monster_hp
        )
        self._combat_last_boss_hp = self._combat_initial_boss_hp
        self._combat_min_boss_hp = self._combat_initial_boss_hp
        self.last_player_hp = self._combat_initial_hp
        self.last_monster_total_hp = self._combat_initial_monster_hp
        self.last_in_combat = self._is_in_combat(self.current_state)

    def _calculate_combat_reward(self) -> float:
        if not self.current_state:
            self._last_reward_parts = {}
            return 0.0

        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            self._last_reward_parts = {}
            return 0.0

        self.combat_step_count += 1
        current_hp = self._current_player_hp(game_state)
        done_reason = self._combat_done_reason(game_state, current_hp)
        self._last_done_reason = done_reason
        observed_monster_hp = self._get_total_monster_hp_observed(game_state)
        if observed_monster_hp is None:
            if done_reason == "win":
                current_monster_hp = 0
            else:
                current_monster_hp = (
                    self._combat_last_monster_hp
                    if self._combat_last_monster_hp is not None
                    else self.last_monster_total_hp
                    if self.last_monster_total_hp is not None
                    else 0
                )
        else:
            current_monster_hp = observed_monster_hp
        observed_boss_hp = self._get_primary_boss_hp_observed(game_state)
        if observed_boss_hp is None:
            if done_reason == "win":
                current_boss_hp = 0
            else:
                current_boss_hp = (
                    self._combat_last_boss_hp
                    if self._combat_last_boss_hp is not None
                    else current_monster_hp
                )
        else:
            current_boss_hp = observed_boss_hp
        self._update_combat_potential_tracking(current_monster_hp, current_boss_hp)

        parts: Dict[str, float] = {}
        if self.sts2_reward_mode == "combat_boss_potential":
            parts.update(
                self._combat_boss_potential_reward_parts(
                    done_reason=done_reason,
                    current_hp=current_hp,
                    current_boss_hp=current_boss_hp,
                )
            )
        elif done_reason == "win":
            parts["terminal_win"] = 1.0
        elif done_reason in {"loss", "timeout"}:
            parts[f"terminal_{done_reason}"] = -1.0
        elif self.sts2_reward_mode == "combat_dense":
            if self.last_monster_total_hp is not None:
                damage_dealt = max(0, self.last_monster_total_hp - current_monster_hp)
                parts["damage_dealt"] = damage_dealt * self.sts2_combat_damage_reward_scale
            if self.last_player_hp is not None and current_hp is not None:
                hp_lost = max(0, self.last_player_hp - current_hp)
                parts["hp_lost"] = -hp_lost * self.sts2_combat_hp_loss_reward_scale
            parts["action_penalty"] = -self.sts2_combat_action_penalty

        reward = float(sum(parts.values()))
        self._last_reward_parts = parts

        self.last_player_hp = current_hp
        self.last_max_hp = self._current_player_max_hp(game_state)
        self.last_monster_total_hp = current_monster_hp
        self._combat_last_monster_hp = current_monster_hp
        self._combat_last_boss_hp = current_boss_hp
        self.last_screen_type = str(game_state.get("screen_type", "NONE"))
        self.last_in_combat = self._is_in_combat(self.current_state)
        self.step_count += 1
        return reward

    def _update_combat_potential_tracking(
        self,
        current_monster_hp: int,
        current_boss_hp: int,
    ) -> None:
        if self._combat_min_monster_hp is None:
            self._combat_min_monster_hp = current_monster_hp
        else:
            self._combat_min_monster_hp = min(self._combat_min_monster_hp, current_monster_hp)
        if self._combat_last_monster_hp is not None:
            self._combat_damage_dealt_total += float(
                max(0, self._combat_last_monster_hp - current_monster_hp)
            )
        if self._combat_min_boss_hp is None:
            self._combat_min_boss_hp = current_boss_hp
        else:
            self._combat_min_boss_hp = min(self._combat_min_boss_hp, current_boss_hp)
        if self._combat_last_boss_hp is not None:
            self._combat_boss_damage_dealt_total += float(
                max(0, self._combat_last_boss_hp - current_boss_hp)
            )
        turn_value = self._safe_int(self._state_turn(self.current_state), 0)
        self._combat_max_turn = max(self._combat_max_turn, turn_value)

    def _combat_boss_potential_reward_parts(
        self,
        *,
        done_reason: str,
        current_hp: Optional[int],
        current_boss_hp: int,
    ) -> Dict[str, float]:
        parts: Dict[str, float] = {}
        initial_boss_hp = max(1, int(self._combat_initial_boss_hp or self._combat_initial_monster_hp or 0))
        if self._combat_last_boss_hp is not None:
            damage_delta = max(0, self._combat_last_boss_hp - current_boss_hp)
            parts["boss_hp_fraction_removed"] = float(damage_delta) / float(initial_boss_hp)
        if self.last_player_hp is not None and current_hp is not None:
            hp_delta = max(0, self.last_player_hp - current_hp)
            initial_hp = max(1, int(self._combat_initial_hp or self.last_player_hp or 1))
            parts["player_hp_fraction_lost"] = -0.35 * float(hp_delta) / float(initial_hp)
        for threshold in (0.75, 0.50, 0.25):
            if threshold in self._combat_boss_milestones_awarded:
                continue
            if current_boss_hp <= initial_boss_hp * threshold:
                parts[f"boss_hp_below_{int(threshold * 100)}"] = 0.25
                self._combat_boss_milestones_awarded.add(threshold)
        parts["action_penalty"] = -self.sts2_combat_action_penalty
        if done_reason == "win":
            parts["terminal_win"] = 10.0
        elif done_reason == "loss":
            parts["terminal_loss"] = -1.0
        elif done_reason == "timeout":
            parts["terminal_timeout"] = -3.0
        return parts

    def _combat_done_reason(
        self,
        game_state: Dict[str, Any],
        current_hp: Optional[int],
    ) -> str:
        screen_type = str(game_state.get("screen_type", "")).upper()
        if screen_type in {"GAME_OVER", "DEATH"} or not self.current_state.get("in_game", True):
            return "loss"
        if current_hp is not None and current_hp <= 0:
            return "loss"
        if screen_type == "COMBAT_REWARD":
            return "win"
        if self.combat_step_count > MAX_COMBAT_STEPS:
            return "timeout"
        return ""

    def _current_player_hp(self, game_state: Dict[str, Any]) -> Optional[int]:
        combat_state = game_state.get("combat_state", {})
        if isinstance(combat_state, dict):
            player = combat_state.get("player", {})
            if isinstance(player, dict) and player.get("current_hp") is not None:
                return self._safe_int(player.get("current_hp"), 0)
        if game_state.get("current_hp") is not None:
            return self._safe_int(game_state.get("current_hp"), 0)
        return None

    def _current_player_max_hp(self, game_state: Dict[str, Any]) -> Optional[int]:
        combat_state = game_state.get("combat_state", {})
        if isinstance(combat_state, dict):
            player = combat_state.get("player", {})
            if isinstance(player, dict) and player.get("max_hp") is not None:
                return self._safe_int(player.get("max_hp"), 0)
        if game_state.get("max_hp") is not None:
            return self._safe_int(game_state.get("max_hp"), 0)
        return None

    def _combat_progress_metrics(self) -> Dict[str, Any]:
        if not self._is_combat_curriculum():
            return {}

        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            game_state = {}
        current_hp = self._current_player_hp(game_state)
        reason = self._last_done_reason or "ongoing"
        observed_monster_hp = self._get_total_monster_hp_observed(game_state)
        if observed_monster_hp is None:
            monster_hp = (
                0
                if reason == "win"
                else self._combat_last_monster_hp
                if self._combat_last_monster_hp is not None
                else 0
            )
        else:
            monster_hp = observed_monster_hp
        observed_boss_hp = self._get_primary_boss_hp_observed(game_state)
        if observed_boss_hp is None:
            boss_hp = (
                0
                if reason == "win"
                else self._combat_last_boss_hp
                if self._combat_last_boss_hp is not None
                else monster_hp
            )
        else:
            boss_hp = observed_boss_hp
        hp_initial = self._combat_initial_hp
        hp_lost = 0.0
        if hp_initial is not None and current_hp is not None:
            hp_lost = float(max(0, hp_initial - current_hp))
        initial_monster_hp = float(max(0, self._combat_initial_monster_hp or 0))
        min_monster_hp = float(
            self._combat_min_monster_hp
            if self._combat_min_monster_hp is not None
            else monster_hp
        )
        damage_dealt_total = float(max(0.0, initial_monster_hp - min_monster_hp))
        if self._combat_damage_dealt_total > damage_dealt_total:
            damage_dealt_total = float(self._combat_damage_dealt_total)
        initial_boss_hp = float(
            max(0, self._combat_initial_boss_hp or self._combat_initial_monster_hp or 0)
        )
        min_boss_hp = float(
            self._combat_min_boss_hp
            if self._combat_min_boss_hp is not None
            else boss_hp
        )
        boss_damage_dealt_total = float(max(0.0, initial_boss_hp - min_boss_hp))
        if self._combat_boss_damage_dealt_total > boss_damage_dealt_total:
            boss_damage_dealt_total = float(self._combat_boss_damage_dealt_total)
        boss_fraction_removed = (
            min(1.0, max(0.0, boss_damage_dealt_total / initial_boss_hp))
            if initial_boss_hp > 0
            else 0.0
        )
        monster_hp_remaining_on_loss = float(monster_hp) if reason == "loss" else 0.0
        boss_hp_remaining_on_loss = float(boss_hp) if reason == "loss" else 0.0

        return {
            "combat_win": float(reason == "win"),
            "combat_loss": float(reason == "loss"),
            "combat_timeout": float(reason == "timeout"),
            "combat_steps": float(self.combat_step_count),
            "hp_remaining_on_win": float(current_hp or 0) if reason == "win" else 0.0,
            "hp_lost": hp_lost,
            "monster_hp_remaining_on_loss": monster_hp_remaining_on_loss,
            "boss_hp_remaining_on_loss": boss_hp_remaining_on_loss,
            "boss_hp_fraction_removed": boss_fraction_removed,
            "min_boss_hp_reached": min_boss_hp,
            "damage_dealt_total": damage_dealt_total,
            "turns_survived": float(self._combat_max_turn),
            "end_turn_with_energy": float(self._combat_end_turn_with_energy),
            "end_turn_with_energy_rate": self._safe_ratio(
                self._combat_end_turn_with_energy,
                self._combat_end_turn_count,
            ),
            "end_turn_with_playable_attack": float(
                self._combat_end_turn_with_playable_attack
            ),
            "end_turn_with_playable_attack_rate": self._safe_ratio(
                self._combat_end_turn_with_playable_attack,
                self._combat_end_turn_count,
            ),
            "end_turn_with_playable_block_when_incoming_damage": float(
                self._combat_end_turn_with_playable_block_when_incoming
            ),
            "end_turn_with_playable_block_when_incoming_damage_rate": self._safe_ratio(
                self._combat_end_turn_with_playable_block_when_incoming,
                self._combat_end_turn_count,
            ),
            "power_play_rate": self._safe_ratio(
                self._combat_power_played_count,
                self._combat_cards_played_count,
            ),
            "block_when_incoming_damage_rate": self._safe_ratio(
                self._combat_block_when_incoming_count,
                self._combat_incoming_steps,
            ),
            "cards_played_by_id": dict(self._combat_cards_played_by_id),
            "encounter_id": self._current_combat_encounter,
            "encounter_pool": self._current_combat_enemy_pool,
            "encounter_pool_ids": list(self._current_combat_encounter_pool),
            "terminated_reason": reason,
            "deck_mode": self._current_deck_mode,
            "deck_source": self._deck_source(),
            "deck_size": self._last_combat_reset_debug.get("deck_size", 0),
            "curriculum_profile": self._current_curriculum_profile,
        }

    @staticmethod
    def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
        try:
            denom = float(denominator)
            if denom <= 0.0:
                return 0.0
            return float(numerator) / denom
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    def _build_combat_reset_debug(self) -> Dict[str, Any]:
        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            game_state = {}
        player = self._state_player(self.current_state)
        deck = self._state_deck(self.current_state)
        relics = self._state_named_list(self.current_state, "relics")
        potions = self._state_named_list(self.current_state, "potions")
        combat_state = game_state.get("combat_state", {})
        if not isinstance(combat_state, dict):
            combat_state = {}

        current_hp = self._current_player_hp(game_state)
        max_hp = self._current_player_max_hp(game_state)
        if current_hp is None:
            current_hp = self._safe_int(player.get("hp", player.get("current_hp")), 0)
        if max_hp is None:
            max_hp = self._safe_int(player.get("max_hp"), 0)

        return {
            "event": "combat_reset",
            "worker_id": self.worker_id,
            "episode": self._episode_index,
            "encounter_id": self._current_combat_encounter,
            "encounter_pool": self._current_combat_enemy_pool,
            "deck_mode": self._current_deck_mode,
            "deck_source": self._deck_source(),
            "curriculum_profile": self._current_curriculum_profile,
            "deck_size": self._safe_int(
                player.get("deck_size", game_state.get("deck_size")),
                len(deck),
            ),
            "deck": [self._deck_card_debug(card, slot) for slot, card in enumerate(deck)],
            "hand": self._hand_debug(self.current_state),
            "added_cards": self._current_deck_spec.get("added_cards", []),
            "removed_cards": self._current_deck_spec.get("removed_cards", []),
            "upgraded_cards": self._current_deck_spec.get("upgraded_cards", []),
            "floor_bucket": self._current_deck_spec.get("floor_bucket"),
            "synthetic_floor": self._current_deck_spec.get("synthetic_floor"),
            "deck_generator_settings": self._current_deck_spec.get("generator_settings", {}),
            "draw_pile_size": self._pile_size(self.current_state, "draw_pile"),
            "discard_pile_size": self._pile_size(self.current_state, "discard_pile"),
            "exhaust_pile_size": self._pile_size(self.current_state, "exhaust_pile"),
            "relics": [self._named_object_debug(item, slot) for slot, item in enumerate(relics)],
            "potions": [self._named_object_debug(item, slot) for slot, item in enumerate(potions)],
            "current_hp": current_hp,
            "max_hp": max_hp,
            "hand_size": len(self._state_hand(self.current_state)),
            "monster_count": len(self._state_enemies(self.current_state)),
            "turn": self._state_turn(self.current_state),
            "timestamp_monotonic": time.monotonic(),
        }

    def _deck_source(self) -> str:
        source = self._current_deck_spec.get("source")
        if source:
            return str(source)
        return "sts2_start_run_default"

    def _maybe_log_combat_reset_debug(self) -> None:
        if self.sts2_debug_episodes <= 0 or self._episode_index > self.sts2_debug_episodes:
            return
        reset = self._last_combat_reset_debug
        print(
            "[STS2 COMBAT RESET] "
            f"worker={self.worker_id} episode={self._episode_index} "
            f"encounter={reset.get('encounter_id')} deck_mode={reset.get('deck_mode')} "
            f"profile={reset.get('curriculum_profile')} "
            f"deck_source={reset.get('deck_source')} deck_size={reset.get('deck_size')} "
            f"hp={reset.get('current_hp')}/{reset.get('max_hp')} "
            f"added={[card.get('name', card.get('id')) for card in reset.get('added_cards', [])]} "
            f"removed={[card.get('name', card.get('id')) for card in reset.get('removed_cards', [])]} "
            f"upgraded={[card.get('name', card.get('id')) for card in reset.get('upgraded_cards', [])]} "
            f"bucket={reset.get('floor_bucket')} "
            f"draw={reset.get('draw_pile_size')} discard={reset.get('discard_pile_size')} "
            f"exhaust={reset.get('exhaust_pile_size')} "
            f"deck={[card.get('name', card.get('id')) for card in reset.get('deck', [])]} "
            f"relics={[item.get('name', item.get('id')) for item in reset.get('relics', [])]} "
            f"potions={[item.get('name', item.get('id')) for item in reset.get('potions', [])]}",
            file=sys.stderr,
        )
        self._append_debug_jsonl(reset)

    def _maybe_log_debug_episode(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.sts2_debug_episodes <= 0 or self._episode_index > self.sts2_debug_episodes:
            return
        game_state = self.current_state.get("game_state", {})
        if not isinstance(game_state, dict):
            game_state = {}
        combat_state = game_state.get("combat_state", {})
        if not isinstance(combat_state, dict):
            combat_state = {}
        player = combat_state.get("player", {}) if isinstance(combat_state, dict) else {}
        monsters = combat_state.get("monsters", []) if isinstance(combat_state, dict) else []
        hand = combat_state.get("hand", []) if isinstance(combat_state, dict) else []
        print(
            "[STS2 COMBAT DEBUG] "
            f"worker={self.worker_id} episode={self._episode_index} "
            f"encounter={self._current_combat_encounter} step={self.combat_step_count} "
            f"hp={player.get('current_hp', game_state.get('current_hp'))} "
            f"block={player.get('block')} energy={player.get('energy')} "
            f"hand={[card.get('name', card.get('id')) for card in hand if isinstance(card, dict)]} "
            f"monsters={self._monster_debug(monsters)} action_id={self._last_action_id} "
            f"action={self._last_action_summary} "
            f"command={self._last_action_command} reward={reward:.3f} "
            f"parts={self._last_reward_parts} done={self._last_done_reason or 'ongoing'} "
            f"terminated={terminated} truncated={truncated}",
            file=sys.stderr,
        )
        self._append_debug_jsonl(
            {
                "event": "combat_step",
                "worker_id": self.worker_id,
                "episode": self._episode_index,
                "encounter_id": self._current_combat_encounter,
                "step": self.combat_step_count,
                "turn": self._state_turn(self.current_state),
                "hp": player.get("current_hp", game_state.get("current_hp")),
                "block": player.get("block"),
                "energy": player.get("energy"),
                "hand": self._hand_debug(self.current_state),
                "monsters": self._monster_debug(monsters),
                "action_id": self._last_action_id,
                "action": self._last_action_summary,
                "command": self._last_action_command,
                "reward": float(reward),
                "reward_parts": dict(self._last_reward_parts),
                "done_reason": self._last_done_reason or "ongoing",
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "timestamp_monotonic": time.monotonic(),
            }
        )

    def _append_debug_jsonl(self, payload: Dict[str, Any]) -> None:
        if not self.sts2_debug_jsonl_path or self._debug_jsonl_failed:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self.sts2_debug_jsonl_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.sts2_debug_jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str))
                handle.write("\n")
        except Exception as exc:
            self._debug_jsonl_failed = True
            print(f"[STS2 DEBUG] Could not write JSONL debug event: {exc}", file=sys.stderr)

    @staticmethod
    def _monster_debug(monsters: Any) -> List[Dict[str, Any]]:
        if not isinstance(monsters, list):
            return []
        result: List[Dict[str, Any]] = []
        for monster in monsters:
            if not isinstance(monster, dict):
                continue
            result.append(
                {
                    "hp": monster.get("current_hp", monster.get("hp")),
                    "max_hp": monster.get("max_hp"),
                    "intent": monster.get("intent"),
                    "damage": monster.get("move_adjusted_damage"),
                }
            )
        return result

    def close(self) -> None:
        """No-op — CommunicationMod manages our lifecycle."""
        self.process_manager.stop()
