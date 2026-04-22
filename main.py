"""
Phase 1: Blind & Deaf Random Agent for Slay the Spire.

This script is launched BY CommunicationMod (configured in config.properties).
It communicates via stdin (JSON game state) and stdout (plain-text commands).

Usage:
  1. Set CommunicationMod config.properties:
     command=py C:\\dev\\sts rl bot\\main.py
  2. Launch Slay the Spire normally (via Steam with mods enabled).
  3. CommunicationMod will automatically launch this script.
"""

import sys
import random
import time
import numpy as np
from typing import List

from env import SlayTheSpireEnv


def main() -> None:
    print("Phase 2: Encoded Random Agent starting...", file=sys.stderr)

    env = SlayTheSpireEnv()

    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"FATAL: Failed to reset environment: {e}", file=sys.stderr)
        return

    terminated = False
    truncated = False
    step_count = 0

    print("Random agent loop started.", file=sys.stderr)

    while not terminated and not truncated:
        action_mask = info.get("action_mask", np.zeros(100, dtype=np.int8))
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) > 0:
            action = int(np.random.choice(valid_actions))
        else:
            action = 98 # Fallback to STATE
            
        action_str = env.action_mapper.get_action_string(action)
        
        # Debug parsing
        raw_state = info.get("raw_state", {})
        game_state = raw_state.get("game_state", {}) if isinstance(raw_state, dict) else {}
        combat_state = game_state.get("combat_state", {}) if isinstance(game_state, dict) else {}
        player = combat_state.get("player", {}) if isinstance(combat_state, dict) else {}
        energy = player.get("energy", 0) if isinstance(player, dict) else 0
        hand = combat_state.get("hand", []) if isinstance(combat_state, dict) else []
        
        print(f"[Step {step_count}] Action: {action} ({action_str})", file=sys.stderr)
        print(f"DEBUG: Wybrano akcję {action_str}, Energia: {energy}, Karty: {len(hand)}", file=sys.stderr)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if step_count % 50 == 0:
            raw_state = info.get("raw_state", {})
            in_game = raw_state.get("in_game", False)
            gs = raw_state.get("game_state", {})
            if isinstance(gs, dict):
                hp = gs.get("current_hp", "?")
                floor_num = gs.get("floor", "?")
                print(
                    f"[Step {step_count}] in_game={in_game}, HP={hp}, floor={floor_num}",
                    file=sys.stderr,
                )

    print(f"Episode ended after {step_count} steps.", file=sys.stderr)
    env.close()


if __name__ == "__main__":
    main()
