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
from typing import List

from env import SlayTheSpireEnv


def pick_random_action(state: dict) -> str:
    """Pick a random valid action from available_commands in the game state.

    CommunicationMod provides available_commands as a list of command names
    like ["play", "end", "potion", "proceed", "choose", ...].
    We need to build a full command string.
    """
    available = state.get("available_commands", [])
    if not isinstance(available, list) or not available:
        return "STATE"

    command = random.choice(available)
    game_state = state.get("game_state", {})
    if not isinstance(game_state, dict):
        game_state = {}

    combat_state = game_state.get("combat_state")

    if command == "play":
        # Play a random card, pick a random target if needed
        hand = []
        if combat_state and isinstance(combat_state, dict):
            hand = combat_state.get("hand", [])
        if hand:
            card_index = random.randint(1, len(hand))  # 1-indexed
            card = hand[card_index - 1]
            if card.get("has_target", False):
                monsters = combat_state.get("monsters", [])
                alive = [i for i, m in enumerate(monsters) if not m.get("is_gone", True) and not m.get("half_dead", False)]
                if alive:
                    target = random.choice(alive)
                    return f"PLAY {card_index} {target}"
                else:
                    return f"PLAY {card_index} 0"
            else:
                return f"PLAY {card_index}"
        else:
            return "END"

    elif command == "end":
        return "END"

    elif command == "proceed":
        return "PROCEED"

    elif command == "choose":
        choices = game_state.get("choice_list", [])
        if choices:
            choice_index = random.randint(0, len(choices) - 1)
            return f"CHOOSE {choice_index}"
        else:
            # Sometimes choices are available but choice_list empty — try index 0
            return "CHOOSE 0"

    elif command == "return":
        return "RETURN"

    elif command == "potion":
        potions = game_state.get("potions", [])
        usable = [(i, p) for i, p in enumerate(potions) if p.get("can_use", False)]
        if usable:
            idx, potion = random.choice(usable)
            if potion.get("requires_target", False) and combat_state:
                monsters = combat_state.get("monsters", [])
                alive = [i for i, m in enumerate(monsters) if not m.get("is_gone", True)]
                target = random.choice(alive) if alive else 0
                return f"POTION Use {idx} {target}"
            else:
                return f"POTION Use {idx}"
        else:
            discardable = [(i, p) for i, p in enumerate(potions) if p.get("can_discard", False)]
            if discardable:
                idx, _ = random.choice(discardable)
                return f"POTION Discard {idx}"
            return "PROCEED"

    elif command == "state":
        return "STATE"

    elif command == "wait":
        return "WAIT 100"

    else:
        return command.upper()


def main() -> None:
    print("Phase 1: Blind & Deaf Random Agent starting...", file=sys.stderr)

    env = SlayTheSpireEnv()

    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"FATAL: Failed to reset environment: {e}", file=sys.stderr)
        return

    raw_state = info.get("raw_state", {})
    terminated = False
    truncated = False
    step_count = 0

    print("Random agent loop started.", file=sys.stderr)

    while not terminated and not truncated:
        action = pick_random_action(raw_state)
        print(f"[Step {step_count}] Action: {action}", file=sys.stderr)

        obs, reward, terminated, truncated, info = env.step(action)
        raw_state = info.get("raw_state", {})
        step_count += 1

        if step_count % 50 == 0:
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
