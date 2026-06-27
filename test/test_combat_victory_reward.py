import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import (
    COMBAT_VICTORY_REWARD,
    SlayTheSpireEnv,
)


class FakeProcessManager:
    def __init__(self, states=None):
        self.states = list(states or [])
        self.sent = []

    def read_state(self):
        if not self.states:
            raise RuntimeError("No fake states left to read")
        return self.states.pop(0)

    def send_command(self, command):
        self.sent.append(command)

    def stop(self):
        pass


def _deck(size=10, upgrades=0):
    return [{"id": f"Card{i}", "upgrades": upgrades} for i in range(size)]


def _relic(relic_id):
    return {"id": relic_id, "name": relic_id}


def _combat_state(floor=1):
    return {
        "in_game": True,
        "available_commands": ["play", "end", "state"],
        "game_state": {
            "screen_type": "NONE",
            "floor": floor,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "relics": [_relic("Burning Blood")],
            "deck": _deck(),
            "combat_state": {
                "player": {"current_hp": 80, "max_hp": 80, "energy": 3},
                "hand": [],
                "monsters": [
                    {
                        "current_hp": 40,
                        "max_hp": 40,
                        "is_gone": False,
                        "half_dead": False,
                    }
                ],
            },
        },
    }


def _screen_state(screen_type, floor=1):
    return {
        "in_game": True,
        "available_commands": ["proceed", "state"],
        "game_state": {
            "screen_type": screen_type,
            "floor": floor,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "relics": [_relic("Burning Blood")],
            "deck": _deck(),
        },
    }


def _env_with_state(state):
    env = SlayTheSpireEnv()
    env.current_state = state
    env.process_manager = FakeProcessManager()
    return env


def _seed_tracking(env, floor=1):
    env.last_player_hp = 80
    env.last_max_hp = 80
    env.last_monster_total_hp = None
    env.last_floor = floor
    env.last_screen_type = "NONE"
    env.last_relic_ids = {"Burning Blood"}
    env.last_deck_size = 10
    env.last_upgraded_cards = 0
    env.last_act = 1
    env.last_in_combat = False
    env.terminal_reward_given = False
    env.combat_victory_reward_given = False


def test_combat_victory_one_shot_loop():
    """Verify that transitioning to COMBAT_REWARD gives victory reward only once, and returning does not farm it."""
    # 1. Start in combat (NONE screen)
    state = _combat_state(floor=1)
    env = _env_with_state(state)
    _seed_tracking(env, floor=1)
    env.last_in_combat = True
    env.last_monster_total_hp = 40

    # Ensure flag starts as False
    assert env.combat_victory_reward_given is False

    # Calculate reward in combat - should be normal (no victory reward)
    reward_in_combat = env._calculate_reward()
    assert reward_in_combat == pytest.approx(0.0)
    assert env.combat_victory_reward_given is False

    # 2. Transition to COMBAT_REWARD screen (victory)
    env.current_state = _screen_state("COMBAT_REWARD", floor=1)
    reward_victory = env._calculate_reward()
    
    # Victory reward should be awarded (+2.0) + monster HP progress reward (+0.4) and flag set to True
    assert reward_victory == pytest.approx(COMBAT_VICTORY_REWARD + 0.4)
    assert env.combat_victory_reward_given is True

    # 3. Transition to CARD_REWARD screen
    env.current_state = _screen_state("CARD_REWARD", floor=1)
    reward_card = env._calculate_reward()
    assert reward_card == pytest.approx(0.0)
    assert env.combat_victory_reward_given is True

    # 4. Transition back to COMBAT_REWARD screen (looping)
    env.current_state = _screen_state("COMBAT_REWARD", floor=1)
    reward_victory_again = env._calculate_reward()
    
    # Repeated reward should be 0.0, because flag is True!
    assert reward_victory_again == pytest.approx(0.0)
    assert env.combat_victory_reward_given is True


def test_combat_victory_reset_on_new_combat_or_floor():
    """Verify that entering combat or changing floor resets the combat_victory_reward_given flag."""
    env = SlayTheSpireEnv()
    _seed_tracking(env, floor=1)
    
    # Manually mark as given
    env.combat_victory_reward_given = True

    # 1. Test reset on entering combat
    env.current_state = _combat_state(floor=1)
    # _calculate_reward checks in_combat and should reset the flag
    env._calculate_reward()
    assert env.combat_victory_reward_given is False

    # Set it back to True to test floor change
    env.combat_victory_reward_given = True

    # 2. Test reset on floor progress
    # Move to floor 2, not in combat
    env.current_state = _screen_state("MAP", floor=2)
    env._calculate_reward()
    assert env.combat_victory_reward_given is False
