import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from action_space import ActionMasker


def _valid(mask):
    return set(np.where(mask == 1)[0])


def _card(cost=1, is_playable=True, has_target=False, card_type="ATTACK"):
    return {
        "cost": cost,
        "is_playable": is_playable,
        "has_target": has_target,
        "type": card_type,
        "damage": 6 if card_type == "ATTACK" else 0,
        "block": 5 if card_type == "SKILL" else 0,
    }


def _combat_state(hand, monsters, energy=3, potions=None):
    return {
        "available_commands": ["play", "end", "potion", "state"],
        "in_game": True,
        "game_state": {
            "screen_type": "NONE",
            "current_hp": 50,
            "max_hp": 80,
            "potions": potions or [],
            "combat_state": {
                "player": {"current_hp": 50, "max_hp": 80, "energy": energy},
                "hand": hand,
                "monsters": monsters,
            },
        },
    }


def _alive_monster(hp=20):
    return {"current_hp": hp, "max_hp": hp, "is_gone": False, "half_dead": False}


def test_empty_hand_in_combat_keeps_end_turn_as_safe_action():
    state = _combat_state(hand=[], monsters=[_alive_monster()])

    mask = ActionMasker().get_mask(state)
    valid = _valid(mask)

    assert 65 in valid
    assert not any(0 <= action <= 59 for action in valid)
    assert valid


def test_targeted_card_disabled_when_no_valid_monster_targets_exist():
    monsters = [
        {"current_hp": 0, "max_hp": 20, "is_gone": False, "half_dead": False},
        {"current_hp": 10, "max_hp": 20, "is_gone": True, "half_dead": False},
        {"current_hp": 10, "max_hp": 20, "is_gone": False, "half_dead": True},
    ]
    state = _combat_state(hand=[_card(has_target=True)], monsters=monsters)

    mask = ActionMasker().get_mask(state)

    assert 65 in _valid(mask)
    assert not any(mask[10 + target] == 1 for target in range(5))


def test_x_cost_card_is_allowed_when_communicationmod_marks_it_playable():
    state = _combat_state(
        hand=[_card(cost=-1, is_playable=True, has_target=False, card_type="SKILL")],
        monsters=[_alive_monster()],
        energy=2,
    )

    mask = ActionMasker().get_mask(state)

    assert mask[0] == 1


def test_target_required_potion_is_masked_until_action_space_supports_targets():
    state = _combat_state(
        hand=[],
        monsters=[_alive_monster()],
        potions=[
            {"id": "Fire Potion", "can_use": True, "requires_target": True},
            {"id": "Potion Slot", "can_use": False, "requires_target": False},
        ],
    )

    mask = ActionMasker().get_mask(state)

    assert mask[60] == 0
    assert mask[65] == 1


def _combat_reward_state(rewards, potions):
    return {
        "available_commands": ["choose", "proceed", "state"],
        "in_game": True,
        "game_state": {
            "screen_type": "COMBAT_REWARD",
            "current_hp": 50,
            "max_hp": 80,
            "potions": potions,
            "screen_state": {"rewards": rewards},
        },
    }


def test_full_potion_slots_mask_potion_reward_but_keep_other_rewards():
    state = _combat_reward_state(
        rewards=[
            {"reward_type": "CARD"},
            {"reward_type": "POTION"},
            {"reward_type": "GOLD"},
        ],
        potions=[
            {"id": "Potion A"},
            {"id": "Potion B"},
            {"id": "Potion C"},
        ],
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 1
    assert mask[69] == 0
    assert mask[70] == 1
    assert mask[66] == 1


def test_full_potion_slots_with_only_potion_reward_can_still_proceed():
    state = _combat_reward_state(
        rewards=[{"reward_type": "POTION"}],
        potions=[{"id": "Potion A"}, {"id": "Potion B"}, {"id": "Potion C"}],
    )

    mask = ActionMasker().get_mask(state)
    valid = _valid(mask)

    assert 68 not in valid
    assert 66 in valid
    assert valid


def test_empty_potion_slot_allows_potion_reward_choice():
    state = _combat_reward_state(
        rewards=[{"reward_type": "POTION"}],
        potions=[{"id": "Potion Slot"}, {"id": "Potion B"}, {"id": "Potion C"}],
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 1


def test_map_choice_screen_blocks_return_to_prevent_reward_room_loops():
    state = {
        "available_commands": ["choose", "return", "state"],
        "in_game": True,
        "game_state": {
            "screen_type": "MAP",
            "current_hp": 50,
            "max_hp": 80,
            "screen_state": {"next_nodes": [{"x": 1, "y": 0}, {"x": 2, "y": 0}]},
        },
    }

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 1
    assert mask[69] == 1
    assert mask[67] == 0


def test_state_fallback_prevents_all_zero_mask_on_unknown_screen():
    state = {
        "available_commands": ["state"],
        "in_game": True,
        "game_state": {"screen_type": "UNKNOWN"},
    }

    mask = ActionMasker().get_mask(state)

    assert _valid(mask) == {98}


def test_empty_grid_selection_falls_back_to_state_instead_of_zero_mask():
    state = {
        "available_commands": ["choose", "key", "click", "wait", "state"],
        "in_game": True,
        "game_state": {
            "screen_type": "GRID",
            "screen_state": {
                "cards": [],
                "confirm_up": False,
                "selected_cards": [],
                "num_cards": 1,
            },
        },
    }

    mask = ActionMasker().get_mask(state)

    assert _valid(mask) == {98}

