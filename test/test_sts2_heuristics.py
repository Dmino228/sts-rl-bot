import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sts2.action_space import BACK_ACTION, CHOICE_BASE, StS2ActionMasker
from sts2.heuristics import StS2StrategicHeuristic
from sts2.state_encoder import normalize_sts2_state


def _decision(state):
    normalized = normalize_sts2_state(state)
    mask = StS2ActionMasker().get_mask(normalized)
    return StS2StrategicHeuristic().select_action(normalized, mask)


def test_sts2_heuristic_does_not_control_combat_play():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "combat_play",
            "hand": [],
            "enemies": [{"index": 0, "hp": 30, "max_hp": 30}],
            "player": {"hp": 70, "max_hp": 80},
        }
    )
    mask = StS2ActionMasker().get_mask(state)

    assert StS2StrategicHeuristic().select_action(state, mask) is None


def test_sts2_heuristic_prefers_safe_monster_over_low_hp_elite():
    decision = _decision(
        {
            "type": "decision",
            "decision": "map_select",
            "choices": [
                {"col": 0, "row": 1, "type": "Elite"},
                {"col": 1, "row": 1, "type": "Monster"},
                {"col": 2, "row": 1, "type": "Shop"},
            ],
            "player": {"hp": 25, "max_hp": 80, "gold": 40},
        }
    )

    assert decision is not None
    assert decision.action_id == CHOICE_BASE + 1
    assert decision.phase == "map_select"


def test_sts2_heuristic_picks_high_value_card_reward():
    decision = _decision(
        {
            "type": "decision",
            "decision": "card_reward",
            "cards": [
                {"index": 0, "name": "Wound", "type": "Curse"},
                {
                    "index": 1,
                    "name": "Big Bonk",
                    "type": "Attack",
                    "rarity": "Rare",
                    "stats": {"damage": 18},
                },
            ],
            "can_skip": True,
            "player": {"hp": 70, "max_hp": 80, "deck": []},
        }
    )

    assert decision is not None
    assert decision.action_id == CHOICE_BASE + 1


def test_sts2_heuristic_skips_bad_card_reward_when_allowed():
    decision = _decision(
        {
            "type": "decision",
            "decision": "card_reward",
            "cards": [{"index": 0, "name": "Regret", "type": "Curse"}],
            "can_skip": True,
            "player": {"hp": 70, "max_hp": 80, "deck": []},
        }
    )

    assert decision is not None
    assert decision.action_id == BACK_ACTION


def test_sts2_heuristic_rests_at_low_hp():
    decision = _decision(
        {
            "type": "decision",
            "decision": "rest_site",
            "options": [
                {"index": 0, "name": "smith"},
                {"index": 1, "name": "rest"},
            ],
            "player": {"hp": 20, "max_hp": 80},
        }
    )

    assert decision is not None
    assert decision.action_id == CHOICE_BASE + 1


def test_sts2_heuristic_hard_mask_keeps_single_action():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "map_select",
            "choices": [
                {"col": 0, "row": 1, "type": "Elite"},
                {"col": 1, "row": 1, "type": "Monster"},
            ],
            "player": {"hp": 25, "max_hp": 80},
        }
    )
    mask = StS2ActionMasker().get_mask(state).astype(np.float32)

    hard_mask, decision = StS2StrategicHeuristic().mask_for_mode(
        state,
        mask,
        mode="hard",
    )

    assert decision is not None
    assert np.flatnonzero(hard_mask).tolist() == [CHOICE_BASE + 1]
