"""
Tests for GRID screen card selection bug fix.

These tests use REAL JSON payloads captured from CommunicationMod logs
during a Living Wall event (card removal). The bug caused the bot to
enter an infinite select→cancel loop because RETURN/cancel was unmasked
alongside CONFIRM after card selection.

Log source: 2026-04-23 21:33:34 — Living Wall event on floor 2, Ironclad run.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from action_space import ActionMapper, ActionMasker


# ---------------------------------------------------------------------------
# Fixtures: real JSON states captured from CommunicationMod logs
# ---------------------------------------------------------------------------

def _make_card(name, card_id, card_type, cost, uuid, has_target=False):
    """Helper to build a card dict matching CommunicationMod format."""
    return {
        "exhausts": False, "cost": cost, "name": name, "id": card_id,
        "type": card_type, "ethereal": False, "uuid": uuid,
        "upgrades": 0, "rarity": "BASIC", "has_target": has_target,
    }

# 11-card deck from the log
_DECK_CARDS = [
    _make_card("Strike", "Strike_R", "ATTACK", 1, "fa12a520-c833-430f-bf8d-4ee48a038c0b", True),
    _make_card("Strike", "Strike_R", "ATTACK", 1, "3752547c-dc36-4dfc-8a61-c87889b9dcb3", True),
    _make_card("Strike", "Strike_R", "ATTACK", 1, "016898b6-52f5-4f93-a033-9de7b5b3f1ee", True),
    _make_card("Strike", "Strike_R", "ATTACK", 1, "e9f43d43-4609-4df8-993e-858dd082ca37", True),
    _make_card("Strike", "Strike_R", "ATTACK", 1, "ef053db1-f616-46d3-8657-fa99bed9a309", True),
    _make_card("Defend", "Defend_R", "SKILL", 1, "17856607-d9eb-4a74-bbc5-30a0aa2dee42", False),
    _make_card("Defend", "Defend_R", "SKILL", 1, "a3831fd1-3cfc-4da7-b10f-afa6b26a2b1f", False),
    _make_card("Defend", "Defend_R", "SKILL", 1, "ff657b7d-41fa-4440-bff2-04e660317ae0", False),
    _make_card("Defend", "Defend_R", "SKILL", 1, "3cdd77d9-d2f7-4245-826c-9aed4cb66321", False),
    _make_card("Bash", "Bash", "ATTACK", 2, "1264ab55-6124-424d-8156-824536d77727", True),
    _make_card("Wild Strike", "Wild Strike", "ATTACK", 1, "32019b48-7185-48df-993a-8aee30eb8ac4", True),
]


def _grid_state_before_selection():
    """
    Real state: GRID screen BEFORE any card is selected.
    From log line 21:33:34.056 — "choose" IS in available_commands.
    confirm_up=false, selected_cards=[], num_cards=1, for_purge=true.
    """
    return {
        "available_commands": ["choose", "key", "click", "wait", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "choice_list": [
                "strike", "strike", "strike", "strike", "strike",
                "defend", "defend", "defend", "defend",
                "bash", "wild strike"
            ],
            "screen_type": "GRID",
            "screen_state": {
                "cards": list(_DECK_CARDS),
                "selected_cards": [],
                "for_transform": False,
                "confirm_up": False,
                "any_number": False,
                "for_upgrade": False,
                "num_cards": 1,
                "for_purge": True,
            },
            "current_hp": 82,
            "max_hp": 88,
            "gold": 119,
            "act": 1,
            "floor": 2,
            "screen_name": "GRID",
            "room_phase": "EVENT",
            "is_screen_up": True,
            "action_phase": "WAITING_ON_USER",
        }
    }


def _grid_state_after_selection():
    """
    Real state: GRID screen AFTER a card is selected (CHOOSE 3 sent).
    From log line 21:33:34.067 — "confirm" and "cancel" in available_commands.
    confirm_up=true, selected_cards=[] (always empty per CommunicationMod bug).
    """
    return {
        "available_commands": ["confirm", "cancel", "key", "click", "wait", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": "GRID",
            "screen_state": {
                "cards": list(_DECK_CARDS),
                "selected_cards": [],  # Always empty — CommunicationMod quirk
                "for_transform": False,
                "confirm_up": True,
                "any_number": False,
                "for_upgrade": False,
                "num_cards": 1,
                "for_purge": True,
            },
            "current_hp": 82,
            "max_hp": 88,
            "gold": 119,
            "act": 1,
            "floor": 2,
            "screen_name": "GRID",
            "room_phase": "EVENT",
            "is_screen_up": True,
            "action_phase": "WAITING_ON_USER",
        }
    }


def _event_state_living_wall():
    """
    Real state: Living Wall EVENT screen (before entering GRID).
    From log line 21:33:34.044.
    """
    return {
        "available_commands": ["choose", "key", "click", "wait", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "choice_list": ["forget", "change", "grow"],
            "screen_type": "EVENT",
            "screen_state": {
                "event_id": "Living Wall",
                "options": [
                    {"choice_index": 0, "disabled": False, "text": "[Forget] Remove a card from your deck.", "label": "Forget"},
                    {"choice_index": 1, "disabled": False, "text": "[Change] Transform a card in your deck.", "label": "Change"},
                    {"choice_index": 2, "disabled": False, "text": "[Grow] Upgrade a card in your deck.", "label": "Grow"},
                ],
            },
            "current_hp": 82,
            "max_hp": 88,
            "gold": 119,
            "act": 1,
            "floor": 2,
        }
    }


# ===========================================================================
# ActionMasker Tests — GRID screen
# ===========================================================================

class TestGridMasker:
    """Tests for ActionMasker on GRID screen states from real logs."""

    def setup_method(self):
        self.masker = ActionMasker()

    # ---- Before card selection ----

    def test_before_selection_choose_enabled(self):
        """Before selecting a card, CHOOSE actions for all 11 cards must be enabled."""
        state = _grid_state_before_selection()
        mask = self.masker.get_mask(state)

        for i in range(11):
            assert mask[68 + i] == 1, f"CHOOSE {i} (index {68+i}) should be enabled before selection"

    def test_before_selection_choose_out_of_bounds_disabled(self):
        """CHOOSE indices beyond the card count must be disabled."""
        state = _grid_state_before_selection()
        mask = self.masker.get_mask(state)

        for i in range(11, 30):
            assert mask[68 + i] == 0, f"CHOOSE {i} (index {68+i}) should be disabled — only 11 cards exist"

    def test_before_selection_confirm_disabled(self):
        """CONFIRM (index 66) must NOT be available before a card is selected."""
        state = _grid_state_before_selection()
        mask = self.masker.get_mask(state)

        assert mask[66] == 0, "CONFIRM should be masked out when confirm_up=false"

    def test_before_selection_return_disabled(self):
        """RETURN/cancel (index 67) must be blocked on GRID to prevent the select→cancel loop."""
        state = _grid_state_before_selection()
        mask = self.masker.get_mask(state)

        assert mask[67] == 0, "RETURN must be masked out on GRID screen to prevent infinite loop"

    def test_before_selection_no_combat_actions(self):
        """Combat actions (PLAY, END, POTION) must be disabled on a GRID screen."""
        state = _grid_state_before_selection()
        mask = self.masker.get_mask(state)

        # PLAY untargeted (0-9)
        for i in range(10):
            assert mask[i] == 0, f"PLAY {i} should be disabled on GRID"
        # PLAY targeted (10-59)
        for i in range(10, 60):
            assert mask[i] == 0, f"PLAY targeted {i} should be disabled on GRID"
        # POTION (60-64)
        for i in range(60, 65):
            assert mask[i] == 0, f"POTION {i} should be disabled on GRID"
        # END (65)
        assert mask[65] == 0, "END should be disabled on GRID"

    # ---- After card selection ----

    def test_after_selection_confirm_only_action(self):
        """
        THE BUG TEST: After card selection, CONFIRM must be the ONLY enabled action.
        
        This is the exact scenario that caused the infinite loop:
        the bot selected a card, then picked RETURN (cancel) because 
        both CONFIRM and RETURN were valid. This test ensures CONFIRM
        is the sole available action.
        """
        state = _grid_state_after_selection()
        mask = self.masker.get_mask(state)

        valid_actions = np.where(mask == 1)[0]
        assert list(valid_actions) == [66], (
            f"After selection, ONLY CONFIRM (66) should be valid. "
            f"Got valid actions: {list(valid_actions)}"
        )

    def test_after_selection_confirm_enabled(self):
        """CONFIRM (index 66) must be enabled when confirm_up=true."""
        state = _grid_state_after_selection()
        mask = self.masker.get_mask(state)

        assert mask[66] == 1, "CONFIRM must be enabled when confirm_up=true"

    def test_after_selection_return_disabled(self):
        """
        RETURN/cancel (index 67) must be blocked even though 'cancel' is in available_commands.
        
        This was the root cause of the bug: 'cancel' appeared in available_commands
        after selection, enabling RETURN. The untrained model picked RETURN over CONFIRM,
        cancelling the selection and restarting the loop.
        """
        state = _grid_state_after_selection()
        mask = self.masker.get_mask(state)

        assert mask[67] == 0, (
            "RETURN must be masked out on GRID even though 'cancel' is in available_commands. "
            "This was the root cause of the select→cancel infinite loop."
        )

    def test_after_selection_all_choose_disabled(self):
        """All CHOOSE actions must be disabled after selection to prevent card toggling."""
        state = _grid_state_after_selection()
        mask = self.masker.get_mask(state)

        for i in range(30):
            assert mask[68 + i] == 0, (
                f"CHOOSE {i} (index {68+i}) must be disabled after selection "
                f"to prevent the bot from toggling cards instead of confirming"
            )

    def test_after_selection_selected_cards_always_empty(self):
        """
        Sanity check: CommunicationMod reports selected_cards=[] even after selection.
        This confirms we CANNOT rely on selected_cards — only on confirm_up.
        """
        state = _grid_state_after_selection()
        screen_state = state["game_state"]["screen_state"]

        assert screen_state["selected_cards"] == [], (
            "CommunicationMod always reports selected_cards=[] — "
            "this test documents the quirk so future developers don't try to rely on it"
        )
        assert screen_state["confirm_up"] is True, (
            "confirm_up is the only reliable signal for selection completion"
        )


# ===========================================================================
# ActionMapper Tests — GRID screen
# ===========================================================================

class TestGridMapper:
    """Tests for ActionMapper on GRID screen states from real logs."""

    def setup_method(self):
        self.mapper = ActionMapper()

    def test_choose_sends_choose_command(self):
        """When 'choose' IS in available_commands, CHOOSE should send CHOOSE N."""
        state = _grid_state_before_selection()

        for i in range(11):
            action_str = self.mapper.get_action_string(68 + i, state)
            assert action_str == f"CHOOSE {i}", (
                f"Action {68+i} should produce 'CHOOSE {i}', got '{action_str}'"
            )

    def test_confirm_sends_confirm_command(self):
        """After selection, index 66 must send CONFIRM (not PROCEED)."""
        state = _grid_state_after_selection()
        action_str = self.mapper.get_action_string(66, state)

        assert action_str == "CONFIRM", (
            f"On GRID with 'confirm' in available_commands, "
            f"action 66 should produce 'CONFIRM', got '{action_str}'"
        )

    def test_confirm_fallback_on_grid_without_confirm_cmd(self):
        """If 'confirm' is somehow missing from available_commands on GRID, still send CONFIRM."""
        state = _grid_state_after_selection()
        # Simulate edge case: remove confirm from available_commands
        state["available_commands"] = ["key", "click", "wait", "state"]
        
        action_str = self.mapper.get_action_string(66, state)
        assert action_str == "CONFIRM", (
            f"GRID fallback should produce 'CONFIRM', got '{action_str}'"
        )

    def test_choose_fallback_click_when_choose_missing(self):
        """If 'choose' is missing from available_commands on GRID, fall back to CLICK."""
        state = _grid_state_before_selection()
        state["available_commands"] = ["key", "click", "wait", "state"]  # Remove "choose"
        
        action_str = self.mapper.get_action_string(68, state)  # CHOOSE 0
        assert action_str.startswith("CLICK Left"), (
            f"When 'choose' is missing on GRID, should fall back to CLICK, got '{action_str}'"
        )

    def test_click_coordinates_grid_layout(self):
        """Verify the CLICK coordinate math for the 5-column grid layout."""
        state = _grid_state_before_selection()
        state["available_commands"] = ["key", "click", "wait", "state"]  # Force CLICK path

        # Row 0, Col 0 → X=500, Y=400
        assert self.mapper.get_action_string(68, state) == "CLICK Left 500 400"
        # Row 0, Col 4 → X=500+4*230=1420, Y=400
        assert self.mapper.get_action_string(72, state) == "CLICK Left 1420 400"
        # Row 1, Col 0 → X=500, Y=750
        assert self.mapper.get_action_string(73, state) == "CLICK Left 500 750"
        # Row 2, Col 0 → X=500, Y=1100
        assert self.mapper.get_action_string(78, state) == "CLICK Left 500 1100"


# ===========================================================================
# Full Flow Tests — simulating the exact bug scenario from logs
# ===========================================================================

class TestGridFullFlow:
    """
    End-to-end tests simulating the exact sequence from the log.
    
    Log sequence (21:33:34):
    1. EVENT screen → bot CHOOSE 0 ("Forget")
    2. GRID screen (confirm_up=false) → bot should CHOOSE a card
    3. GRID screen (confirm_up=true) → bot MUST CONFIRM (not RETURN!)
    """

    def setup_method(self):
        self.masker = ActionMasker()
        self.mapper = ActionMapper()

    def test_event_before_grid(self):
        """EVENT screen: bot can choose from 3 options."""
        state = _event_state_living_wall()
        mask = self.masker.get_mask(state)

        assert mask[68] == 1, "CHOOSE 0 (Forget)"
        assert mask[69] == 1, "CHOOSE 1 (Change)"
        assert mask[70] == 1, "CHOOSE 2 (Grow)"
        assert mask[71] == 0, "CHOOSE 3 should not exist"
        
        action_str = self.mapper.get_action_string(68, state)
        assert action_str == "CHOOSE 0"

    def test_full_bug_scenario_no_return_after_selection(self):
        """
        THE FULL BUG REPRODUCTION:
        
        Simulates the exact sequence from the log that caused the infinite loop.
        After card selection on GRID, the bot MUST only have CONFIRM available.
        If RETURN is available, the bot will cancel and restart — infinite loop.
        """
        # Step 1: GRID appears, no card selected
        state_before = _grid_state_before_selection()
        mask_before = self.masker.get_mask(state_before)

        # Bot should be able to choose any of the 11 cards
        choose_actions = [i for i in range(68, 98) if mask_before[i] == 1]
        assert len(choose_actions) == 11, f"Should have 11 CHOOSE options, got {len(choose_actions)}"
        assert mask_before[66] == 0, "CONFIRM should NOT be available yet"
        assert mask_before[67] == 0, "RETURN should NOT be available on GRID"

        # Bot picks CHOOSE 3 (same as in the log)
        action_str = self.mapper.get_action_string(71, state_before)
        assert action_str == "CHOOSE 3", "Bot should send CHOOSE 3"

        # Step 2: Game responds with confirm_up=true, cancel in available_commands
        state_after = _grid_state_after_selection()
        mask_after = self.masker.get_mask(state_after)

        # THIS IS THE BUG CHECK: RETURN must NOT be available
        assert mask_after[67] == 0, (
            "BUG REPRODUCTION: RETURN was enabled after card selection, "
            "causing the bot to cancel instead of confirm. "
            "This test ensures the fix prevents the infinite loop."
        )

        # CONFIRM must be the only action
        assert mask_after[66] == 1, "CONFIRM must be available"
        
        # No CHOOSE should be available either
        for i in range(30):
            assert mask_after[68 + i] == 0, f"CHOOSE {i} must be disabled after selection"

        # Bot sends CONFIRM
        confirm_str = self.mapper.get_action_string(66, state_after)
        assert confirm_str == "CONFIRM", f"Bot should send CONFIRM, got '{confirm_str}'"

    def test_mask_has_at_least_one_valid_action_before_selection(self):
        """Safety: mask must always have at least one valid action to prevent PPO crash."""
        state = _grid_state_before_selection()
        mask = self.masker.get_mask(state)
        assert np.any(mask == 1), "Mask must have at least one valid action"

    def test_mask_has_at_least_one_valid_action_after_selection(self):
        """Safety: mask must always have at least one valid action to prevent PPO crash."""
        state = _grid_state_after_selection()
        mask = self.masker.get_mask(state)
        assert np.any(mask == 1), "Mask must have at least one valid action"


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestGridEdgeCases:
    """Edge cases for GRID screen handling."""

    def setup_method(self):
        self.masker = ActionMasker()

    def test_grid_upgrade_screen(self):
        """GRID for card upgrade (for_upgrade=true) should follow same masking rules."""
        state = _grid_state_before_selection()
        state["game_state"]["screen_state"]["for_purge"] = False
        state["game_state"]["screen_state"]["for_upgrade"] = True
        
        mask = self.masker.get_mask(state)
        assert mask[67] == 0, "RETURN must be blocked on upgrade GRID too"
        assert mask[66] == 0, "CONFIRM blocked before selection on upgrade GRID"
        assert any(mask[68 + i] == 1 for i in range(11)), "CHOOSE must be available"

    def test_grid_transform_screen(self):
        """GRID for card transform (for_transform=true) should follow same masking rules."""
        state = _grid_state_before_selection()
        state["game_state"]["screen_state"]["for_purge"] = False
        state["game_state"]["screen_state"]["for_transform"] = True
        
        mask = self.masker.get_mask(state)
        assert mask[67] == 0, "RETURN must be blocked on transform GRID too"

    def test_grid_empty_cards_list(self):
        """If cards list is somehow empty, mask should not crash."""
        state = _grid_state_before_selection()
        state["game_state"]["screen_state"]["cards"] = []
        
        mask = self.masker.get_mask(state)
        # No CHOOSE should be enabled (no cards)
        for i in range(30):
            assert mask[68 + i] == 0, f"CHOOSE {i} should be disabled with empty cards"

    def test_grid_confirm_up_with_empty_selected_cards(self):
        """
        Confirms the key design decision: confirm_up=true takes precedence
        even when selected_cards is empty (CommunicationMod quirk).
        """
        state = _grid_state_after_selection()
        assert state["game_state"]["screen_state"]["selected_cards"] == []
        assert state["game_state"]["screen_state"]["confirm_up"] is True
        
        mask = self.masker.get_mask(state)
        assert mask[66] == 1, "CONFIRM must work even with empty selected_cards"
        assert mask[67] == 0, "RETURN must be blocked regardless of selected_cards"
