"""Tests for SHOP_SCREEN action masking.

Covers the CommunicationMod known issue:
  "There is no feedback or state change if you attempt to take or buy
   a potion while your potion inventory is full."

When the bot sends CHOOSE for a potion it can't hold, CommunicationMod
silently no-ops and never sends a new state — causing a permanent deadlock.
The ActionMasker must prevent this by masking out unaffordable items and
potions when slots are full.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from action_space import ActionMasker


def _valid(mask):
    return set(np.where(mask == 1)[0])


# ─── Helpers ────────────────────────────────────────────────────────────────

def _shop_state(
    shop_cards=None,
    shop_relics=None,
    shop_potions=None,
    purge_available=False,
    purge_cost=100,
    gold=200,
    potions_inventory=None,
):
    """Build a minimal SHOP_SCREEN state dict."""
    if shop_cards is None:
        shop_cards = []
    if shop_relics is None:
        shop_relics = []
    if shop_potions is None:
        shop_potions = []
    if potions_inventory is None:
        potions_inventory = [
            {"id": "Potion Slot", "can_use": False},
            {"id": "Potion Slot", "can_use": False},
            {"id": "Potion Slot", "can_use": False},
        ]

    # CommunicationMod provides choice_list on SHOP_SCREEN matching the
    # concatenation: cards + relics + potions [+ "purge"].
    choice_list = []
    for c in shop_cards:
        choice_list.append(c.get("id", "card"))
    for r in shop_relics:
        choice_list.append(r.get("id", "relic"))
    for p in shop_potions:
        choice_list.append(p.get("id", "potion"))
    if purge_available:
        choice_list.append("purge")

    return {
        "available_commands": ["choose", "leave", "state"],
        "in_game": True,
        "game_state": {
            "screen_type": "SHOP_SCREEN",
            "current_hp": 50,
            "max_hp": 80,
            "gold": gold,
            "potions": potions_inventory,
            "choice_list": choice_list,
            "screen_state": {
                "cards": shop_cards,
                "relics": shop_relics,
                "potions": shop_potions,
                "purge_available": purge_available,
                "purge_cost": purge_cost,
            },
        },
    }


def _shop_card(card_id="Strike_R", price=75):
    return {"id": card_id, "name": "Strike", "price": price}


def _shop_relic(relic_id="Vajra", price=150):
    return {"id": relic_id, "name": "Vajra", "price": price}


def _shop_potion(potion_id="Fire Potion", price=50):
    return {"id": potion_id, "name": "Fire Potion", "price": price}


# ─── Tests: Potion slot masking ─────────────────────────────────────────────

def test_shop_full_potion_slots_masks_potion_choices():
    """Potions must be masked when all inventory slots are occupied."""
    state = _shop_state(
        shop_cards=[_shop_card(price=50)],
        shop_relics=[_shop_relic(price=50)],
        shop_potions=[_shop_potion(price=50)],
        gold=200,
        potions_inventory=[
            {"id": "Fire Potion", "can_use": True},
            {"id": "Block Potion", "can_use": True},
            {"id": "Strength Potion", "can_use": True},
        ],
    )

    mask = ActionMasker().get_mask(state)

    # card (idx 0) → mask[68] should be allowed (affordable)
    assert mask[68] == 1, "Affordable card should be enabled"
    # relic (idx 1) → mask[69] should be allowed (affordable)
    assert mask[69] == 1, "Affordable relic should be enabled"
    # potion (idx 2) → mask[70] should be MASKED (no empty slots)
    assert mask[70] == 0, "Potion with full slots must be masked"


def test_shop_empty_potion_slot_allows_potion_purchase():
    """When there's at least one empty slot, potions should be purchasable."""
    state = _shop_state(
        shop_potions=[_shop_potion(price=50)],
        gold=200,
        potions_inventory=[
            {"id": "Fire Potion", "can_use": True},
            {"id": "Potion Slot", "can_use": False},
            {"id": "Strength Potion", "can_use": True},
        ],
    )

    mask = ActionMasker().get_mask(state)

    # potion (idx 0) → mask[68] should be allowed
    assert mask[68] == 1, "Potion should be buyable with an empty slot"


def test_shop_all_potions_full_only_potion_items_can_still_leave():
    """If the shop only has potions and slots are full, LEAVE must remain valid."""
    state = _shop_state(
        shop_potions=[
            _shop_potion("Potion A", 50),
            _shop_potion("Potion B", 50),
        ],
        gold=200,
        potions_inventory=[
            {"id": "X", "can_use": True},
            {"id": "Y", "can_use": True},
            {"id": "Z", "can_use": True},
        ],
    )

    mask = ActionMasker().get_mask(state)
    valid = _valid(mask)

    # Both potions should be masked
    assert 68 not in valid, "Potion A should be masked"
    assert 69 not in valid, "Potion B should be masked"
    # LEAVE (mask[67]) must remain enabled
    assert 67 in valid, "LEAVE must be available to exit the shop"
    # Mask must not be all-zero
    assert valid, "Mask must not be empty"


# ─── Tests: Gold / affordability masking ────────────────────────────────────

def test_shop_masks_unaffordable_card():
    """Cards costing more than player gold should be masked."""
    state = _shop_state(
        shop_cards=[_shop_card(price=300)],
        gold=100,
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 0, "Card too expensive should be masked"


def test_shop_masks_unaffordable_relic():
    """Relics costing more than player gold should be masked."""
    state = _shop_state(
        shop_cards=[_shop_card(price=50)],
        shop_relics=[_shop_relic(price=999)],
        gold=100,
    )

    mask = ActionMasker().get_mask(state)

    # card (idx 0) → affordable
    assert mask[68] == 1, "Affordable card should be enabled"
    # relic (idx 1) → unaffordable
    assert mask[69] == 0, "Unaffordable relic should be masked"


def test_shop_masks_unaffordable_potion_even_with_empty_slot():
    """Potions too expensive should be masked even if inventory has space."""
    state = _shop_state(
        shop_potions=[_shop_potion(price=999)],
        gold=100,
        potions_inventory=[{"id": "Potion Slot", "can_use": False}],
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 0, "Unaffordable potion should be masked"


def test_shop_masks_unaffordable_purge():
    """Card removal (purge) costing more than gold should be masked."""
    state = _shop_state(
        purge_available=True,
        purge_cost=500,
        gold=100,
    )

    mask = ActionMasker().get_mask(state)

    # purge is at idx 0 (no cards/relics/potions) → mask[68]
    assert mask[68] == 0, "Unaffordable purge should be masked"


def test_shop_allows_affordable_purge():
    """Card removal within budget should be allowed."""
    state = _shop_state(
        purge_available=True,
        purge_cost=50,
        gold=100,
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 1, "Affordable purge should be enabled"


# ─── Tests: Combined scenarios ──────────────────────────────────────────────

def test_shop_mixed_affordable_and_unaffordable():
    """Verify correct per-item masking with mixed prices."""
    state = _shop_state(
        shop_cards=[
            _shop_card("Cheap Card", price=50),
            _shop_card("Expensive Card", price=500),
        ],
        shop_relics=[_shop_relic("Cheap Relic", price=80)],
        shop_potions=[_shop_potion("Cheap Potion", price=30)],
        purge_available=True,
        purge_cost=75,
        gold=100,
    )

    mask = ActionMasker().get_mask(state)

    # idx 0: cheap card (50) → affordable
    assert mask[68] == 1
    # idx 1: expensive card (500) → masked
    assert mask[69] == 0
    # idx 2: cheap relic (80) → affordable
    assert mask[70] == 1
    # idx 3: cheap potion (30), has empty slots → affordable
    assert mask[71] == 1
    # idx 4: purge (75) → affordable
    assert mask[72] == 1


def test_shop_no_items_only_leave_available():
    """Edge case: empty shop should still allow leaving."""
    state = _shop_state(gold=100)

    mask = ActionMasker().get_mask(state)
    valid = _valid(mask)

    # LEAVE should be available
    assert 67 in valid, "Must be able to leave empty shop"


def test_shop_zero_gold_masks_everything():
    """With 0 gold, all purchasable items should be masked."""
    state = _shop_state(
        shop_cards=[_shop_card(price=50)],
        shop_relics=[_shop_relic(price=100)],
        shop_potions=[_shop_potion(price=25)],
        purge_available=True,
        purge_cost=50,
        gold=0,
    )

    mask = ActionMasker().get_mask(state)

    # All 4 items should be masked
    assert mask[68] == 0, "Card should be masked at 0 gold"
    assert mask[69] == 0, "Relic should be masked at 0 gold"
    assert mask[70] == 0, "Potion should be masked at 0 gold"
    assert mask[71] == 0, "Purge should be masked at 0 gold"
    # But LEAVE must remain
    assert mask[67] == 1


def test_shop_exact_gold_for_item():
    """Items costing exactly the player's gold should be allowed."""
    state = _shop_state(
        shop_cards=[_shop_card(price=100)],
        gold=100,
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 1, "Item at exact gold should be buyable"


def test_shop_potion_both_unaffordable_and_full_slots():
    """Potion that is both unaffordable AND slots full — must be masked."""
    state = _shop_state(
        shop_potions=[_shop_potion(price=999)],
        gold=50,
        potions_inventory=[
            {"id": "X", "can_use": True},
            {"id": "Y", "can_use": True},
        ],
    )

    mask = ActionMasker().get_mask(state)

    assert mask[68] == 0, "Potion must be double-masked"
