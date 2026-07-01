"""Discrete action mapping for the sts2-cli JSON protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np


STS2_ACTION_SPACE_SIZE = 100
MAX_HAND_CARDS = 10
MAX_ENEMIES = 5
MAX_POTIONS = 5
MAX_CHOICES = 30

PLAY_CARD_BASE = 0
TARGETED_PLAY_BASE = 10
POTION_BASE = 60
END_TURN_ACTION = 65
PROCEED_ACTION = 66
BACK_ACTION = 67
CHOICE_BASE = 68
FALLBACK_ACTION = 98
QUIT_ACTION = 99


JsonCommand = dict[str, Any]

CARD_SELECTION_SOURCE_IDS = {
    "ARMAMENTS",
    "BODYGUARD",
    "BURNING_PACT",
    "HIDDEN_DAGGERS",
    "SCULPTING_STRIKE",
    "SEEKER_STRIKE",
    "SURVIVOR",
    "TRUE_GRIT",
}

CARD_SELECTION_DESCRIPTION_MARKERS = (
    "choose ",
    "select ",
    "discard ",
    "upgrade a card",
    "exhaust 1 card",
    "exhaust a card",
    "exhaust any number of cards",
    "to a card in your",
    "summon",
    "draw pile",
    "discard pile",
)


class StS2ActionMapper:
    """Map the shared 100-way discrete action space to sts2-cli JSON commands."""

    def __init__(self) -> None:
        self.action_space_size = STS2_ACTION_SPACE_SIZE

    def get_action_string(
        self,
        action_id: int,
        state: Optional[dict[str, Any]] = None,
    ) -> JsonCommand:
        if not 0 <= action_id < self.action_space_size:
            raise ValueError(f"Invalid action ID: {action_id}")

        commands = build_legal_commands(state or {})
        if action_id in commands:
            return commands[action_id]
        raise ValueError(f"Action {action_id} is not legal for current STS2 state.")


class StS2ActionMasker:
    """Build a legal-action mask from sts2-cli decision points."""

    def __init__(self) -> None:
        self.action_space_size = STS2_ACTION_SPACE_SIZE

    def get_mask(self, state: dict[str, Any]) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        for action_id in build_legal_commands(state):
            mask[action_id] = 1

        if not np.any(mask):
            mask[FALLBACK_ACTION] = 1

        return mask


def build_legal_commands(state: dict[str, Any]) -> dict[int, JsonCommand]:
    """Return all action ids currently accepted by sts2-cli."""
    decision = str(state.get("decision") or "").lower()
    commands: dict[int, JsonCommand] = {}

    if decision == "combat_play":
        _add_combat_actions(commands, state)
    elif decision == "map_select":
        _add_indexed_choices(commands, state.get("choices"), _map_choice_command)
    elif decision == "card_reward":
        _add_indexed_choices(commands, state.get("cards"), _card_reward_command)
        if bool(state.get("can_skip", True)):
            commands[BACK_ACTION] = _action("skip_card_reward")
    elif decision == "event_choice":
        _add_indexed_choices(
            commands,
            state.get("options"),
            _option_command,
            include_item=_is_unlocked_event_option,
        )
        commands.setdefault(BACK_ACTION, _action("leave_room"))
    elif decision == "rest_site":
        _add_indexed_choices(
            commands,
            state.get("options"),
            _option_command,
            include_item=_is_enabled_rest_option,
        )
    elif decision == "shop":
        _add_shop_actions(commands, state)
    elif decision == "card_select":
        _add_card_select_actions(commands, state)
    elif decision == "bundle_select":
        _add_indexed_choices(commands, state.get("bundles"), _bundle_command)
    elif decision == "game_over":
        commands[QUIT_ACTION] = {"cmd": "quit"}
    elif decision == "unknown":
        commands[PROCEED_ACTION] = _action("proceed")
        commands[BACK_ACTION] = _action("leave_room")
    elif state.get("type") == "error":
        commands[QUIT_ACTION] = {"cmd": "quit"}

    if not commands and state:
        commands[FALLBACK_ACTION] = _action("proceed")

    return commands


def _add_combat_actions(commands: dict[int, JsonCommand], state: dict[str, Any]) -> None:
    hand = _as_list(state.get("hand"))
    enemies = _as_list(state.get("enemies"))[:MAX_ENEMIES]

    for slot, card in enumerate(hand[:MAX_HAND_CARDS]):
        if not isinstance(card, Mapping):
            continue
        if not bool(card.get("can_play", True)):
            continue
        if _is_card_selection_source(card):
            continue

        card_index = _int(card.get("index"), slot)
        target_type = str(card.get("target_type") or "").lower()
        targets_single_enemy = "anyenemy" in target_type or "enemy" == target_type

        if targets_single_enemy:
            for target_index, _enemy in enumerate(enemies):
                action_id = TARGETED_PLAY_BASE + slot * MAX_ENEMIES + target_index
                commands[action_id] = _action(
                    "play_card",
                    {"card_index": card_index, "target_index": target_index},
                )
        else:
            commands[PLAY_CARD_BASE + slot] = _action(
                "play_card",
                {"card_index": card_index},
            )

    potions = _as_list(_player(state).get("potions"))[:MAX_POTIONS]
    for slot, potion in enumerate(potions):
        if not isinstance(potion, Mapping):
            continue
        if _is_card_selection_source(potion):
            continue
        target_type = str(potion.get("target_type") or "").lower()
        potion_index = _int(potion.get("index"), slot)
        args: dict[str, Any] = {"potion_index": potion_index}
        if "anyenemy" in target_type:
            if len(enemies) != 1:
                continue
            args["target_index"] = 0
        commands[POTION_BASE + slot] = _action("use_potion", args)

    commands[END_TURN_ACTION] = _action("end_turn")


def _add_shop_actions(commands: dict[int, JsonCommand], state: dict[str, Any]) -> None:
    gold = _int(_player(state).get("gold"), 0)
    next_slot = CHOICE_BASE

    for key, action_name, arg_name in (
        ("cards", "buy_card", "card_index"),
        ("relics", "buy_relic", "relic_index"),
        ("potions", "buy_potion", "potion_index"),
    ):
        for item in _as_list(state.get(key)):
            if next_slot >= CHOICE_BASE + MAX_CHOICES:
                break
            if not isinstance(item, Mapping):
                continue
            if item.get("is_stocked") is False:
                continue
            if _int(item.get("cost"), 0) > gold:
                continue
            index = _int(item.get("index"), next_slot - CHOICE_BASE)
            commands[next_slot] = _action(action_name, {arg_name: index})
            next_slot += 1

    removal_cost = state.get("card_removal_cost")
    if (
        next_slot < CHOICE_BASE + MAX_CHOICES
        and removal_cost is not None
        and _int(removal_cost, gold + 1) <= gold
    ):
        commands[next_slot] = _action("remove_card")

    commands[BACK_ACTION] = _action("leave_room")


def _add_card_select_actions(commands: dict[int, JsonCommand], state: dict[str, Any]) -> None:
    cards = _as_list(state.get("cards"))
    min_select = max(0, _int(state.get("min_select"), 1))
    max_select = max(min_select, _int(state.get("max_select"), min_select))
    target_select = _bounded_select_count(max_select if max_select > 0 else min_select, cards)
    if target_select <= 0 and cards:
        target_select = 1

    available_indices = [
        slot
        for slot, card in enumerate(cards[:MAX_CHOICES])
        if isinstance(card, Mapping)
    ]

    for slot, card in enumerate(cards[:MAX_CHOICES]):
        if not isinstance(card, Mapping):
            continue
        indices = [slot]
        for fallback_index in available_indices:
            if fallback_index not in indices:
                indices.append(fallback_index)
            if len(indices) >= target_select:
                break
        commands[CHOICE_BASE + slot] = _action(
            "select_cards",
            {"indices": ",".join(str(index) for index in indices[:target_select])},
        )

    if min_select == 0:
        commands[BACK_ACTION] = _action("skip_select")


def _add_indexed_choices(
    commands: dict[int, JsonCommand],
    raw_items: Any,
    command_builder: Any,
    include_item: Any | None = None,
) -> None:
    for slot, item in enumerate(_as_list(raw_items)[:MAX_CHOICES]):
        if include_item is not None and not include_item(item):
            continue
        commands[CHOICE_BASE + slot] = command_builder(item, slot)


def _map_choice_command(item: Any, slot: int) -> JsonCommand:
    if isinstance(item, Mapping):
        return _action(
            "select_map_node",
            {
                "col": _int(item.get("col"), _int(item.get("x"), slot)),
                "row": _int(item.get("row"), _int(item.get("y"), 0)),
            },
        )
    return _action("select_map_node", {"col": slot, "row": 0})


def _card_reward_command(item: Any, slot: int) -> JsonCommand:
    index = _int(item.get("index"), slot) if isinstance(item, Mapping) else slot
    return _action("select_card_reward", {"card_index": index})


def _option_command(item: Any, slot: int) -> JsonCommand:
    index = _int(item.get("index"), slot) if isinstance(item, Mapping) else slot
    return _action("choose_option", {"option_index": index})


def _bundle_command(item: Any, slot: int) -> JsonCommand:
    index = _int(item.get("index"), slot) if isinstance(item, Mapping) else slot
    return _action("select_bundle", {"bundle_index": index})


def _is_unlocked_event_option(item: Any) -> bool:
    return not isinstance(item, Mapping) or not bool(item.get("is_locked", False))


def _is_enabled_rest_option(item: Any) -> bool:
    return not isinstance(item, Mapping) or bool(item.get("is_enabled", True))


def _action(action: str, args: Optional[dict[str, Any]] = None) -> JsonCommand:
    command: JsonCommand = {"cmd": "action", "action": action}
    if args:
        command["args"] = args
    return command


def _player(state: dict[str, Any]) -> Mapping[str, Any]:
    player = state.get("player")
    return player if isinstance(player, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _is_card_selection_source(item: Mapping[str, Any]) -> bool:
    identifier = _normalized_identifier(item.get("id") or item.get("name"))
    if identifier in CARD_SELECTION_SOURCE_IDS:
        return True

    description = str(item.get("description") or "").lower()
    if not description:
        return False
    return any(marker in description for marker in CARD_SELECTION_DESCRIPTION_MARKERS)


def _normalized_identifier(value: Any) -> str:
    text = str(value or "").strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return "_".join(text.replace("-", " ").split())


def _bounded_select_count(value: int, cards: list[Any]) -> int:
    selectable_count = sum(1 for card in cards[:MAX_CHOICES] if isinstance(card, Mapping))
    if selectable_count <= 0:
        return 0
    return max(0, min(value, selectable_count))


def _int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
