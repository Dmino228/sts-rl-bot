"""State adapter and fixed-size encoder for Slay the Spire 2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from sts2.spire_codex import SpireCodex


SCREEN_TYPES = (
    "NONE",
    "MAP",
    "COMBAT_REWARD",
    "EVENT",
    "REST",
    "SHOP",
    "GRID",
    "GAME_OVER",
    "UNKNOWN",
)

SCREEN_BY_DECISION = {
    "combat_play": "NONE",
    "map_select": "MAP",
    "card_reward": "COMBAT_REWARD",
    "event_choice": "EVENT",
    "rest_site": "REST",
    "shop": "SHOP",
    "card_select": "GRID",
    "bundle_select": "GRID",
    "game_over": "GAME_OVER",
}

CARD_TYPE_IDS = {
    "attack": 1.0,
    "skill": 2.0,
    "power": 3.0,
    "curse": 4.0,
    "status": 5.0,
}

TARGET_TYPE_IDS = {
    "none": 0.0,
    "self": 1.0,
    "allenemies": 2.0,
    "anyenemy": 3.0,
    "ally": 4.0,
}

INTENT_IDS = {
    "attack": 1.0,
    "attackbuff": 2.0,
    "attackdebuff": 3.0,
    "attackdefend": 4.0,
    "buff": 5.0,
    "debuff": 6.0,
    "defend": 7.0,
    "escape": 8.0,
    "sleep": 9.0,
    "stun": 10.0,
    "unknown": 0.0,
}
class StS2StateEncoder:
    """Encode sts2-cli decision JSON into the shared flat observation shape."""

    def __init__(self, codex: SpireCodex | None = None) -> None:
        self.codex = codex or SpireCodex()
        
        self.legacy_size = 205
        self.card_count = self.codex.get_card_count()
        self.relic_count = self.codex.get_relic_count()
        self.potion_count = self.codex.get_potion_count()
        self.monster_count = self.codex.get_monster_count()
        
        self.hand_slots = 10
        self.potion_slots = 5
        self.monster_slots = 5
        
        self.card_slot_size = self.card_count + 5  # ID one-hot + 5 static stats
        self.monster_slot_size = self.monster_count + 4  # ID one-hot + 4 power stats
        
        size = (
            self.legacy_size
            + (self.hand_slots * self.card_slot_size)
            + self.relic_count
            + (self.potion_slots * self.potion_count)
            + (self.monster_slots * self.monster_slot_size)
        )
        self.shape = (size,)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32,
        )

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        obs = np.zeros(self.shape, dtype=np.float32)

        # 1. Override path for testing stubs
        raw_obs = state.get("observation")
        if raw_obs is None:
            raw_obs = state.get("game_state", {}).get("observation")
        if raw_obs is not None:
            values = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
            limit = min(values.size, obs.size)
            obs[:limit] = values[:limit]
            np.clip(obs, -1.0, 1.0, out=obs)
            return obs

        # Normalization of raw state
        normalized = normalize_sts2_state(state)
        game_state = _mapping(normalized.get("game_state"))
        player = _mapping(normalized.get("player"))
        combat_state = _mapping(game_state.get("combat_state"))

        # 2. Populate base 205 features exactly as in the legacy system
        screen_type = str(game_state.get("screen_type", "UNKNOWN")).upper()
        _set_one_hot(obs, 0, SCREEN_TYPES, screen_type)

        obs[9] = _number(game_state.get("act"), 1) / 4.0
        obs[10] = _number(game_state.get("floor"), 0) / 60.0
        obs[11] = _number(game_state.get("gold"), player.get("gold", 0)) / 1000.0

        current_hp = _number(game_state.get("current_hp"), player.get("hp", 0))
        max_hp = max(1.0, _number(game_state.get("max_hp"), player.get("max_hp", 1)))
        obs[12] = current_hp / max_hp
        obs[13] = current_hp / 120.0
        obs[14] = max_hp / 120.0
        obs[15] = _number(combat_state.get("player", {}).get("energy"), normalized.get("energy", 0)) / 10.0
        obs[16] = _number(player.get("block"), 0) / 100.0
        obs[17] = _number(player.get("deck_size"), len(_list(player.get("deck")))) / 80.0
        obs[18] = len(_list(player.get("relics"))) / 40.0
        obs[19] = len(_list(player.get("potions"))) / 5.0
        obs[20] = _number(normalized.get("round"), 0) / 20.0
        obs[21] = len(_list(normalized.get("choices"))) / 30.0
        obs[22] = len(_list(normalized.get("cards"))) / 30.0

        _encode_potions(obs, 24, _list(player.get("potions")))
        _encode_cards(obs, 40, _list(normalized.get("hand")))
        _encode_enemies(obs, 120, _list(normalized.get("enemies")))
        _encode_decision_items(obs, 170, normalized)

        # 3. Append Hand Cards Semantic Profiles
        hand_offset = self.legacy_size
        hand = _list(normalized.get("hand"))
        for slot in range(self.hand_slots):
            slot_start = hand_offset + slot * self.card_slot_size
            if slot < len(hand) and isinstance(hand[slot], Mapping):
                card = hand[slot]
                card_id = card.get("id") or card.get("name")
                card_idx = self.codex.get_card_index(card_id)
                if card_idx is not None:
                    # One-hot identity
                    obs[slot_start + card_idx] = 1.0
                    # Static metadata
                    meta = self.codex.get_card_static_metadata(card_idx)
                    obs[slot_start + self.card_count + 0] = meta["cost"] / 5.0
                    obs[slot_start + self.card_count + 1] = meta["is_x"]
                    obs[slot_start + self.card_count + 2] = meta["is_attack"]
                    obs[slot_start + self.card_count + 3] = meta["is_skill"]
                    obs[slot_start + self.card_count + 4] = meta["is_power"]

        # 4. Append Relics Semantic Profile
        relics_offset = hand_offset + self.hand_slots * self.card_slot_size
        relics = _list(player.get("relics")) or _list(game_state.get("relics"))
        for relic in relics:
            relic_id = relic.get("id") or relic.get("name") if isinstance(relic, Mapping) else relic
            relic_idx = self.codex.get_relic_index(relic_id)
            if relic_idx is not None:
                obs[relics_offset + relic_idx] = 1.0

        # 5. Append Potions Semantic Profiles
        potions_offset = relics_offset + self.relic_count
        potions = _list(player.get("potions"))
        for slot in range(self.potion_slots):
            slot_start = potions_offset + slot * self.potion_count
            if slot < len(potions) and isinstance(potions[slot], Mapping):
                potion = potions[slot]
                potion_id = potion.get("id") or potion.get("name")
                potion_idx = self.codex.get_potion_index(potion_id)
                if potion_idx is not None:
                    obs[slot_start + potion_idx] = 1.0

        # 6. Append Monsters Semantic & Power Profiles
        monsters_offset = potions_offset + self.potion_slots * self.potion_count
        enemies = _list(normalized.get("enemies"))
        for slot in range(self.monster_slots):
            slot_start = monsters_offset + slot * self.monster_slot_size
            if slot < len(enemies) and isinstance(enemies[slot], Mapping):
                enemy = enemies[slot]
                # One-hot identity
                monster_id = enemy.get("id") or enemy.get("name")
                monster_idx = self.codex.get_monster_index(monster_id)
                if monster_idx is not None:
                    obs[slot_start + monster_idx] = 1.0
                
                # Active powers: Strength, Vulnerable, Weak, Ritual
                powers = _list(enemy.get("powers"))
                for power in powers:
                    if not isinstance(power, Mapping):
                        continue
                    p_name = str(power.get("name") or "").strip().lower()
                    p_amount = _number(power.get("amount"), 0)
                    if p_name == "strength":
                        obs[slot_start + self.monster_count + 0] = p_amount / 10.0
                    elif p_name == "vulnerable":
                        obs[slot_start + self.monster_count + 1] = p_amount / 5.0
                    elif p_name == "weak":
                        obs[slot_start + self.monster_count + 2] = p_amount / 5.0
                    elif p_name == "ritual":
                        obs[slot_start + self.monster_count + 3] = p_amount / 10.0

        np.clip(obs, -1.0, 1.0, out=obs)
        return obs




def normalize_sts2_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Adapt a raw sts2-cli response to the env's legacy-friendly state shape."""
    if raw_state.get("_game_version") == "sts2":
        return raw_state

    state = dict(raw_state)
    decision = str(state.get("decision") or "").lower()
    player = _mapping(state.get("player"))
    context = _mapping(state.get("context"))

    game_state = dict(_mapping(state.get("game_state")))
    screen_type = SCREEN_BY_DECISION.get(decision, "UNKNOWN")
    if state.get("type") == "error":
        screen_type = "GAME_OVER"

    game_state.setdefault("screen_type", screen_type)
    game_state.setdefault("act", state.get("act", context.get("act", 1)))
    game_state.setdefault("floor", state.get("floor", context.get("floor", 0)))
    game_state.setdefault("current_hp", player.get("hp", 0))
    game_state.setdefault("max_hp", player.get("max_hp", 1))
    game_state.setdefault("gold", player.get("gold", 0))
    game_state.setdefault("relics", _list(player.get("relics")))
    game_state.setdefault("deck", [_adapt_deck_card(card) for card in _list(player.get("deck"))])
    game_state.setdefault("potions", _list(player.get("potions")))

    if decision == "combat_play":
        game_state["combat_state"] = _combat_state(state, player)
    elif decision == "map_select":
        game_state["screen_state"] = {
            "next_nodes": [
                {"x": choice.get("col", 0), "y": choice.get("row", 0), **dict(choice)}
                for choice in _list_of_mappings(state.get("choices"))
            ]
        }
    elif decision in {"card_reward", "card_select"}:
        game_state["screen_state"] = {"cards": _list(state.get("cards"))}
    elif decision == "shop":
        game_state["screen_state"] = {
            "cards": _list(state.get("cards")),
            "relics": _list(state.get("relics")),
            "potions": _list(state.get("potions")),
        }

    state["game_state"] = game_state
    state["available_commands"] = _available_commands(state)
    state["in_game"] = decision != "game_over" and state.get("type") != "error"
    state["_game_version"] = "sts2"
    return state


def _combat_state(state: dict[str, Any], player: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "player": {
            "current_hp": player.get("hp", 0),
            "max_hp": player.get("max_hp", 1),
            "energy": state.get("energy", 0),
            "max_energy": state.get("max_energy", 0),
            "block": player.get("block", 0),
            "powers": _list(state.get("player_powers")),
        },
        "hand": _list(state.get("hand")),
        "monsters": [_adapt_enemy(enemy) for enemy in _list_of_mappings(state.get("enemies"))],
    }


def _adapt_enemy(enemy: Mapping[str, Any]) -> dict[str, Any]:
    intents = _list_of_mappings(enemy.get("intents"))
    first_intent = intents[0] if intents else {}
    return {
        **dict(enemy),
        "current_hp": enemy.get("hp", enemy.get("current_hp", 0)),
        "max_hp": enemy.get("max_hp", 1),
        "intent": first_intent.get("type", "UNKNOWN"),
        "move_adjusted_damage": first_intent.get(
            "total_damage",
            first_intent.get("damage", 0),
        ),
        "move_hits": first_intent.get("hits", 1),
        "is_gone": False,
    }


def _adapt_deck_card(card: Any) -> Any:
    if not isinstance(card, Mapping):
        return card
    adapted = dict(card)
    adapted.setdefault("upgrades", 1 if adapted.get("upgraded") else 0)
    return adapted


def _available_commands(state: dict[str, Any]) -> list[str]:
    decision = str(state.get("decision") or "").lower()
    if decision == "combat_play":
        commands = ["play", "end", "state"]
        if _list(_mapping(state.get("player")).get("potions")):
            commands.append("potion")
        return commands
    if decision in {"map_select", "event_choice", "rest_site", "card_select", "bundle_select"}:
        return ["choose", "return", "state"]
    if decision == "card_reward":
        return ["choose", "skip", "state"] if state.get("can_skip", True) else ["choose", "state"]
    if decision == "shop":
        return ["choose", "return", "state"]
    return []


def _encode_potions(obs: np.ndarray, start: int, potions: list[Any]) -> None:
    width = 3
    for slot, potion in enumerate(potions[:5]):
        if not isinstance(potion, Mapping):
            continue
        offset = start + slot * width
        obs[offset] = 1.0
        obs[offset + 1] = _type_id(potion.get("target_type"), TARGET_TYPE_IDS) / 10.0
        obs[offset + 2] = len(_mapping(potion.get("vars"))) / 10.0


def _encode_cards(obs: np.ndarray, start: int, cards: list[Any]) -> None:
    width = 8
    for slot, card in enumerate(cards[:10]):
        if not isinstance(card, Mapping):
            continue
        stats = _mapping(card.get("stats"))
        offset = start + slot * width
        obs[offset] = 1.0
        obs[offset + 1] = _number(card.get("cost"), 0) / 5.0
        obs[offset + 2] = 1.0 if card.get("can_play", True) else 0.0
        obs[offset + 3] = _type_id(card.get("type"), CARD_TYPE_IDS) / 10.0
        obs[offset + 4] = _type_id(card.get("target_type"), TARGET_TYPE_IDS) / 10.0
        obs[offset + 5] = _number(stats.get("damage"), stats.get("calculateddamage", 0)) / 100.0
        obs[offset + 6] = _number(stats.get("block"), 0) / 100.0
        obs[offset + 7] = _number(stats.get("magic"), stats.get("misc", 0)) / 100.0


def _encode_enemies(obs: np.ndarray, start: int, enemies: list[Any]) -> None:
    width = 8
    for slot, enemy in enumerate(enemies[:5]):
        if not isinstance(enemy, Mapping):
            continue
        intents = _list_of_mappings(enemy.get("intents"))
        first_intent = intents[0] if intents else {}
        hp = _number(enemy.get("hp", enemy.get("current_hp")), 0)
        max_hp = max(1.0, _number(enemy.get("max_hp"), 1))
        offset = start + slot * width
        obs[offset] = 1.0
        obs[offset + 1] = hp / max_hp
        obs[offset + 2] = hp / 300.0
        obs[offset + 3] = _number(enemy.get("block"), 0) / 100.0
        obs[offset + 4] = 1.0 if enemy.get("intends_attack") else 0.0
        obs[offset + 5] = _number(first_intent.get("total_damage", first_intent.get("damage")), 0) / 100.0
        obs[offset + 6] = _number(first_intent.get("hits"), 1) / 10.0
        obs[offset + 7] = _type_id(first_intent.get("type"), INTENT_IDS) / 10.0


def _encode_decision_items(obs: np.ndarray, start: int, state: dict[str, Any]) -> None:
    decision = str(state.get("decision") or "").lower()
    items = (
        _list(state.get("choices"))
        or _list(state.get("options"))
        or _list(state.get("cards"))
        or _list(state.get("bundles"))
    )
    obs[start] = min(len(items), 30) / 30.0
    obs[start + 1] = 1.0 if decision == "game_over" and state.get("victory") else 0.0
    obs[start + 2] = 1.0 if decision == "game_over" and not state.get("victory") else 0.0


def _set_one_hot(obs: np.ndarray, start: int, labels: tuple[str, ...], value: str) -> None:
    try:
        index = labels.index(value)
    except ValueError:
        index = labels.index("UNKNOWN")
    obs[start + index] = 1.0


def _type_id(value: Any, mapping: dict[str, float]) -> float:
    key = str(value or "").replace("_", "").lower()
    return mapping.get(key, 0.0)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]


def _number(value: Any, default: Any) -> float:
    try:
        if value is None:
            value = default
        return float(value)
    except (TypeError, ValueError):
        return float(default or 0.0)
