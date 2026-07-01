"""Strategic heuristics for STS2 non-combat decisions.

The first integration target is Option A from the curriculum notes: let PPO
control combat while a deterministic heuristic controls non-combat decisions.
The policy is written as an action ranker so it can later power top-k masks
(Option B) or behavior-cloning labels (Option C).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from sts2.action_space import (
    BACK_ACTION,
    CHOICE_BASE,
    END_TURN_ACTION,
    FALLBACK_ACTION,
    PROCEED_ACTION,
    QUIT_ACTION,
    build_legal_commands,
)


COMBAT_ROOM_TYPES = {"boss", "elite", "monster"}
COMBAT_CONTROL_DECISIONS = {"combat_play"}
NONCOMBAT_DECISIONS = {
    "map_select",
    "card_reward",
    "event_choice",
    "rest_site",
    "shop",
    "card_select",
    "bundle_select",
    "game_over",
    "unknown",
}


@dataclass(frozen=True)
class HeuristicAction:
    """One ranked action candidate."""

    action_id: int
    score: float
    reason: str
    phase: str


@dataclass(frozen=True)
class HeuristicDecision:
    """Selected action plus a compact top-k explanation for logs/datasets."""

    action_id: int
    reason: str
    phase: str
    candidates: tuple[HeuristicAction, ...]

    def as_info(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "reason": self.reason,
            "phase": self.phase,
            "candidates": [
                {
                    "action_id": candidate.action_id,
                    "score": round(candidate.score, 3),
                    "reason": candidate.reason,
                    "phase": candidate.phase,
                }
                for candidate in self.candidates[:5]
            ],
        }


class StS2StrategicHeuristic:
    """Conservative non-combat policy for the shared 100-action STS2 space."""

    def select_action(
        self,
        state: Mapping[str, Any],
        action_mask: np.ndarray,
    ) -> Optional[HeuristicDecision]:
        ranked = self.rank_actions(state, action_mask)
        if not ranked:
            return None
        best = ranked[0]
        return HeuristicDecision(
            action_id=best.action_id,
            reason=best.reason,
            phase=best.phase,
            candidates=tuple(ranked[:5]),
        )

    def mask_for_mode(
        self,
        state: Mapping[str, Any],
        action_mask: np.ndarray,
        *,
        mode: str,
        top_k: int = 1,
    ) -> tuple[np.ndarray, Optional[HeuristicDecision]]:
        decision = self.select_action(state, action_mask)
        if decision is None:
            return action_mask, None

        normalized_mode = mode.strip().lower()
        if normalized_mode == "hard":
            return _one_hot_like(action_mask, decision.action_id), decision

        if normalized_mode == "mask":
            ranked = decision.candidates[: max(1, int(top_k))]
            mask = np.zeros_like(action_mask)
            for candidate in ranked:
                mask[candidate.action_id] = 1
            if not np.any(mask > 0):
                return action_mask, None
            return mask, decision

        return action_mask, None

    def rank_actions(
        self,
        state: Mapping[str, Any],
        action_mask: np.ndarray,
    ) -> list[HeuristicAction]:
        if not self.should_control(state):
            return []

        legal_actions = _legal_action_ids(action_mask)
        if not legal_actions:
            return []

        decision = _decision(state)
        commands = build_legal_commands(dict(state))
        rankers = {
            "map_select": self._rank_map_select,
            "card_reward": self._rank_card_reward,
            "event_choice": self._rank_event_choice,
            "rest_site": self._rank_rest_site,
            "shop": self._rank_shop,
            "card_select": self._rank_card_select,
            "bundle_select": self._rank_indexed_first,
            "game_over": self._rank_game_over,
            "unknown": self._rank_unknown,
        }
        ranker = rankers.get(decision, self._rank_fallback)
        ranked = ranker(state, legal_actions, commands)
        return sorted(
            ranked,
            key=lambda candidate: (-candidate.score, candidate.action_id),
        )

    def should_control(self, state: Mapping[str, Any]) -> bool:
        decision = _decision(state)
        if decision in COMBAT_CONTROL_DECISIONS:
            return False
        if decision == "card_select" and _room_type(state) in COMBAT_ROOM_TYPES:
            return False
        return decision in NONCOMBAT_DECISIONS

    def _rank_map_select(
        self,
        state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        choices = _list(state.get("choices"))
        hp_ratio = _hp_ratio(state)
        gold = _number(_player(state).get("gold"), 0.0)
        ranked: list[HeuristicAction] = []

        for action_id in legal_actions:
            slot = action_id - CHOICE_BASE
            choice = choices[slot] if 0 <= slot < len(choices) else {}
            node_type = _node_type(choice)
            score = {
                "boss": 100.0,
                "rest": 8.0 if hp_ratio < 0.55 else 2.5,
                "treasure": 6.0,
                "monster": 5.0,
                "event": 4.0,
                "shop": 4.0 if gold >= 120 else 1.0,
                "elite": 4.0 if hp_ratio >= 0.70 else -3.0,
            }.get(node_type, 0.0)
            score -= slot * 0.01
            ranked.append(
                HeuristicAction(action_id, score, f"map:{node_type}", "map_select")
            )
        return ranked

    def _rank_card_reward(
        self,
        state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        cards = _list(state.get("cards"))
        can_skip = bool(state.get("can_skip", True))
        ranked: list[HeuristicAction] = []

        for action_id in legal_actions:
            if action_id == BACK_ACTION:
                ranked.append(
                    HeuristicAction(
                        action_id,
                        1.25 if can_skip else -20.0,
                        "card_reward:skip",
                        "card_reward",
                    )
                )
                continue
            slot = action_id - CHOICE_BASE
            card = cards[slot] if 0 <= slot < len(cards) else {}
            score = self._score_card_pick(card, state)
            ranked.append(
                HeuristicAction(
                    action_id,
                    score,
                    f"card_reward:{_card_name(card)}",
                    "card_reward",
                )
            )
        return ranked

    def _rank_event_choice(
        self,
        _state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        ranked: list[HeuristicAction] = []
        for action_id in legal_actions:
            if action_id == BACK_ACTION:
                score = -1.0
                reason = "event:leave"
            else:
                slot = max(0, action_id - CHOICE_BASE)
                score = 5.0 - slot * 0.1
                reason = f"event:option_{slot}"
            ranked.append(HeuristicAction(action_id, score, reason, "event_choice"))
        return ranked

    def _rank_rest_site(
        self,
        state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        options = _list(state.get("options"))
        hp_ratio = _hp_ratio(state)
        ranked: list[HeuristicAction] = []
        for action_id in legal_actions:
            slot = action_id - CHOICE_BASE
            option = options[slot] if 0 <= slot < len(options) else {}
            label = _option_label(option)
            if "rest" in label or "heal" in label:
                score = 10.0 if hp_ratio < 0.50 else 4.0 if hp_ratio < 0.72 else 0.5
            elif "smith" in label or "upgrade" in label:
                score = 8.0 if hp_ratio >= 0.45 else 1.0
            elif "remove" in label or "toke" in label:
                score = 4.0
            elif "dig" in label or "lift" in label:
                score = 3.0
            else:
                score = 1.0 - max(0, slot) * 0.1
            ranked.append(
                HeuristicAction(action_id, score, f"rest:{label or slot}", "rest_site")
            )
        return ranked

    def _rank_shop(
        self,
        state: Mapping[str, Any],
        legal_actions: list[int],
        commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        ranked: list[HeuristicAction] = []
        for action_id in legal_actions:
            command = commands.get(action_id, {})
            action = str(command.get("action") or "").lower()
            args = command.get("args") if isinstance(command.get("args"), Mapping) else {}

            if action == "leave_room":
                ranked.append(HeuristicAction(action_id, 0.0, "shop:leave", "shop"))
            elif action == "buy_relic":
                relic = _indexed_item(state.get("relics"), args.get("relic_index"))
                cost = _number(relic.get("cost") if isinstance(relic, Mapping) else None, 0.0)
                ranked.append(
                    HeuristicAction(action_id, 6.0 - cost / 250.0, "shop:buy_relic", "shop")
                )
            elif action == "buy_card":
                card = _indexed_item(state.get("cards"), args.get("card_index"))
                cost = _number(card.get("cost") if isinstance(card, Mapping) else None, 0.0)
                score = self._score_card_pick(card, state) - cost / 120.0
                ranked.append(
                    HeuristicAction(
                        action_id,
                        score,
                        f"shop:buy_card:{_card_name(card)}",
                        "shop",
                    )
                )
            elif action == "remove_card":
                score = 5.0 if _deck_has_bad_cards(state) else 2.0
                ranked.append(HeuristicAction(action_id, score, "shop:remove_card", "shop"))
            elif action == "buy_potion":
                ranked.append(HeuristicAction(action_id, 1.5, "shop:buy_potion", "shop"))
            else:
                ranked.append(HeuristicAction(action_id, -1.0, f"shop:{action}", "shop"))
        return ranked

    def _rank_card_select(
        self,
        state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        cards = _list(state.get("cards"))
        prompt = str(state.get("prompt") or state.get("message") or "").lower()
        min_select = int(_number(state.get("min_select"), 1.0))
        choose_worst = any(token in prompt for token in ("remove", "purge", "discard", "exhaust"))
        ranked: list[HeuristicAction] = []

        for action_id in legal_actions:
            if action_id == BACK_ACTION:
                score = 2.0 if min_select <= 0 else -20.0
                ranked.append(HeuristicAction(action_id, score, "card_select:skip", "card_select"))
                continue
            slot = action_id - CHOICE_BASE
            card = cards[slot] if 0 <= slot < len(cards) else {}
            card_score = self._score_card_pick(card, state)
            score = -card_score if choose_worst else card_score
            reason = "worst" if choose_worst else "best"
            ranked.append(
                HeuristicAction(
                    action_id,
                    score,
                    f"card_select:{reason}:{_card_name(card)}",
                    "card_select",
                )
            )
        return ranked

    def _rank_indexed_first(
        self,
        _state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        return [
            HeuristicAction(action_id, 1.0 - idx * 0.01, "indexed:first", "indexed")
            for idx, action_id in enumerate(legal_actions)
        ]

    def _rank_game_over(
        self,
        _state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        return [
            HeuristicAction(
                action_id,
                10.0 if action_id == QUIT_ACTION else 0.0,
                "game_over:quit",
                "game_over",
            )
            for action_id in legal_actions
        ]

    def _rank_unknown(
        self,
        _state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        return [
            HeuristicAction(
                action_id,
                10.0
                if action_id == PROCEED_ACTION
                else 5.0
                if action_id == FALLBACK_ACTION
                else 0.0,
                "unknown:advance",
                "unknown",
            )
            for action_id in legal_actions
        ]

    def _rank_fallback(
        self,
        _state: Mapping[str, Any],
        legal_actions: list[int],
        _commands: Mapping[int, Mapping[str, Any]],
    ) -> list[HeuristicAction]:
        priority = {
            PROCEED_ACTION: 10.0,
            FALLBACK_ACTION: 5.0,
            BACK_ACTION: 1.0,
            END_TURN_ACTION: -10.0,
        }
        return [
            HeuristicAction(
                action_id,
                priority.get(action_id, 0.0),
                "fallback",
                "fallback",
            )
            for action_id in legal_actions
        ]

    def _score_card_pick(self, card: Any, state: Mapping[str, Any]) -> float:
        if not isinstance(card, Mapping):
            return -5.0
        card_type = str(card.get("type") or "").lower()
        name = _normalized_name(card.get("id") or card.get("name"))
        rarity = str(card.get("rarity") or "").lower()
        cost = _number(card.get("cost"), 1.0)
        stats = card.get("stats") if isinstance(card.get("stats"), Mapping) else {}
        damage = _number(card.get("damage", stats.get("damage") if isinstance(stats, Mapping) else 0), 0.0)
        block = _number(card.get("block", stats.get("block") if isinstance(stats, Mapping) else 0), 0.0)
        floor = _number(_game_state(state).get("floor"), 0.0)
        deck_size = _number(_player(state).get("deck_size"), len(_list(_player(state).get("deck"))))

        if card_type in {"curse", "status"} or name in {"curse", "dazed", "slimed", "wound"}:
            return -10.0

        score = {
            "rare": 3.5,
            "uncommon": 2.0,
            "common": 1.0,
            "basic": -0.5,
        }.get(rarity, 0.75)

        if card_type == "attack":
            score += 2.0 if floor <= 7 or deck_size <= 14 else 0.75
            score += min(3.0, damage / 8.0)
        elif card_type == "skill":
            score += 0.75 + min(2.0, block / 8.0)
        elif card_type == "power":
            score += 2.0

        if cost == 0:
            score += 0.5
        elif cost >= 3:
            score -= 0.75

        copies = _deck_count(state, name)
        if copies >= 2 and name not in {"strike", "defend"}:
            score -= 0.75 * (copies - 1)
        if name in {"strike", "defend", "strike+", "defend+"}:
            score -= 1.0

        return score


def _one_hot_like(mask: np.ndarray, action_id: int) -> np.ndarray:
    result = np.zeros_like(mask)
    if 0 <= action_id < len(result):
        result[action_id] = 1
    return result


def _legal_action_ids(mask: np.ndarray) -> list[int]:
    return [int(action_id) for action_id in np.flatnonzero(np.asarray(mask) > 0)]


def _decision(state: Mapping[str, Any]) -> str:
    return str(state.get("decision") or "").strip().lower()


def _game_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = state.get("game_state")
    return value if isinstance(value, Mapping) else {}


def _player(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = state.get("player")
    if isinstance(value, Mapping):
        return value
    game_player = _game_state(state).get("player")
    return game_player if isinstance(game_player, Mapping) else {}


def _room_type(state: Mapping[str, Any]) -> str:
    context = state.get("context")
    if not isinstance(context, Mapping):
        context = {}
    return str(context.get("room_type") or _game_state(state).get("room_type") or "").lower()


def _hp_ratio(state: Mapping[str, Any]) -> float:
    player = _player(state)
    current_hp = _number(player.get("hp"), _game_state(state).get("current_hp", 0.0))
    max_hp = max(1.0, _number(player.get("max_hp"), _game_state(state).get("max_hp", 1.0)))
    return max(0.0, min(1.0, current_hp / max_hp))


def _node_type(choice: Any) -> str:
    if not isinstance(choice, Mapping):
        return "unknown"
    raw = str(
        choice.get("type")
        or choice.get("room_type")
        or choice.get("symbol")
        or choice.get("node_type")
        or ""
    ).lower()
    compact = raw.replace("_", "").replace(" ", "")
    if compact in {"m", "monster", "hallway"}:
        return "monster"
    if compact in {"e", "elite"}:
        return "elite"
    if compact in {"?", "event", "unknown"}:
        return "event"
    if compact in {"$", "shop", "merchant"}:
        return "shop"
    if compact in {"r", "rest", "restsite", "campfire"}:
        return "rest"
    if compact in {"t", "treasure", "chest"}:
        return "treasure"
    if compact in {"boss"}:
        return "boss"
    return compact or "unknown"


def _option_label(option: Any) -> str:
    if not isinstance(option, Mapping):
        return ""
    return str(
        option.get("id")
        or option.get("name")
        or option.get("label")
        or option.get("text")
        or option.get("type")
        or ""
    ).lower()


def _indexed_item(items: Any, item_index: Any) -> Any:
    candidates = _list(items)
    wanted = int(_number(item_index, -1.0))
    for slot, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            continue
        if int(_number(item.get("index"), slot)) == wanted:
            return item
    if 0 <= wanted < len(candidates):
        return candidates[wanted]
    return {}


def _deck_has_bad_cards(state: Mapping[str, Any]) -> bool:
    deck = _list(_player(state).get("deck")) or _list(_game_state(state).get("deck"))
    for card in deck:
        if not isinstance(card, Mapping):
            continue
        card_type = str(card.get("type") or "").lower()
        name = _normalized_name(card.get("id") or card.get("name"))
        if card_type == "curse" or name in {"curse", "regret", "pain", "doubt", "shame"}:
            return True
    return False


def _deck_count(state: Mapping[str, Any], normalized_name: str) -> int:
    deck = _list(_player(state).get("deck")) or _list(_game_state(state).get("deck"))
    return sum(
        1
        for card in deck
        if isinstance(card, Mapping)
        and _normalized_name(card.get("id") or card.get("name")) == normalized_name
    )


def _card_name(card: Any) -> str:
    if not isinstance(card, Mapping):
        return "unknown"
    return str(card.get("name") or card.get("id") or "unknown")


def _normalized_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return "_".join(text.replace("-", " ").split())


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any, default: Any = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)
