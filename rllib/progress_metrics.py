"""RLlib callbacks for run-progress metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sts2.encounters import known_combat_encounter_ids

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except ImportError:  # pragma: no cover - older Ray fallback
    from ray.rllib.agents.callbacks import DefaultCallbacks  # type: ignore


class ProgressMetricsCallback(DefaultCallbacks):
    """Aggregate floor and act progress from env info dictionaries."""

    def on_episode_start(self, *, episode: Any, **kwargs: Any) -> None:
        episode.user_data["floor_values"] = []
        episode.user_data["max_floor"] = 0
        episode.user_data["boss_reached"] = 0.0
        episode.user_data["boss_killed"] = 0.0
        episode.user_data["act2"] = 0.0
        episode.user_data["combat_metrics"] = {}

    def on_episode_step(self, *, episode: Any, **kwargs: Any) -> None:
        self._record_progress(episode, _last_episode_info(episode))

    def on_episode_end(self, *, episode: Any, **kwargs: Any) -> None:
        self._record_progress(episode, _last_episode_info(episode))
        floors = episode.user_data.get("floor_values") or [0]
        episode.custom_metrics["floor"] = float(max(floors))
        episode.custom_metrics["max_floor"] = float(max(floors))
        episode.custom_metrics["boss_reached_pct"] = 100.0 * float(
            episode.user_data.get("boss_reached", 0.0)
        )
        episode.custom_metrics["boss_killed_pct"] = 100.0 * float(
            episode.user_data.get("boss_killed", 0.0)
        )
        episode.custom_metrics["act2_pct"] = 100.0 * float(
            episode.user_data.get("act2", 0.0)
        )
        combat = episode.user_data.get("combat_metrics")
        if isinstance(combat, dict) and combat:
            episode.custom_metrics["combat_win_rate"] = _safe_float(
                combat.get("combat_win"),
                0.0,
            )
            episode.custom_metrics["combat_loss_rate"] = _safe_float(
                combat.get("combat_loss"),
                0.0,
            )
            episode.custom_metrics["combat_timeout_rate"] = _safe_float(
                combat.get("combat_timeout"),
                0.0,
            )
            episode.custom_metrics["avg_combat_steps"] = _safe_float(
                combat.get("combat_steps"),
                0.0,
            )
            episode.custom_metrics["avg_hp_remaining_on_win"] = _safe_float(
                combat.get("hp_remaining_on_win"),
                0.0,
            )
            episode.custom_metrics["avg_hp_lost"] = _safe_float(
                combat.get("hp_lost"),
                0.0,
            )
            episode.custom_metrics["avg_monster_hp_remaining_on_loss"] = _safe_float(
                combat.get("monster_hp_remaining_on_loss"),
                0.0,
            )
            for metric_name in (
                "boss_hp_remaining_on_loss",
                "boss_hp_fraction_removed",
                "min_boss_hp_reached",
                "damage_dealt_total",
                "add_damage_dealt_total",
                "add_hp_fraction_removed",
                "turns_survived",
                "end_turn_with_energy",
                "end_turn_with_energy_rate",
                "end_turn_with_playable_attack",
                "end_turn_with_playable_attack_rate",
                "end_turn_with_playable_block_when_incoming_damage",
                "end_turn_with_playable_block_when_incoming_damage_rate",
                "power_play_rate",
                "block_when_incoming_damage_rate",
            ):
                episode.custom_metrics[metric_name] = _safe_float(
                    combat.get(metric_name),
                    0.0,
                )
            episode.custom_metrics["deck_size"] = _safe_float(
                combat.get("deck_size"),
                0.0,
            )
            encounter = _metric_key(combat.get("encounter_id"))
            reason = _metric_key(combat.get("terminated_reason"))
            pool_ids = combat.get("encounter_pool_ids")
            if not isinstance(pool_ids, list) or not pool_ids:
                pool_ids = list(known_combat_encounter_ids())
            for known_encounter in pool_ids:
                known_key = _metric_key(known_encounter)
                if known_key:
                    episode.custom_metrics[f"encounter_id_{known_key}"] = float(
                        known_key == encounter
                    )
            for known_reason in ("win", "loss", "timeout", "ongoing"):
                episode.custom_metrics[f"terminated_reason_{known_reason}"] = float(
                    known_reason == reason
                )
            # Grouped category metrics (weak/normal/elite/boss)
            raw_encounter = str(combat.get("encounter_id") or "")
            category = classify_encounter(raw_encounter)
            is_win = float(_safe_float(combat.get("combat_win"), 0.0))
            hp_lost = _safe_float(combat.get("hp_lost"), 0.0)
            boss_hp_loss_remaining = _safe_float(
                combat.get("boss_hp_remaining_on_loss"),
                0.0,
            )
            for cat in ("weak", "normal", "elite", "boss"):
                in_category = category == cat
                episode.custom_metrics[f"{cat}_win_count"] = is_win if in_category else 0.0
                episode.custom_metrics[f"{cat}_hp_lost_sum"] = hp_lost if in_category else 0.0
                episode.custom_metrics[f"{cat}_encounter_count"] = 1.0 if in_category else 0.0
                # Backward-compatible raw one-hot fields. Do not use these as
                # category win rates across mixed pools; train_rllib derives
                # true category rates from win_count / encounter_count.
                episode.custom_metrics[f"{cat}_win_rate"] = is_win if in_category else 0.0
                episode.custom_metrics[f"{cat}_avg_hp_lost"] = hp_lost if in_category else 0.0
            for known_encounter in pool_ids:
                known_key = _metric_key(known_encounter)
                if not known_key:
                    continue
                in_encounter = known_key == encounter
                is_boss = classify_encounter(str(known_encounter)) == "boss"
                if is_boss:
                    episode.custom_metrics[f"boss_{known_key}_fight_count"] = float(
                        in_encounter
                    )
                    episode.custom_metrics[f"boss_{known_key}_win_count"] = (
                        is_win if in_encounter else 0.0
                    )
                    episode.custom_metrics[f"boss_{known_key}_hp_lost_sum"] = (
                        hp_lost if in_encounter else 0.0
                    )
                    episode.custom_metrics[f"boss_{known_key}_hp_remaining_on_loss_sum"] = (
                        boss_hp_loss_remaining if in_encounter else 0.0
                    )
            cards_played = combat.get("cards_played_by_id")
            if isinstance(cards_played, dict):
                for raw_card_id, count in cards_played.items():
                    card_key = _metric_key(raw_card_id)
                    if card_key:
                        episode.custom_metrics[f"card_played_{card_key}"] = _safe_float(
                            count,
                            0.0,
                        )

    @staticmethod
    def _record_progress(episode: Any, info: Any) -> None:
        if not isinstance(info, dict):
            return
        progress = info.get("progress_metrics")
        if not isinstance(progress, dict):
            return

        floor = _safe_float(progress.get("floor"), 0.0)
        episode.user_data.setdefault("floor_values", []).append(floor)
        episode.user_data["max_floor"] = max(
            float(episode.user_data.get("max_floor", 0.0)),
            floor,
        )
        for key in ("boss_reached", "boss_killed", "act2"):
            episode.user_data[key] = max(
                float(episode.user_data.get(key, 0.0)),
                _safe_float(progress.get(key), 0.0),
            )
        combat_keys = (
            "combat_win",
            "combat_loss",
            "combat_timeout",
            "combat_steps",
            "hp_remaining_on_win",
            "hp_lost",
            "monster_hp_remaining_on_loss",
            "boss_hp_remaining_on_loss",
            "boss_hp_fraction_removed",
            "min_boss_hp_reached",
            "damage_dealt_total",
            "add_damage_dealt_total",
            "add_hp_fraction_removed",
            "turns_survived",
            "end_turn_with_energy",
            "end_turn_with_energy_rate",
            "end_turn_with_playable_attack",
            "end_turn_with_playable_attack_rate",
            "end_turn_with_playable_block_when_incoming_damage",
            "end_turn_with_playable_block_when_incoming_damage_rate",
            "power_play_rate",
            "block_when_incoming_damage_rate",
            "cards_played_by_id",
            "encounter_id",
            "encounter_pool",
            "encounter_pool_ids",
            "terminated_reason",
            "deck_mode",
            "deck_source",
            "deck_size",
        )
        if any(key in progress for key in combat_keys):
            combat = episode.user_data.setdefault("combat_metrics", {})
            for key in combat_keys:
                if key in progress:
                    combat[key] = progress[key]


def _last_episode_info(episode: Any) -> Any:
    getter = getattr(episode, "last_info_for", None)
    if callable(getter):
        for args in ((), ("agent0",), ("__default_policy__",)):
            try:
                info = getter(*args)
            except TypeError:
                continue
            if info is not None:
                return info

    infos = getattr(episode, "_last_infos", None)
    if isinstance(infos, dict) and infos:
        return next(iter(infos.values()))
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_key(value: Any) -> str:
    text = str(value or "").strip()
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def classify_encounter(encounter_id: str) -> str:
    """Classify an encounter as weak/normal/elite/boss/other by suffix."""
    upper = str(encounter_id or "").strip().upper()
    if upper.endswith("_WEAK"):
        return "weak"
    if upper.endswith("_NORMAL"):
        return "normal"
    if upper.endswith("_ELITE"):
        return "elite"
    if upper.endswith("_BOSS"):
        return "boss"
    return "other"
