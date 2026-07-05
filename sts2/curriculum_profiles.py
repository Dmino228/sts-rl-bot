"""Small named STS2 curriculum profiles for staged combat training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CurriculumProfile:
    name: str
    options: dict[str, Any]


THE_KIN_5_DECK_PROFILE_SEEDS = (101, 202, 303, 404, 505)


def sample_curriculum_profile(
    mix: str | None,
    *,
    rng: random.Random,
) -> CurriculumProfile | None:
    """Sample one named curriculum profile from a weighted spec.

    Spec format: ``name:weight,name:weight``. Unknown names raise ``ValueError``
    so bad long-run experiments fail before silently training on the wrong task.
    """
    entries = parse_curriculum_mix(mix)
    if not entries:
        return None
    total = sum(weight for _name, weight in entries)
    threshold = rng.random() * total
    cumulative = 0.0
    selected = entries[-1][0]
    for name, weight in entries:
        cumulative += weight
        if threshold <= cumulative:
            selected = name
            break
    return curriculum_profile(selected, rng=rng)


def parse_curriculum_mix(mix: str | None) -> list[tuple[str, float]]:
    text = str(mix or "").strip()
    if not text:
        return []
    entries: list[tuple[str, float]] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            name, raw_weight = part.split(":", 1)
        else:
            name, raw_weight = part, "1.0"
        name = name.strip()
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid curriculum mix weight in {part!r}") from exc
        if weight <= 0.0:
            raise ValueError(f"Curriculum mix weight must be positive in {part!r}")
        # Validate profile name eagerly.
        _profile_options(name, rng=random.Random(0))
        entries.append((name, weight))
    if not entries:
        return []
    return entries


def curriculum_profile(name: str, *, rng: random.Random) -> CurriculumProfile:
    normalized = _normalize_profile_name(name)
    return CurriculumProfile(normalized, _profile_options(normalized, rng=rng))


def _profile_options(name: str, *, rng: random.Random) -> dict[str, Any]:
    normalized = _normalize_profile_name(name)
    if normalized == "c0_the_kin_exact":
        return {
            "combat_room_type": "combat",
            "combat_encounter": "THE_KIN_BOSS",
            "combat_enemy_pool": "fixed",
            "deck_mode": "fixed_the_kin_overfit",
            "deck_seed": 20260705,
            "deck_episode_id": 0,
            "run_seed": 20260705,
        }
    if normalized == "c1_the_kin_5_decks":
        slot = rng.randrange(len(THE_KIN_5_DECK_PROFILE_SEEDS))
        return {
            "combat_room_type": "combat",
            "combat_encounter": "THE_KIN_BOSS",
            "combat_enemy_pool": "fixed",
            "deck_mode": "random_boss_synthetic_safe",
            "deck_seed": THE_KIN_5_DECK_PROFILE_SEEDS[slot],
            "deck_episode_id": 0,
            "deck_profile_slot": slot,
            "run_seed": 20260705 + slot,
        }
    if normalized == "c2_the_kin_random_safe":
        return {
            "combat_room_type": "combat",
            "combat_encounter": "THE_KIN_BOSS",
            "combat_enemy_pool": "fixed",
            "deck_mode": "random_boss_synthetic_safe",
        }
    raise ValueError(
        f"Unknown STS2 curriculum profile {name!r}. "
        "Known: c0_the_kin_exact, c1_the_kin_5_decks, c2_the_kin_random_safe."
    )


def _normalize_profile_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
