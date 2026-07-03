"""Curated STS2 combat encounter pools for curriculum experiments."""

from __future__ import annotations

from typing import Iterable


FIXED_DEFAULT_ENCOUNTER = "SHRINKER_BEETLE_WEAK"

# STS2 Act 1 is Overgrowth in the current sts2-cli build. Keep these lists
# explicit so curriculum experiments are reproducible across training runs.
ACT1_HALLWAY_ENCOUNTERS: tuple[str, ...] = (
    "SHRINKER_BEETLE_WEAK",
    "TOADPOLES_WEAK",
    "FUZZY_WURM_CRAWLER_WEAK",
    "SLUDGE_SPINNER_WEAK",
    "THIEVING_HOPPER_WEAK",
    "TUNNELER_WEAK",
    "FLYCONID_NORMAL",
    "OVERGROWTH_CRAWLERS",
    "CHOMPERS_NORMAL",
    "SLUMBERING_BEETLE_NORMAL",
    "SPINY_TOAD_NORMAL",
    "VINE_SHAMBLER_NORMAL",
)

ACT1_ELITE_ENCOUNTERS: tuple[str, ...] = (
    "BYRDONIS_ELITE",
    "ENTOMANCER_ELITE",
    "PHROG_PARASITE_ELITE",
)

ACT1_BOSS_ENCOUNTERS: tuple[str, ...] = (
    "CEREMONIAL_BEAST_BOSS",
    "THE_KIN_BOSS",
    "VANTOM_BOSS",
)

# Act 2 (Underdocks)
ACT2_HALLWAY_ENCOUNTERS: tuple[str, ...] = (
    "SEAPUNK_WEAK",
    "NIBBITS_WEAK",
    "SEAPUNK_NORMAL",
    "SEWER_CLAM_NORMAL",
    "FROG_KNIGHT_NORMAL",
    "HAUNTED_SHIP_NORMAL",
    "INKLETS_NORMAL",
    "NIBBITS_NORMAL",
    "SLITHERING_STRANGLER_NORMAL",
    "TWO_TAILED_RATS_NORMAL",
    "OWL_MAGISTRATE_NORMAL",
    "FOSSIL_STALKER_NORMAL",
    "SLIMED_BERSERKER_NORMAL",
    "LIVING_FOG_NORMAL",
    "SLIMES_NORMAL",
    "SLIMES_WEAK",
    "GREMLIN_MERC_NORMAL",
)

ACT2_ELITE_ENCOUNTERS: tuple[str, ...] = (
    "TERROR_EEL_ELITE",
    "KNIGHTS_ELITE",
    "SOUL_NEXUS_ELITE",
)

ACT2_BOSS_ENCOUNTERS: tuple[str, ...] = (
    "WATERFALL_GIANT_BOSS",
    "KAISER_CRAB_BOSS",
    "SOUL_FYSH_BOSS",
    "LAGAVULIN_MATRIARCH_BOSS",
)

# Act 3 (Hive)
ACT3_HALLWAY_ENCOUNTERS: tuple[str, ...] = (
    "BOWLBUGS_WEAK",
    "CORPSE_SLUGS_WEAK",
    "EXOSKELETONS_WEAK",
    "TURRET_OPERATOR_WEAK",
    "SCROLLS_OF_BITING_WEAK",
    "BOWLBUGS_NORMAL",
    "CORPSE_SLUGS_NORMAL",
    "EXOSKELETONS_NORMAL",
    "AXEBOTS_NORMAL",
    "CONSTRUCT_MENAGERIE_NORMAL",
    "CUBEX_CONSTRUCT_NORMAL",
    "FABRICATOR_NORMAL",
    "GLOBE_HEAD_NORMAL",
    "HUNTER_KILLER_NORMAL",
    "LOUSE_PROGENITOR_NORMAL",
    "MYTES_NORMAL",
    "OVICOPTER_NORMAL",
    "PUNCH_CONSTRUCT_NORMAL",
    "SCROLLS_OF_BITING_NORMAL",
    "MAWLER_NORMAL",
    "CULTISTS_NORMAL",
    "THE_LOST_AND_FORGOTTEN_NORMAL",
    "THE_OBSCURA_NORMAL",
    "RUBY_RAIDERS_NORMAL",
    "FOGMOG_NORMAL",
)

ACT3_ELITE_ENCOUNTERS: tuple[str, ...] = (
    "BYGONE_EFFIGY_ELITE",
    "DECIMILLIPEDE_ELITE",
    "INFESTED_PRISMS_ELITE",
    "MECHA_KNIGHT_ELITE",
    "SKULKING_COLONY_ELITE",
    "PHANTASMAL_GARDENERS_ELITE",
)

ACT3_BOSS_ENCOUNTERS: tuple[str, ...] = (
    "QUEEN_BOSS",
    "AEONGLASS_BOSS",
    "KNOWLEDGE_DEMON_BOSS",
    "TEST_SUBJECT_BOSS",
    "THE_INSATIABLE_BOSS",
)

COMBAT_ENEMY_POOLS: dict[str, tuple[str, ...]] = {
    "fixed": (FIXED_DEFAULT_ENCOUNTER,),
    "act1_hallway": ACT1_HALLWAY_ENCOUNTERS,
    "act1_elite": ACT1_ELITE_ENCOUNTERS,
    "act1_boss": ACT1_BOSS_ENCOUNTERS,
    "act1_hallway_elite": (
        *ACT1_HALLWAY_ENCOUNTERS,
        *ACT1_HALLWAY_ENCOUNTERS,
        *ACT1_ELITE_ENCOUNTERS,
    ),
    "act1_mixed": (
        *ACT1_HALLWAY_ENCOUNTERS,
        *ACT1_HALLWAY_ENCOUNTERS,
        *ACT1_ELITE_ENCOUNTERS,
        *ACT1_BOSS_ENCOUNTERS,
    ),
    "act2_hallway": ACT2_HALLWAY_ENCOUNTERS,
    "act2_elite": ACT2_ELITE_ENCOUNTERS,
    "act2_boss": ACT2_BOSS_ENCOUNTERS,
    "act2_mixed": (
        *ACT2_HALLWAY_ENCOUNTERS,
        *ACT2_HALLWAY_ENCOUNTERS,
        *ACT2_ELITE_ENCOUNTERS,
        *ACT2_BOSS_ENCOUNTERS,
    ),
    "act3_hallway": ACT3_HALLWAY_ENCOUNTERS,
    "act3_elite": ACT3_ELITE_ENCOUNTERS,
    "act3_boss": ACT3_BOSS_ENCOUNTERS,
    "act3_mixed": (
        *ACT3_HALLWAY_ENCOUNTERS,
        *ACT3_HALLWAY_ENCOUNTERS,
        *ACT3_ELITE_ENCOUNTERS,
        *ACT3_BOSS_ENCOUNTERS,
    ),
    "all_mixed": (
        *ACT1_HALLWAY_ENCOUNTERS,
        *ACT1_ELITE_ENCOUNTERS,
        *ACT1_BOSS_ENCOUNTERS,
        *ACT2_HALLWAY_ENCOUNTERS,
        *ACT2_ELITE_ENCOUNTERS,
        *ACT2_BOSS_ENCOUNTERS,
        *ACT3_HALLWAY_ENCOUNTERS,
        *ACT3_ELITE_ENCOUNTERS,
        *ACT3_BOSS_ENCOUNTERS,
    ),
}


def combat_pool_names() -> tuple[str, ...]:
    """Return supported combat enemy pool names."""
    return tuple(COMBAT_ENEMY_POOLS)


def combat_pool_ids(pool_name: str, *, fixed_encounter: str | None = None) -> tuple[str, ...]:
    """Return encounter IDs for a pool.

    The `fixed` pool is special: it uses the user-selected encounter so smoke
    tests can stay pinned to a single fight.
    """
    normalized = _normalize_pool_name(pool_name)
    if normalized == "fixed":
        return (_normalize_encounter(fixed_encounter),)
    try:
        return COMBAT_ENEMY_POOLS[normalized]
    except KeyError as exc:
        supported = ", ".join(combat_pool_names())
        raise ValueError(
            f"Unsupported STS2 combat enemy pool: {pool_name!r}. "
            f"Supported pools: {supported}"
        ) from exc


def known_combat_encounter_ids() -> tuple[str, ...]:
    """Return stable unique encounter IDs used by all built-in combat pools."""
    return _unique(
        encounter
        for pool_name in combat_pool_names()
        for encounter in COMBAT_ENEMY_POOLS[pool_name]
    )


def _normalize_pool_name(pool_name: str | None) -> str:
    return str(pool_name or "fixed").strip().lower()


def _normalize_encounter(encounter: str | None) -> str:
    text = str(encounter or "").strip()
    return text or FIXED_DEFAULT_ENCOUNTER


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)
