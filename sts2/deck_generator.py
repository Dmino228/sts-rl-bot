"""Combat-curriculum deck generators for STS2 Headless."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from sts2.spire_codex import SpireCodex


SUPPORTED_DECK_MODES = frozenset(
    {
        "starter",
        "random_synthetic",
        "random_act1_floor_bucket",
        "random_boss_synthetic_safe",
    }
)

STARTER_RELICS = ("RELIC.BURNING_BLOOD",)
STARTER_POTIONS: tuple[str, ...] = ()
IRONCLAD_STARTER_DECK = (
    "CARD.STRIKE_IRONCLAD",
    "CARD.STRIKE_IRONCLAD",
    "CARD.STRIKE_IRONCLAD",
    "CARD.STRIKE_IRONCLAD",
    "CARD.STRIKE_IRONCLAD",
    "CARD.DEFEND_IRONCLAD",
    "CARD.DEFEND_IRONCLAD",
    "CARD.DEFEND_IRONCLAD",
    "CARD.DEFEND_IRONCLAD",
    "CARD.BASH",
)
STARTER_ATTACKS = frozenset({"CARD.STRIKE_IRONCLAD", "CARD.BASH"})
STARTER_BLOCKS = frozenset({"CARD.DEFEND_IRONCLAD"})
DEFAULT_DUPLICATE_CAP = 2

# Conservative exclusions for early combat curriculum. These are cards that
# commonly open extra selection/creation paths or are too dependent on missing
# run context for a first synthetic deck sweep.
DEFAULT_EXCLUDED_CARD_IDS = frozenset(
    {
        "CARD.BURNING_PACT",
        "CARD.FIEND_FIRE",
        "CARD.INFERNAL_BLADE",
        "CARD.SECOND_WIND",
        "CARD.TRUE_GRIT",
    }
)

BOSS_SAFE_CARD_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("SHRUG_IT_OFF", 4),
    ("IRON_WAVE", 4),
    ("POMMEL_STRIKE", 4),
    ("BATTLE_TRANCE", 3),
    ("INFLAME", 3),
    ("BODY_SLAM", 2),
    ("BARRICADE", 2),
    ("DEMON_FORM", 2),
    ("DEMONIC_SHIELD", 2),
    ("JUGGLING", 1),
)
BOSS_SAFE_RELIC_POOL = (
    "RELIC.ANCHOR",
    "RELIC.VAJRA",
    "RELIC.BAG_OF_PREPARATION",
    "RELIC.ODDLY_SMOOTH_STONE",
    "RELIC.ORICHALCUM",
    "RELIC.LANTERN",
)
BOSS_SAFE_POTION_POOL = (
    "POTION.BLOCK_POTION",
    "POTION.STRENGTH_POTION",
    "POTION.DEXTERITY_POTION",
    "POTION.FIRE_POTION",
)


@dataclass(frozen=True)
class DeckCardSpec:
    """One card entry in a generated deck."""

    id: str
    upgraded: bool = False
    name: str = ""
    rarity: str = ""
    type: str = ""

    @property
    def bare_id(self) -> str:
        return bare_model_id(self.id, "CARD")

    def to_command(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "upgraded": bool(self.upgraded),
        }

    def to_debug(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name or self.bare_id,
            "upgraded": bool(self.upgraded),
            "upgrades": 1 if self.upgraded else 0,
            "rarity": self.rarity or None,
            "type": self.type or None,
        }


@dataclass(frozen=True)
class CombatDeckSpec:
    """A complete player setup to apply before entering combat."""

    mode: str
    source: str
    cards: tuple[DeckCardSpec, ...]
    relics: tuple[str, ...] = STARTER_RELICS
    potions: tuple[str, ...] = STARTER_POTIONS
    hp: int = 80
    max_hp: int = 80
    added_cards: tuple[DeckCardSpec, ...] = ()
    removed_cards: tuple[DeckCardSpec, ...] = ()
    upgraded_cards: tuple[DeckCardSpec, ...] = ()
    floor_bucket: str | None = None
    synthetic_floor: int | None = None
    generator_settings: dict[str, Any] = field(default_factory=dict)
    apply_to_headless: bool = False

    @property
    def deck_size(self) -> int:
        return len(self.cards)

    def to_engine_options(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": self.source,
            "apply_to_headless": self.apply_to_headless,
            "cards": [card.to_command() for card in self.cards],
            "hp": int(self.hp),
            "max_hp": int(self.max_hp),
            "relics": list(self.relics),
            "potions": list(self.potions),
            "floor_bucket": self.floor_bucket,
            "synthetic_floor": self.synthetic_floor,
            "added_cards": [card.to_debug() for card in self.added_cards],
            "removed_cards": [card.to_debug() for card in self.removed_cards],
            "upgraded_cards": [card.to_debug() for card in self.upgraded_cards],
            "generator_settings": dict(self.generator_settings),
        }

    def to_debug(self) -> dict[str, Any]:
        data = self.to_engine_options()
        data["deck_size"] = self.deck_size
        data["deck"] = [card.to_debug() for card in self.cards]
        return data


def build_combat_deck_spec(
    *,
    mode: str,
    character: str = "Ironclad",
    seed: int | str | None = None,
    worker_id: int | None = None,
    episode_id: int = 0,
    duplicate_cap: int = DEFAULT_DUPLICATE_CAP,
    allow_problematic_cards: bool = False,
    codex: SpireCodex | None = None,
) -> CombatDeckSpec:
    """Build a deterministic combat deck spec for a reset."""
    normalized_mode = normalize_deck_mode(mode)
    if str(character).strip().lower() != "ironclad":
        raise ValueError(
            f"STS2 combat deck modes currently support Ironclad only, got {character!r}."
        )

    codex = codex or SpireCodex()
    settings = {
        "duplicate_cap": max(1, int(duplicate_cap)),
        "allow_problematic_cards": bool(allow_problematic_cards),
    }

    if normalized_mode == "starter":
        return _starter_spec(settings=settings)

    rng = random.Random(_stable_seed(seed, worker_id, episode_id, normalized_mode))
    pool = _load_ironclad_reward_pool(
        codex=codex,
        allow_problematic_cards=allow_problematic_cards,
    )
    if not pool:
        raise RuntimeError("No legal Ironclad reward cards available for STS2 deck generation.")

    if normalized_mode == "random_synthetic":
        return _random_synthetic_spec(
            rng=rng,
            pool=pool,
            duplicate_cap=max(1, int(duplicate_cap)),
            settings=settings,
        )
    if normalized_mode == "random_act1_floor_bucket":
        return _random_floor_bucket_spec(
            rng=rng,
            pool=pool,
            duplicate_cap=max(1, int(duplicate_cap)),
            settings=settings,
        )
    if normalized_mode == "random_boss_synthetic_safe":
        return _random_boss_safe_spec(
            rng=rng,
            pool=pool,
            duplicate_cap=max(2, int(duplicate_cap)),
            settings=settings,
        )
    raise ValueError(f"Unsupported STS2 deck mode: {mode!r}")


def normalize_deck_mode(mode: str | None) -> str:
    text = str(mode or "starter").strip().lower()
    if text in {"", "unspecified"}:
        text = "starter"
    if text not in SUPPORTED_DECK_MODES:
        raise ValueError(
            f"Unsupported STS2 deck mode {mode!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_DECK_MODES))}"
        )
    return text


def bare_model_id(identifier: str, kind: str) -> str:
    text = str(identifier or "").strip()
    prefix = f"{kind}."
    if text.upper().startswith(prefix):
        return text[len(prefix):]
    return text


def prefixed_card_id(identifier: str) -> str:
    bare = bare_model_id(identifier, "CARD")
    return f"CARD.{bare}"


def _starter_spec(*, settings: dict[str, Any]) -> CombatDeckSpec:
    return CombatDeckSpec(
        mode="starter",
        source="sts2_start_run_default",
        cards=tuple(_starter_cards()),
        generator_settings=settings,
        apply_to_headless=False,
    )


def _starter_cards() -> list[DeckCardSpec]:
    return [
        DeckCardSpec(
            id=card_id,
            name=_starter_name(card_id),
            rarity="Basic",
            type="Attack" if card_id in STARTER_ATTACKS else "Skill",
        )
        for card_id in IRONCLAD_STARTER_DECK
    ]


def _starter_name(card_id: str) -> str:
    return {
        "CARD.STRIKE_IRONCLAD": "Strike",
        "CARD.DEFEND_IRONCLAD": "Defend",
        "CARD.BASH": "Bash",
    }.get(card_id, bare_model_id(card_id, "CARD"))


def _random_synthetic_spec(
    *,
    rng: random.Random,
    pool: list[DeckCardSpec],
    duplicate_cap: int,
    settings: dict[str, Any],
) -> CombatDeckSpec:
    cards = _starter_cards()
    added = _sample_reward_cards(
        rng=rng,
        pool=pool,
        count=rng.randint(3, 8),
        duplicate_cap=duplicate_cap,
    )
    cards.extend(added)
    removed = _remove_starter_cards(rng=rng, cards=cards, max_remove=2)
    upgraded = _upgrade_cards(rng=rng, cards=cards, max_upgrades=2)
    return CombatDeckSpec(
        mode="random_synthetic",
        source="synthetic_ironclad_act1_reward_pool",
        cards=tuple(cards),
        added_cards=tuple(added),
        removed_cards=tuple(removed),
        upgraded_cards=tuple(upgraded),
        generator_settings=settings,
        apply_to_headless=True,
    )


def _random_floor_bucket_spec(
    *,
    rng: random.Random,
    pool: list[DeckCardSpec],
    duplicate_cap: int,
    settings: dict[str, Any],
) -> CombatDeckSpec:
    bucket = _weighted_choice(rng, [("early", 0.35), ("mid", 0.40), ("late", 0.25)])
    if bucket == "early":
        add_count = rng.randint(0, 2)
        max_remove = 1 if rng.random() < 0.10 else 0
        max_upgrades = 1 if rng.random() < 0.15 else 0
        synthetic_floor = rng.randint(1, 5)
        hp = rng.randint(64, 80)
    elif bucket == "mid":
        add_count = rng.randint(2, 5)
        max_remove = 1
        max_upgrades = 1
        synthetic_floor = rng.randint(6, 11)
        hp = rng.randint(50, 80)
    else:
        add_count = rng.randint(5, 9)
        max_remove = 2
        max_upgrades = rng.randint(1, 3)
        synthetic_floor = rng.randint(12, 16)
        hp = rng.randint(35, 80)

    cards = _starter_cards()
    added = _sample_reward_cards(
        rng=rng,
        pool=pool,
        count=add_count,
        duplicate_cap=duplicate_cap,
    )
    cards.extend(added)
    removed = _remove_starter_cards(rng=rng, cards=cards, max_remove=max_remove)
    upgraded = _upgrade_cards(rng=rng, cards=cards, max_upgrades=max_upgrades)
    return CombatDeckSpec(
        mode="random_act1_floor_bucket",
        source="synthetic_act1_floor_bucket",
        cards=tuple(cards),
        hp=hp,
        max_hp=80,
        added_cards=tuple(added),
        removed_cards=tuple(removed),
        upgraded_cards=tuple(upgraded),
        floor_bucket=bucket,
        synthetic_floor=synthetic_floor,
        generator_settings={**settings, "bucket": bucket},
        apply_to_headless=True,
    )


def _random_boss_safe_spec(
    *,
    rng: random.Random,
    pool: list[DeckCardSpec],
    duplicate_cap: int,
    settings: dict[str, Any],
) -> CombatDeckSpec:
    cards = _starter_cards()
    added = _sample_boss_safe_cards(
        rng=rng,
        pool=pool,
        count=rng.randint(7, 11),
        duplicate_cap=duplicate_cap,
    )
    cards.extend(added)
    removed = _remove_starter_cards(rng=rng, cards=cards, max_remove=2)
    upgraded = _upgrade_cards(rng=rng, cards=cards, max_upgrades=rng.randint(2, 4))
    extra_relics = tuple(rng.sample(list(BOSS_SAFE_RELIC_POOL), k=rng.randint(1, 2)))
    potions = (rng.choice(BOSS_SAFE_POTION_POOL),)
    return CombatDeckSpec(
        mode="random_boss_synthetic_safe",
        source="synthetic_boss_safe_ironclad",
        cards=tuple(cards),
        relics=tuple(dict.fromkeys((*STARTER_RELICS, *extra_relics))),
        potions=potions,
        hp=rng.randint(68, 80),
        max_hp=80,
        added_cards=tuple(added),
        removed_cards=tuple(removed),
        upgraded_cards=tuple(upgraded),
        generator_settings={
            **settings,
            "safe_card_weights": dict(BOSS_SAFE_CARD_WEIGHTS),
            "relic_pool": list(BOSS_SAFE_RELIC_POOL),
            "potion_pool": list(BOSS_SAFE_POTION_POOL),
        },
        apply_to_headless=True,
    )


def _sample_reward_cards(
    *,
    rng: random.Random,
    pool: list[DeckCardSpec],
    count: int,
    duplicate_cap: int,
) -> list[DeckCardSpec]:
    added: list[DeckCardSpec] = []
    counts: Counter[str] = Counter()
    attempts = 0
    while len(added) < count and attempts < count * 50:
        attempts += 1
        rarity = _weighted_choice(rng, [("Common", 0.75), ("Uncommon", 0.22), ("Rare", 0.03)])
        candidates = [card for card in pool if card.rarity.lower() == rarity.lower()]
        if not candidates:
            candidates = pool
        chosen = rng.choice(candidates)
        if counts[chosen.id] >= duplicate_cap:
            continue
        counts[chosen.id] += 1
        added.append(chosen)
    return added


def _sample_boss_safe_cards(
    *,
    rng: random.Random,
    pool: list[DeckCardSpec],
    count: int,
    duplicate_cap: int,
) -> list[DeckCardSpec]:
    by_bare_id = {card.bare_id.upper(): card for card in pool}
    weighted: list[DeckCardSpec] = []
    for bare_id, weight in BOSS_SAFE_CARD_WEIGHTS:
        card = by_bare_id.get(bare_id.upper())
        if card is not None:
            weighted.extend([card] * max(1, int(weight)))
    if not weighted:
        weighted = [
            card
            for card in pool
            if card.type.lower() in {"attack", "skill", "power"}
        ] or list(pool)

    added: list[DeckCardSpec] = []
    counts: Counter[str] = Counter()
    attempts = 0
    while len(added) < count and attempts < count * 80:
        attempts += 1
        chosen = rng.choice(weighted)
        if counts[chosen.id] >= duplicate_cap:
            continue
        counts[chosen.id] += 1
        added.append(chosen)
    if len(added) < count:
        added.extend(
            _sample_reward_cards(
                rng=rng,
                pool=pool,
                count=count - len(added),
                duplicate_cap=duplicate_cap,
            )
        )
    return added


def _remove_starter_cards(
    *,
    rng: random.Random,
    cards: list[DeckCardSpec],
    max_remove: int,
) -> list[DeckCardSpec]:
    if max_remove <= 0:
        return []
    remove_count = min(max_remove, _weighted_int(rng, [(0, 0.55), (1, 0.30), (2, 0.15)]))
    removed: list[DeckCardSpec] = []
    for _ in range(remove_count):
        candidates = [
            (idx, card)
            for idx, card in enumerate(cards)
            if card.id in {"CARD.STRIKE_IRONCLAD", "CARD.DEFEND_IRONCLAD"}
        ]
        if not candidates:
            break
        weighted: list[tuple[int, DeckCardSpec]] = []
        for idx, card in candidates:
            weight = 3 if card.id == "CARD.STRIKE_IRONCLAD" else 2
            weighted.extend([(idx, card)] * weight)
        idx, card = rng.choice(weighted)
        trial = cards[:idx] + cards[idx + 1:]
        if not _has_attack_and_block(trial):
            continue
        removed.append(cards.pop(idx))
    return removed


def _upgrade_cards(
    *,
    rng: random.Random,
    cards: list[DeckCardSpec],
    max_upgrades: int,
) -> list[DeckCardSpec]:
    if max_upgrades <= 0:
        return []
    if max_upgrades <= 2:
        choices = [(0, 0.50), (1, 0.35), (2, 0.15)]
    else:
        choices = [(0, 0.20), (1, 0.30), (2, 0.25), (3, 0.15), (4, 0.10)]
    upgrade_count = min(max_upgrades, _weighted_int(rng, choices))
    upgradable_indexes = [idx for idx, card in enumerate(cards) if not card.upgraded]
    rng.shuffle(upgradable_indexes)
    upgraded: list[DeckCardSpec] = []
    for idx in upgradable_indexes[:upgrade_count]:
        card = cards[idx]
        replacement = DeckCardSpec(
            id=card.id,
            upgraded=True,
            name=card.name,
            rarity=card.rarity,
            type=card.type,
        )
        cards[idx] = replacement
        upgraded.append(replacement)
    return upgraded


def _has_attack_and_block(cards: Iterable[DeckCardSpec]) -> bool:
    has_attack = False
    has_block = False
    for card in cards:
        if card.id in STARTER_ATTACKS or card.type.lower() == "attack":
            has_attack = True
        if card.id in STARTER_BLOCKS or card.type.lower() == "skill":
            has_block = True
    return has_attack and has_block


def _load_ironclad_reward_pool(
    *,
    codex: SpireCodex,
    allow_problematic_cards: bool,
) -> list[DeckCardSpec]:
    cards_file = os.path.join(codex.localization_path or "", "cards.json")
    if not os.path.isfile(cards_file):
        return _fallback_ironclad_pool(allow_problematic_cards=allow_problematic_cards)
    with open(cards_file, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    result: list[DeckCardSpec] = []
    for record in records:
        if not _is_legal_ironclad_reward_card(record, allow_problematic_cards):
            continue
        card_id = prefixed_card_id(str(record.get("id") or ""))
        result.append(
            DeckCardSpec(
                id=card_id,
                name=str(record.get("name") or bare_model_id(card_id, "CARD")),
                rarity=str(record.get("rarity_key") or record.get("rarity") or ""),
                type=str(record.get("type_key") or record.get("type") or ""),
            )
        )
    return sorted(result, key=lambda card: (card.rarity, card.id))


def _fallback_ironclad_pool(*, allow_problematic_cards: bool) -> list[DeckCardSpec]:
    safe = [
        ("POMMEL_STRIKE", "Pommel Strike", "Common", "Attack"),
        ("SHRUG_IT_OFF", "Shrug It Off", "Common", "Skill"),
        ("IRON_WAVE", "Iron Wave", "Common", "Attack"),
        ("ANGER", "Anger", "Common", "Attack"),
        ("THUNDERCLAP", "Thunderclap", "Common", "Attack"),
        ("BATTLE_TRANCE", "Battle Trance", "Uncommon", "Skill"),
        ("UPPERCUT", "Uppercut", "Uncommon", "Attack"),
        ("INFLAME", "Inflame", "Uncommon", "Power"),
        ("BLUDGEON", "Bludgeon", "Rare", "Attack"),
    ]
    return [
        DeckCardSpec(id=prefixed_card_id(card_id), name=name, rarity=rarity, type=typ)
        for card_id, name, rarity, typ in safe
        if allow_problematic_cards or prefixed_card_id(card_id) not in DEFAULT_EXCLUDED_CARD_IDS
    ]


def _is_legal_ironclad_reward_card(record: dict[str, Any], allow_problematic_cards: bool) -> bool:
    card_id = prefixed_card_id(str(record.get("id") or ""))
    if not allow_problematic_cards and card_id in DEFAULT_EXCLUDED_CARD_IDS:
        return False
    color = str(record.get("color") or "").lower()
    rarity = str(record.get("rarity_key") or record.get("rarity") or "").lower()
    card_type = str(record.get("type_key") or record.get("type") or "").lower()
    target = str(record.get("target") or "").lower()
    if color != "ironclad":
        return False
    if rarity not in {"common", "uncommon", "rare"}:
        return False
    if card_type in {"curse", "status"}:
        return False
    if record.get("is_x_cost") or record.get("is_x_star_cost") or record.get("star_cost") is not None:
        return False
    if target in {"anyally", "allallies"}:
        return False
    if not allow_problematic_cards:
        text = " ".join(
            str(record.get(key) or "")
            for key in ("description", "description_raw", "upgrade_description")
        ).lower()
        if "choose" in text or "transform" in text:
            return False
    return True


def _stable_seed(
    seed: int | str | None,
    worker_id: int | None,
    episode_id: int,
    mode: str,
) -> int:
    text = f"{seed if seed is not None else 0}|{worker_id if worker_id is not None else 0}|{episode_id}|{mode}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _weighted_choice(rng: random.Random, items: list[tuple[Any, float]]) -> Any:
    total = sum(max(0.0, weight) for _, weight in items)
    if total <= 0:
        return items[0][0]
    pick = rng.random() * total
    running = 0.0
    for value, weight in items:
        running += max(0.0, weight)
        if pick <= running:
            return value
    return items[-1][0]


def _weighted_int(rng: random.Random, items: list[tuple[int, float]]) -> int:
    return int(_weighted_choice(rng, items))
