"""Adapter for loading database files from ptrlrd/spire-codex."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

RARITY_VALUES: dict[str, float] = {
    "basic": 0.1,
    "common": 0.2,
    "uncommon": 0.4,
    "rare": 0.6,
    "curse": 0.8,
    "status": 0.9,
}

TARGET_VALUES: dict[str, float] = {
    "none": 0.0,
    "self": 0.1,
    "anyenemy": 0.3,
    "allenemies": 0.4,
    "randomenemy": 0.5,
    "anyally": 0.6,
    "allallies": 0.7,
}


class SpireCodex:
    """Interface to the game database extracted in the ptrlrd/spire-codex repository.

    Args:
        localization_path: Explicit path to ``spire-codex/data/eng``.
            When *None*, resolved from ``SPIRE_CODEX_DATA_PATH`` env-var
            or well-known filesystem candidates.
        strict: When *True*, raise ``RuntimeError`` if the data path
            cannot be resolved, any entity file fails to load, or the
            loaded counts do not match ``EXPECTED_COUNTS``.  Use *True*
            for training; *False* (default) for unit-tests that may run
            without the full spire-codex checkout.
    """

    SCHEMA_VERSION = "sts2_codex_v1"

    EXPECTED_COUNTS: dict[str, int] = {
        "cards": 577,
        "relics": 296,
        "potions": 63,
        "monsters": 115,
    }

    # Fallback counts used only when strict=False and data is unavailable.
    _FALLBACK_COUNTS = EXPECTED_COUNTS.copy()

    def __init__(
        self,
        localization_path: str | None = None,
        *,
        strict: bool = False,
    ) -> None:
        self.strict = strict
        self.localization_path = localization_path or self._resolve_localization_path()

        self.card_ids: list[str] = []
        self.relic_ids: list[str] = []
        self.potion_ids: list[str] = []
        self.monster_ids: list[str] = []

        self.card_id_to_index: dict[str, int] = {}
        self.card_name_to_index: dict[str, int] = {}
        self.card_metadata: list[dict[str, float]] = []

        self.relic_id_to_index: dict[str, int] = {}
        self.relic_name_to_index: dict[str, int] = {}

        self.potion_id_to_index: dict[str, int] = {}
        self.potion_name_to_index: dict[str, int] = {}

        self.monster_id_to_index: dict[str, int] = {}
        self.monster_name_to_index: dict[str, int] = {}

        self._load_codex()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_localization_path(self) -> str:
        env_path = os.environ.get("SPIRE_CODEX_DATA_PATH")
        if env_path and os.path.isdir(env_path):
            return env_path

        candidates = [
            r"C:\dev\spire-codex\data\eng",
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "spire-codex", "data", "eng",
                )
            ),
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "..", "spire-codex", "data", "eng",
                )
            ),
        ]
        for path in candidates:
            if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "cards.json")
            ):
                return path

        if self.strict:
            raise RuntimeError(
                "[SpireCodex] STRICT: Could not resolve spire-codex data path. "
                "Set SPIRE_CODEX_DATA_PATH or place the repository at a known location."
            )
        return ""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_codex(self) -> None:
        if not self.localization_path:
            if self.strict:
                raise RuntimeError(
                    "[SpireCodex] STRICT: localization_path is empty — "
                    "cannot load entity databases."
                )
            logger.warning(
                "[SpireCodex] Localization path to spire-codex/data/eng not resolved."
            )
            return

        self._load_cards()
        self._load_relics()
        self._load_potions()
        self._load_monsters()

        if self.strict:
            self._validate_counts()

    def _load_cards(self) -> None:
        cards_file = os.path.join(self.localization_path, "cards.json")
        if not os.path.isfile(cards_file):
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: cards.json not found at {cards_file}")
            return
        try:
            with open(cards_file, "r", encoding="utf-8") as f:
                cards_list: list[dict[str, Any]] = json.load(f)
            cards_list = sorted(cards_list, key=lambda c: c.get("id", ""))
            self.card_ids = [c.get("id", "") for c in cards_list]
            for idx, card in enumerate(cards_list):
                cid = card.get("id", "")
                name = card.get("name", "")
                self.card_id_to_index[cid] = idx
                self.card_id_to_index[cid.lower()] = idx
                if name:
                    self.card_name_to_index[name.lower()] = idx

                # ---- Expanded static metadata ----
                cost = card.get("cost")
                cost_val = float(cost) if cost is not None else 0.0
                is_x = 1.0 if card.get("is_x_cost") or card.get("is_x_star_cost") else 0.0

                card_type = str(card.get("type_key") or card.get("type") or "").lower()
                is_attack = 1.0 if card_type == "attack" else 0.0
                is_skill = 1.0 if card_type == "skill" else 0.0
                is_power = 1.0 if card_type == "power" else 0.0

                rarity_key = str(card.get("rarity_key") or card.get("rarity") or "").lower()
                rarity_val = RARITY_VALUES.get(rarity_key, 0.0)

                target_key = str(card.get("target") or "").lower()
                target_val = TARGET_VALUES.get(target_key, 0.0)

                base_damage = float(card["damage"]) if card.get("damage") is not None else 0.0
                base_block = float(card["block"]) if card.get("block") is not None else 0.0

                # base_magic: first var that isn't Damage or Block
                base_magic = 0.0
                card_vars = card.get("vars") or {}
                for vk, vv in card_vars.items():
                    if vk.lower() not in ("damage", "block"):
                        try:
                            base_magic = float(vv)
                        except (TypeError, ValueError):
                            pass
                        break

                self.card_metadata.append({
                    "cost": cost_val,
                    "is_x": is_x,
                    "is_attack": is_attack,
                    "is_skill": is_skill,
                    "is_power": is_power,
                    "rarity": rarity_val,
                    "target": target_val,
                    "base_damage": base_damage,
                    "base_block": base_block,
                    "base_magic": base_magic,
                })
            logger.info(f"[SpireCodex] Loaded {len(self.card_ids)} cards.")
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: Failed to load cards: {e}") from e
            logger.error(f"[SpireCodex] Failed to load cards: {e}")

    def _load_relics(self) -> None:
        relics_file = os.path.join(self.localization_path, "relics.json")
        if not os.path.isfile(relics_file):
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: relics.json not found at {relics_file}")
            return
        try:
            with open(relics_file, "r", encoding="utf-8") as f:
                relics_list: list[dict[str, Any]] = json.load(f)
            relics_list = sorted(relics_list, key=lambda r: r.get("id", ""))
            self.relic_ids = [r.get("id", "") for r in relics_list]
            for idx, relic in enumerate(relics_list):
                rid = relic.get("id", "")
                name = relic.get("name", "")
                self.relic_id_to_index[rid] = idx
                self.relic_id_to_index[rid.lower()] = idx
                if name:
                    self.relic_name_to_index[name.lower()] = idx
            logger.info(f"[SpireCodex] Loaded {len(self.relic_ids)} relics.")
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: Failed to load relics: {e}") from e
            logger.error(f"[SpireCodex] Failed to load relics: {e}")

    def _load_potions(self) -> None:
        potions_file = os.path.join(self.localization_path, "potions.json")
        if not os.path.isfile(potions_file):
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: potions.json not found at {potions_file}")
            return
        try:
            with open(potions_file, "r", encoding="utf-8") as f:
                potions_list: list[dict[str, Any]] = json.load(f)
            potions_list = sorted(potions_list, key=lambda p: p.get("id", ""))
            self.potion_ids = [p.get("id", "") for p in potions_list]
            for idx, potion in enumerate(potions_list):
                pid = potion.get("id", "")
                name = potion.get("name", "")
                self.potion_id_to_index[pid] = idx
                self.potion_id_to_index[pid.lower()] = idx
                if name:
                    self.potion_name_to_index[name.lower()] = idx
            logger.info(f"[SpireCodex] Loaded {len(self.potion_ids)} potions.")
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: Failed to load potions: {e}") from e
            logger.error(f"[SpireCodex] Failed to load potions: {e}")

    def _load_monsters(self) -> None:
        monsters_file = os.path.join(self.localization_path, "monsters.json")
        if not os.path.isfile(monsters_file):
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: monsters.json not found at {monsters_file}")
            return
        try:
            with open(monsters_file, "r", encoding="utf-8") as f:
                monsters_list: list[dict[str, Any]] = json.load(f)
            monsters_list = sorted(monsters_list, key=lambda m: m.get("id", ""))
            self.monster_ids = [m.get("id", "") for m in monsters_list]
            for idx, monster in enumerate(monsters_list):
                mid = monster.get("id", "")
                name = monster.get("name", "")
                self.monster_id_to_index[mid] = idx
                self.monster_id_to_index[mid.lower()] = idx
                if name:
                    self.monster_name_to_index[name.lower()] = idx
            logger.info(f"[SpireCodex] Loaded {len(self.monster_ids)} monsters.")
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"[SpireCodex] STRICT: Failed to load monsters: {e}") from e
            logger.error(f"[SpireCodex] Failed to load monsters: {e}")

    def _validate_counts(self) -> None:
        actual = {
            "cards": len(self.card_ids),
            "relics": len(self.relic_ids),
            "potions": len(self.potion_ids),
            "monsters": len(self.monster_ids),
        }
        for entity, expected in self.EXPECTED_COUNTS.items():
            got = actual[entity]
            if got != expected:
                raise RuntimeError(
                    f"[SpireCodex] STRICT: {entity} count mismatch: "
                    f"expected {expected}, got {got}. "
                    f"Schema {self.SCHEMA_VERSION} requires exact counts."
                )

    # ------------------------------------------------------------------
    # Count accessors
    # ------------------------------------------------------------------

    def get_card_count(self) -> int:
        return len(self.card_ids) if self.card_ids else self._FALLBACK_COUNTS["cards"]

    def get_relic_count(self) -> int:
        return len(self.relic_ids) if self.relic_ids else self._FALLBACK_COUNTS["relics"]

    def get_potion_count(self) -> int:
        return len(self.potion_ids) if self.potion_ids else self._FALLBACK_COUNTS["potions"]

    def get_monster_count(self) -> int:
        return len(self.monster_ids) if self.monster_ids else self._FALLBACK_COUNTS["monsters"]

    # ------------------------------------------------------------------
    # Index lookups
    # ------------------------------------------------------------------

    def _lookup(
        self,
        identifier: str | None,
        id_map: dict[str, int],
        name_map: dict[str, int],
    ) -> int | None:
        if not identifier:
            return None
        cleaned = identifier.strip().lower()
        if cleaned in id_map:
            return id_map[cleaned]
        if cleaned in name_map:
            return name_map[cleaned]
        norm = cleaned.replace(" ", "_").replace("-", "_")
        if norm in id_map:
            return id_map[norm]
        norm_space = cleaned.replace("_", " ")
        if norm_space in name_map:
            return name_map[norm_space]
        return None

    def get_card_index(self, card_identifier: str | None) -> int | None:
        return self._lookup(card_identifier, self.card_id_to_index, self.card_name_to_index)

    def get_relic_index(self, relic_identifier: str | None) -> int | None:
        return self._lookup(relic_identifier, self.relic_id_to_index, self.relic_name_to_index)

    def get_potion_index(self, potion_identifier: str | None) -> int | None:
        return self._lookup(potion_identifier, self.potion_id_to_index, self.potion_name_to_index)

    def get_monster_index(self, monster_identifier: str | None) -> int | None:
        return self._lookup(monster_identifier, self.monster_id_to_index, self.monster_name_to_index)

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------

    _EMPTY_CARD_META: dict[str, float] = {
        "cost": 0.0, "is_x": 0.0,
        "is_attack": 0.0, "is_skill": 0.0, "is_power": 0.0,
        "rarity": 0.0, "target": 0.0,
        "base_damage": 0.0, "base_block": 0.0, "base_magic": 0.0,
    }

    def get_card_static_metadata(self, card_idx: int) -> dict[str, float]:
        """Return static features dictionary for card index."""
        if 0 <= card_idx < len(self.card_metadata):
            return self.card_metadata[card_idx]
        return self._EMPTY_CARD_META.copy()
