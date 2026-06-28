"""Adapter for loading database files from ptrlrd/spire-codex."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class SpireCodex:
    """Interface to the game database extracted in the ptrlrd/spire-codex repository."""

    def __init__(self, localization_path: str | None = None) -> None:
        self.localization_path = localization_path or self._resolve_localization_path()
        self.card_ids: list[str] = []
        self.relic_ids: list[str] = []
        self.potion_ids: list[str] = []
        self.monster_ids: list[str] = []

        self.card_id_to_index: dict[str, int] = {}
        self.card_name_to_index: dict[str, int] = {}
        self.card_metadata: list[dict[str, Any]] = []

        self.relic_id_to_index: dict[str, int] = {}
        self.relic_name_to_index: dict[str, int] = {}

        self.potion_id_to_index: dict[str, int] = {}
        self.potion_name_to_index: dict[str, int] = {}

        self.monster_id_to_index: dict[str, int] = {}
        self.monster_name_to_index: dict[str, int] = {}

        self._load_codex()

    def _resolve_localization_path(self) -> str:
        # Check environment variable
        env_path = os.environ.get("SPIRE_CODEX_DATA_PATH")
        if env_path and os.path.isdir(env_path):
            return env_path

        # Candidate paths to spire-codex/data/eng
        candidates = [
            r"C:\dev\spire-codex\data\eng",
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "spire-codex",
                    "data",
                    "eng",
                )
            ),
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "spire-codex",
                    "data",
                    "eng",
                )
            ),
        ]
        for path in candidates:
            if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "cards.json")
            ):
                return path
        return ""

    def _load_codex(self) -> None:
        if not self.localization_path:
            logger.warning(
                "[SpireCodex] Localization path to spire-codex/data/eng not resolved."
            )
            return

        # 1. Cards
        cards_file = os.path.join(self.localization_path, "cards.json")
        if os.path.isfile(cards_file):
            try:
                with open(cards_file, "r", encoding="utf-8") as f:
                    cards_list = json.load(f)
                cards_list = sorted(cards_list, key=lambda c: c.get("id", ""))
                self.card_ids = [c.get("id", "") for c in cards_list]
                for idx, card in enumerate(cards_list):
                    cid = card.get("id", "")
                    name = card.get("name", "")
                    self.card_id_to_index[cid] = idx
                    self.card_id_to_index[cid.lower()] = idx
                    if name:
                        self.card_name_to_index[name.lower()] = idx

                    # Static metadata
                    cost = card.get("cost")
                    cost_val = float(cost) if cost is not None else 0.0
                    is_x = (
                        1.0
                        if card.get("is_x_cost") or card.get("is_x_star_cost")
                        else 0.0
                    )

                    card_type = str(card.get("type", "")).lower()
                    is_attack = 1.0 if card_type == "attack" else 0.0
                    is_skill = 1.0 if card_type == "skill" else 0.0
                    is_power = 1.0 if card_type == "power" else 0.0

                    self.card_metadata.append(
                        {
                            "cost": cost_val,
                            "is_x": is_x,
                            "is_attack": is_attack,
                            "is_skill": is_skill,
                            "is_power": is_power,
                        }
                    )
                logger.info(
                    f"[SpireCodex] Loaded {len(self.card_ids)} cards from {cards_file}."
                )
            except Exception as e:
                logger.error(f"[SpireCodex] Failed to load cards: {e}")

        # 2. Relics
        relics_file = os.path.join(self.localization_path, "relics.json")
        if os.path.isfile(relics_file):
            try:
                with open(relics_file, "r", encoding="utf-8") as f:
                    relics_list = json.load(f)
                relics_list = sorted(relics_list, key=lambda r: r.get("id", ""))
                self.relic_ids = [r.get("id", "") for r in relics_list]
                for idx, relic in enumerate(relics_list):
                    rid = relic.get("id", "")
                    name = relic.get("name", "")
                    self.relic_id_to_index[rid] = idx
                    self.relic_id_to_index[rid.lower()] = idx
                    if name:
                        self.relic_name_to_index[name.lower()] = idx
                logger.info(
                    f"[SpireCodex] Loaded {len(self.relic_ids)} relics from {relics_file}."
                )
            except Exception as e:
                logger.error(f"[SpireCodex] Failed to load relics: {e}")

        # 3. Potions
        potions_file = os.path.join(self.localization_path, "potions.json")
        if os.path.isfile(potions_file):
            try:
                with open(potions_file, "r", encoding="utf-8") as f:
                    potions_list = json.load(f)
                potions_list = sorted(potions_list, key=lambda p: p.get("id", ""))
                self.potion_ids = [p.get("id", "") for p in potions_list]
                for idx, potion in enumerate(potions_list):
                    pid = potion.get("id", "")
                    name = potion.get("name", "")
                    self.potion_id_to_index[pid] = idx
                    self.potion_id_to_index[pid.lower()] = idx
                    if name:
                        self.potion_name_to_index[name.lower()] = idx
                logger.info(
                    f"[SpireCodex] Loaded {len(self.potion_ids)} potions from {potions_file}."
                )
            except Exception as e:
                logger.error(f"[SpireCodex] Failed to load potions: {e}")

        # 4. Monsters
        monsters_file = os.path.join(self.localization_path, "monsters.json")
        if os.path.isfile(monsters_file):
            try:
                with open(monsters_file, "r", encoding="utf-8") as f:
                    monsters_list = json.load(f)
                monsters_list = sorted(monsters_list, key=lambda m: m.get("id", ""))
                self.monster_ids = [m.get("id", "") for m in monsters_list]
                for idx, monster in enumerate(monsters_list):
                    mid = monster.get("id", "")
                    name = monster.get("name", "")
                    self.monster_id_to_index[mid] = idx
                    self.monster_id_to_index[mid.lower()] = idx
                    if name:
                        self.monster_name_to_index[name.lower()] = idx
                logger.info(
                    f"[SpireCodex] Loaded {len(self.monster_ids)} monsters from {monsters_file}."
                )
            except Exception as e:
                logger.error(f"[SpireCodex] Failed to load monsters: {e}")

    def get_card_count(self) -> int:
        """Return total number of unique cards."""
        return len(self.card_ids) if self.card_ids else 577

    def get_relic_count(self) -> int:
        """Return total number of unique relics."""
        return len(self.relic_ids) if self.relic_ids else 296

    def get_potion_count(self) -> int:
        """Return total number of unique potions."""
        return len(self.potion_ids) if self.potion_ids else 63

    def get_monster_count(self) -> int:
        """Return total number of unique monsters."""
        return len(self.monster_ids) if self.monster_ids else 115

    def get_card_index(self, card_identifier: str | None) -> int | None:
        """Look up card index by card ID or name."""
        if not card_identifier:
            return None
        cleaned = card_identifier.strip().lower()
        if cleaned in self.card_id_to_index:
            return self.card_id_to_index[cleaned]
        if cleaned in self.card_name_to_index:
            return self.card_name_to_index[cleaned]
        norm = cleaned.replace(" ", "_").replace("-", "_")
        if norm in self.card_id_to_index:
            return self.card_id_to_index[norm]
        norm_space = cleaned.replace("_", " ")
        if norm_space in self.card_name_to_index:
            return self.card_name_to_index[norm_space]
        return None

    def get_relic_index(self, relic_identifier: str | None) -> int | None:
        """Look up relic index by relic ID or name."""
        if not relic_identifier:
            return None
        cleaned = relic_identifier.strip().lower()
        if cleaned in self.relic_id_to_index:
            return self.relic_id_to_index[cleaned]
        if cleaned in self.relic_name_to_index:
            return self.relic_name_to_index[cleaned]
        norm = cleaned.replace(" ", "_").replace("-", "_")
        if norm in self.relic_id_to_index:
            return self.relic_id_to_index[norm]
        norm_space = cleaned.replace("_", " ")
        if norm_space in self.relic_name_to_index:
            return self.relic_name_to_index[norm_space]
        return None

    def get_potion_index(self, potion_identifier: str | None) -> int | None:
        """Look up potion index by potion ID or name."""
        if not potion_identifier:
            return None
        cleaned = potion_identifier.strip().lower()
        if cleaned in self.potion_id_to_index:
            return self.potion_id_to_index[cleaned]
        if cleaned in self.potion_name_to_index:
            return self.potion_name_to_index[cleaned]
        norm = cleaned.replace(" ", "_").replace("-", "_")
        if norm in self.potion_id_to_index:
            return self.potion_id_to_index[norm]
        norm_space = cleaned.replace("_", " ")
        if norm_space in self.potion_name_to_index:
            return self.potion_name_to_index[norm_space]
        return None

    def get_monster_index(self, monster_identifier: str | None) -> int | None:
        """Look up monster index by monster ID or name."""
        if not monster_identifier:
            return None
        cleaned = monster_identifier.strip().lower()
        if cleaned in self.monster_id_to_index:
            return self.monster_id_to_index[cleaned]
        if cleaned in self.monster_name_to_index:
            return self.monster_name_to_index[cleaned]
        norm = cleaned.replace(" ", "_").replace("-", "_")
        if norm in self.monster_id_to_index:
            return self.monster_id_to_index[norm]
        norm_space = cleaned.replace("_", " ")
        if norm_space in self.monster_name_to_index:
            return self.monster_name_to_index[norm_space]
        return None

    def get_card_static_metadata(self, card_idx: int) -> dict[str, float]:
        """Return static features dictionary for card index."""
        if 0 <= card_idx < len(self.card_metadata):
            return self.card_metadata[card_idx]
        return {
            "cost": 0.0,
            "is_x": 0.0,
            "is_attack": 0.0,
            "is_skill": 0.0,
            "is_power": 0.0,
        }
