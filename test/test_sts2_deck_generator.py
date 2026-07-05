from __future__ import annotations

from collections import Counter

from sts2.deck_generator import (
    IRONCLAD_STARTER_DECK,
    build_combat_deck_spec,
    bare_model_id,
)
from sts2.spire_codex import SpireCodex


def test_starter_mode_produces_exact_ironclad_starter_deck() -> None:
    spec = build_combat_deck_spec(mode="starter", seed=123, worker_id=1, episode_id=1)

    assert [card.id for card in spec.cards] == list(IRONCLAD_STARTER_DECK)
    assert all(not card.upgraded for card in spec.cards)
    assert spec.apply_to_headless is False
    assert spec.source == "sts2_start_run_default"


def test_random_synthetic_usually_contains_non_starter_cards() -> None:
    non_starter_samples = 0
    starter_ids = set(IRONCLAD_STARTER_DECK)
    for episode in range(20):
        spec = build_combat_deck_spec(
            mode="random_synthetic",
            seed=123,
            worker_id=2,
            episode_id=episode,
        )
        if any(card.id not in starter_ids for card in spec.cards):
            non_starter_samples += 1

    assert non_starter_samples >= 18


def test_generated_card_ids_are_valid_in_spire_codex() -> None:
    codex = SpireCodex()
    spec = build_combat_deck_spec(
        mode="random_synthetic",
        seed=77,
        worker_id=3,
        episode_id=4,
        codex=codex,
    )

    for card in spec.cards:
        assert codex.get_card_index(bare_model_id(card.id, "CARD")) is not None


def test_duplicate_cap_is_respected_for_non_starter_additions() -> None:
    spec = build_combat_deck_spec(
        mode="random_synthetic",
        seed=7,
        worker_id=1,
        episode_id=1,
        duplicate_cap=1,
    )
    added_counts = Counter(card.id for card in spec.added_cards)

    assert added_counts
    assert max(added_counts.values()) <= 1


def test_same_seed_worker_episode_produces_same_deck() -> None:
    a = build_combat_deck_spec(
        mode="random_synthetic",
        seed=123,
        worker_id=1,
        episode_id=9,
    )
    b = build_combat_deck_spec(
        mode="random_synthetic",
        seed=123,
        worker_id=1,
        episode_id=9,
    )

    assert a.to_debug() == b.to_debug()


def test_different_episode_changes_random_synthetic_deck() -> None:
    a = build_combat_deck_spec(
        mode="random_synthetic",
        seed=123,
        worker_id=1,
        episode_id=9,
    )
    b = build_combat_deck_spec(
        mode="random_synthetic",
        seed=123,
        worker_id=1,
        episode_id=10,
    )

    assert [card.to_debug() for card in a.cards] != [card.to_debug() for card in b.cards]


def test_random_floor_bucket_logs_bucket_and_floor() -> None:
    spec = build_combat_deck_spec(
        mode="random_act1_floor_bucket",
        seed=123,
        worker_id=1,
        episode_id=10,
    )

    assert spec.floor_bucket in {"early", "mid", "late"}
    assert spec.synthetic_floor is not None
    assert spec.apply_to_headless is True
    assert 1 <= spec.synthetic_floor <= 16


def test_random_boss_synthetic_safe_adds_stronger_resources() -> None:
    spec = build_combat_deck_spec(
        mode="random_boss_synthetic_safe",
        seed=123,
        worker_id=1,
        episode_id=5,
    )
    starter_ids = set(IRONCLAD_STARTER_DECK)

    assert spec.apply_to_headless is True
    assert spec.source == "synthetic_boss_safe_ironclad"
    assert len(spec.cards) > len(IRONCLAD_STARTER_DECK)
    assert any(card.id not in starter_ids for card in spec.cards)
    assert len(spec.relics) >= 2
    assert spec.potions
    assert spec.hp >= 68


def test_fixed_the_kin_overfit_deck_is_exact_and_seed_independent() -> None:
    a = build_combat_deck_spec(
        mode="fixed_the_kin_overfit",
        seed=1,
        worker_id=1,
        episode_id=1,
    )
    b = build_combat_deck_spec(
        mode="fixed_the_kin_overfit",
        seed=999,
        worker_id=8,
        episode_id=500,
    )
    codex = SpireCodex()

    assert a.to_debug() == b.to_debug()
    assert a.apply_to_headless is True
    assert a.source == "fixed_the_kin_overfit_v1"
    assert a.hp == 80
    assert a.max_hp == 80
    assert a.relics == (
        "RELIC.BURNING_BLOOD",
        "RELIC.ANCHOR",
        "RELIC.VAJRA",
        "RELIC.ODDLY_SMOOTH_STONE",
        "RELIC.BAG_OF_PREPARATION",
    )
    assert a.potions == ("POTION.STRENGTH_POTION",)
    assert len(a.cards) == 18
    assert sum(card.upgraded for card in a.cards) == 8
    for card in a.cards:
        assert codex.get_card_index(bare_model_id(card.id, "CARD")) is not None
