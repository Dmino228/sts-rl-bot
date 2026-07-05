from __future__ import annotations

import random

import pytest

from sts2.curriculum_profiles import (
    curriculum_profile,
    parse_curriculum_mix,
    sample_curriculum_profile,
)


def test_parse_curriculum_mix_validates_and_normalizes_weights() -> None:
    entries = parse_curriculum_mix("c0_the_kin_exact:0.8,c1_the_kin_5_decks:0.2")

    assert entries == [("c0_the_kin_exact", 0.8), ("c1_the_kin_5_decks", 0.2)]


def test_parse_curriculum_mix_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown STS2 curriculum profile"):
        parse_curriculum_mix("typo_profile:1.0")


def test_c0_profile_is_exact_the_kin_overfit() -> None:
    profile = curriculum_profile("c0_the_kin_exact", rng=random.Random(1))

    assert profile.options["combat_encounter"] == "THE_KIN_BOSS"
    assert profile.options["deck_mode"] == "fixed_the_kin_overfit"
    assert profile.options["deck_seed"] == 20260705
    assert profile.options["run_seed"] == 20260705


def test_sample_curriculum_profile_uses_weighted_profile_names() -> None:
    profile = sample_curriculum_profile(
        "c2_the_kin_random_safe:1.0",
        rng=random.Random(123),
    )

    assert profile is not None
    assert profile.name == "c2_the_kin_random_safe"
