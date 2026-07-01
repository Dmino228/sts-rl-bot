import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import SlayTheSpireEnv
from rllib.env_wrapper import select_character
from sts2.action_space import (
    BACK_ACTION,
    CHOICE_BASE,
    END_TURN_ACTION,
    PLAY_CARD_BASE,
    POTION_BASE,
    StS2ActionMapper,
    StS2ActionMasker,
    TARGETED_PLAY_BASE,
)
from sts2.io import StS2StdIOOverlay
from sts2.process_manager import StS2CliProcessManager
from sts2.spire_codex import SpireCodex
from sts2.state_encoder import StS2StateEncoder, StS2StateEncoderFlat, normalize_sts2_state


class FakeStS2ProcessManager:
    auto_launch = False

    def __init__(self, states):
        self.states = list(states)
        self.sent = []
        self._proc = object()

    def launch_game(self):
        raise AssertionError("Fake manager should not launch a process")

    def signal_ready(self):
        raise AssertionError("Fake manager should not wait for ready")

    def read_state(self):
        if not self.states:
            raise RuntimeError("No fake STS2 states left")
        return self.states.pop(0)

    def send_command(self, command):
        self.sent.append(command)

    def is_process_alive(self):
        return True

    def stop(self):
        pass

    def terminate(self):
        pass


class FakeLiveProcess:
    pid = 4242

    def poll(self):
        return None


def _map_decision():
    return {
        "type": "decision",
        "decision": "map_select",
        "context": {"act": 1, "floor": 0},
        "choices": [{"col": 2, "row": 0, "type": "Monster"}],
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": [], "relics": []},
    }


def _combat_decision():
    return {
        "type": "decision",
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1},
        "energy": 3,
        "max_energy": 3,
        "hand": [
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "Self",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
        "enemies": [
            {"index": 0, "hp": 30, "max_hp": 30, "intents": [{"type": "Attack", "damage": 5}]},
            {"index": 1, "hp": 25, "max_hp": 25, "intents": [{"type": "Buff"}]},
        ],
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": 3,
            "gold": 12,
            "deck_size": 10,
            "deck": [{"name": "Strike", "upgraded": True}],
            "relics": [{"name": "Burning Blood"}],
            "potions": [{"index": 0, "name": "Block Potion", "target_type": "Self"}],
        },
    }


def test_stdio_overlay_serializes_json_commands():
    overlay = StS2StdIOOverlay()

    assert (
        overlay.encode_command({"cmd": "action", "action": "end_turn"})
        == '{"cmd":"action","action":"end_turn"}\n'
    )
    assert overlay.encode_command('{"cmd":"quit"}') == '{"cmd":"quit"}\n'


def test_sts2_action_mask_and_mapper_use_decision_json():
    state = normalize_sts2_state(_combat_decision())
    mask = StS2ActionMasker().get_mask(state)
    mapper = StS2ActionMapper()

    assert mask[TARGETED_PLAY_BASE] == 1
    assert mask[TARGETED_PLAY_BASE + 1] == 1
    assert mask[1] == 1
    assert mask[END_TURN_ACTION] == 1

    command = mapper.get_action_string(TARGETED_PLAY_BASE + 1, state)
    assert command == {
        "cmd": "action",
        "action": "play_card",
        "args": {"card_index": 0, "target_index": 1},
    }


def test_sts2_map_choice_maps_to_select_map_node():
    state = normalize_sts2_state(_map_decision())

    assert StS2ActionMasker().get_mask(state)[CHOICE_BASE] == 1
    assert StS2ActionMapper().get_action_string(CHOICE_BASE, state) == {
        "cmd": "action",
        "action": "select_map_node",
        "args": {"col": 2, "row": 0},
    }


def test_sts2_card_reward_skip_uses_json_action():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "card_reward",
            "cards": [{"index": 0, "name": "Strike"}],
            "can_skip": True,
            "player": {"hp": 70, "max_hp": 80},
        }
    )

    assert StS2ActionMasker().get_mask(state)[BACK_ACTION] == 1
    assert StS2ActionMapper().get_action_string(BACK_ACTION, state) == {
        "cmd": "action",
        "action": "skip_card_reward",
    }


def test_sts2_card_select_fills_selection_atomically():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "card_select",
            "cards": [
                {"index": 0, "name": "Strike"},
                {"index": 1, "name": "Defend"},
                {"index": 2, "name": "Bash"},
            ],
            "min_select": 1,
            "max_select": 2,
            "player": {"hp": 70, "max_hp": 80},
        }
    )

    command = StS2ActionMapper().get_action_string(CHOICE_BASE + 2, state)

    assert command == {
        "cmd": "action",
        "action": "select_cards",
        "args": {"indices": "2,0"},
    }


def test_sts2_card_select_optional_keeps_skip_action():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "card_select",
            "cards": [{"index": 0, "name": "Strike"}],
            "min_select": 0,
            "max_select": 1,
            "player": {"hp": 70, "max_hp": 80},
        }
    )

    mask = StS2ActionMasker().get_mask(state)

    assert mask[CHOICE_BASE] == 1
    assert mask[BACK_ACTION] == 1
    assert StS2ActionMapper().get_action_string(BACK_ACTION, state) == {
        "cmd": "action",
        "action": "skip_select",
    }


def test_sts2_combat_card_select_selects_when_reached():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "card_select",
            "context": {"act": 1, "floor": 5, "room_type": "Monster"},
            "cards": [
                {"index": 0, "name": "Anger"},
                {"index": 1, "name": "Shrug It Off"},
                {"index": 2, "name": "Pommel Strike"},
            ],
            "min_select": 1,
            "max_select": 1,
            "player": {"hp": 70, "max_hp": 80},
        }
    )

    mapper = StS2ActionMapper()
    mask = StS2ActionMasker().get_mask(state)

    assert mask[CHOICE_BASE + 2] == 1
    assert mask[BACK_ACTION] == 0
    assert mapper.get_action_string(CHOICE_BASE + 2, state) == {
        "cmd": "action",
        "action": "select_cards",
        "args": {"indices": "2"},
    }


def test_sts2_masks_combat_cards_that_spawn_card_select():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 5, "room_type": "Monster"},
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.SEEKER_STRIKE",
                    "name": "Seeker Strike",
                    "cost": 1,
                    "type": "Attack",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                    "description": "Deal 9 damage. Choose 1 of 3 cards in your Draw Pile.",
                },
                {
                    "index": 1,
                    "id": "CARD.ARMAMENTS",
                    "name": "Armaments",
                    "cost": 1,
                    "type": "Skill",
                    "target_type": "Self",
                    "can_play": True,
                    "description": "Gain Block. Upgrade a card in your Hand.",
                },
            ],
            "enemies": [{"index": 0, "hp": 30, "max_hp": 30}],
            "player": {"hp": 70, "max_hp": 80},
        }
    )

    mask = StS2ActionMasker().get_mask(state)

    assert mask[TARGETED_PLAY_BASE] == 0
    assert mask[PLAY_CARD_BASE + 1] == 0
    assert mask[END_TURN_ACTION] == 1


@pytest.mark.parametrize(
    ("card_id", "name", "description", "action_id"),
    [
        (
            "CARD.SCULPTING_STRIKE",
            "Sculpting Strike",
            "Deal 9 damage. Add Ethereal to a card in your Hand.",
            TARGETED_PLAY_BASE,
        ),
        (
            "CARD.HIDDEN_DAGGERS",
            "Hidden Daggers",
            "Discard 2 cards. Add 2 Shivs into your Hand.",
            PLAY_CARD_BASE,
        ),
        (
            "CARD.SURVIVOR",
            "Survivor",
            "Gain 8 Block. Discard 1 card.",
            PLAY_CARD_BASE,
        ),
        (
            "CARD.BODYGUARD",
            "Bodyguard",
            "Summon 5.",
            PLAY_CARD_BASE,
        ),
    ],
)
def test_sts2_masks_multi_character_cards_that_spawn_card_select(
    card_id,
    name,
    description,
    action_id,
):
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 5, "room_type": "Monster"},
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": card_id,
                    "name": name,
                    "cost": 1,
                    "type": "Attack" if action_id == TARGETED_PLAY_BASE else "Skill",
                    "target_type": "AnyEnemy" if action_id == TARGETED_PLAY_BASE else "Self",
                    "can_play": True,
                    "description": description,
                }
            ],
            "enemies": [{"index": 0, "hp": 30, "max_hp": 30}],
            "player": {"hp": 70, "max_hp": 80},
        }
    )

    mask = StS2ActionMasker().get_mask(state)

    assert mask[action_id] == 0
    assert mask[END_TURN_ACTION] == 1


def test_sts2_masks_potions_that_spawn_card_select():
    state = normalize_sts2_state(
        {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 5, "room_type": "Monster"},
            "energy": 3,
            "max_energy": 3,
            "hand": [],
            "enemies": [{"index": 0, "hp": 30, "max_hp": 30}],
            "player": {
                "hp": 70,
                "max_hp": 80,
                "potions": [
                    {
                        "index": 0,
                        "name": "Skill Potion",
                        "target_type": "Self",
                        "description": "Choose 1 of 3 random Skill cards.",
                    },
                    {
                        "index": 1,
                        "name": "Block Potion",
                        "target_type": "Self",
                        "description": "Gain Block.",
                    },
                ],
            },
        }
    )

    mask = StS2ActionMasker().get_mask(state)

    assert mask[POTION_BASE] == 0
    assert mask[POTION_BASE + 1] == 1


# ---------------------------------------------------------------------------
# Compact Encoder Tests
# ---------------------------------------------------------------------------

def test_sts2_compact_encoder_shape():
    state = normalize_sts2_state(_combat_decision())
    encoded = StS2StateEncoder().encode(state)

    assert encoded.shape == (349,)
    assert np.all(encoded >= -1.0)
    assert np.all(encoded <= 1.0)


def test_sts2_compact_encoder_schema_version():
    assert StS2StateEncoder.SCHEMA_VERSION == "sts2_compact_v1"


def test_sts2_compact_encoder_detailed_encoding():
    state = normalize_sts2_state(_combat_decision())
    encoder = StS2StateEncoder()
    encoded = encoder.encode(state)

    assert encoded.shape == (349,)

    # Screen type: NONE → index 0
    assert encoded[0] == 1.0
    assert encoded[1] == 0.0

    # Global scalars (ptr=9):
    # act=1 → 1/4=0.25
    assert abs(encoded[9] - 0.25) < 1e-4
    # gold=12 → 12/1000=0.012
    assert abs(encoded[11] - 0.012) < 1e-4
    # hp_ratio = 70/80 = 0.875
    assert abs(encoded[12] - 0.875) < 1e-4
    # energy = 3 → 3/10 = 0.3
    assert abs(encoded[15] - 0.3) < 1e-4
    # block = 3 → 3/100 = 0.03
    assert abs(encoded[16] - 0.03) < 1e-4

    # Potion summary (ptr=23): 1 potion of 5 slots → 0.2
    assert abs(encoded[23] - 0.2) < 1e-4

    # Hand card slot 0 (ptr=24, 15 features per slot):
    # is_present
    assert encoded[24] == 1.0
    # cost = 1 → 1/5 = 0.2
    assert abs(encoded[25] - 0.2) < 1e-4
    # can_play = True → 1.0
    assert encoded[26] == 1.0
    # type flags: Attack → [1,0,0,0,0]
    assert encoded[27] == 1.0  # is_attack
    assert encoded[28] == 0.0  # is_skill
    assert encoded[29] == 0.0  # is_power
    # target flags: AnyEnemy → off+9 = 1.0
    assert encoded[33] == 1.0  # anyenemy
    # damage = 6 → 6/100 = 0.06
    assert abs(encoded[36] - 0.06) < 1e-4

    # Hand card slot 1 (ptr=24+15=39):
    assert encoded[39] == 1.0  # is_present
    assert encoded[41] == 1.0  # can_play
    assert encoded[42] == 0.0  # NOT attack
    assert encoded[43] == 1.0  # IS skill
    # block = 5 → 5/100 = 0.05
    assert abs(encoded[52] - 0.05) < 1e-4

    # Enemy slot 0 (ptr=174, 14 features per slot):
    assert encoded[174] == 1.0  # is_present
    # hp_ratio = 30/30 = 1.0
    assert abs(encoded[175] - 1.0) < 1e-4
    # hp_norm = 30/300 = 0.1
    assert abs(encoded[176] - 0.1) < 1e-4
    # Intent type one-hot: "Attack" → INTENT_LABELS index 1
    assert encoded[177 + 1] == 1.0  # attack flag

    # Enemy slot 1 (ptr=174+14=188):
    assert encoded[188] == 1.0  # is_present
    # Intent type: "Buff" → INTENT_LABELS index 5
    assert encoded[191 + 5] == 1.0  # buff flag


def test_env_reset_starts_sts2_run_with_json_start_command():
    env = SlayTheSpireEnv(
        game_version=2,
        sts2_cli_path="fake-sts2-cli",
        sts2_ascension=7,
        sts2_lang="en",
    )
    fake_manager = FakeStS2ProcessManager([_map_decision()])
    env.process_manager = fake_manager

    obs, info = env.reset(seed=123)

    assert fake_manager.sent == [
        {
            "cmd": "start_run",
            "character": "Ironclad",
            "ascension": 7,
            "lang": "en",
            "seed": "123",
        }
    ]
    assert obs.shape == (349,)
    assert info["action_mask"][CHOICE_BASE] == 1
    env.close()


def test_sts2_process_manager_infers_cwd_from_csproj_global_json(tmp_path):
    repo = tmp_path / "sts2-cli"
    project_dir = repo / "src" / "Sts2Headless"
    project_dir.mkdir(parents=True)
    (repo / "global.json").write_text("{}", encoding="utf-8")
    project = project_dir / "Sts2Headless.csproj"
    project.write_text("<Project />", encoding="utf-8")

    manager = StS2CliProcessManager(
        worker_dir=str(tmp_path / "worker"),
        cli_path="dotnet",
        cli_args=["run", "--project", str(project)],
    )

    assert manager._resolve_process_cwd() == str(repo)


def test_sts2_process_manager_timeout_message_includes_last_command(tmp_path):
    manager = StS2CliProcessManager(
        timeout=3.0,
        worker_dir=str(tmp_path / "sts2_worker_7"),
        cli_path="fake-sts2-cli",
    )
    command = {"cmd": "action", "action": "end_turn"}
    manager._last_command = command
    manager._last_command_at = 100.0
    manager._last_state = {
        "type": "decision",
        "decision": "card_select",
        "min_select": 1,
        "max_select": 2,
        "cards": [{"index": 0, "name": "Strike", "type": "Attack"}, {"index": 1}],
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
    }

    message = manager._timeout_message("No sts2-cli JSON state received")

    assert "within 3.0s" in message
    assert "worker=sts2_worker_7" in message
    assert "last_command={'cmd': 'action', 'action': 'end_turn'}" in message
    assert "'decision': 'card_select'" in message
    assert "'max_select': 2" in message
    assert "'cards_count': 2" in message
    assert "'room_type': 'Monster'" in message
    assert "'name': 'Strike'" in message


def test_sts2_process_manager_recycle_reason_after_episode_limit(tmp_path):
    manager = StS2CliProcessManager(
        worker_dir=str(tmp_path / "sts2_worker_2"),
        cli_path="fake-sts2-cli",
        recycle_every_episodes=2,
    )
    manager._proc = FakeLiveProcess()

    manager.record_run_started()
    assert manager._recycle_reason() is None

    manager.record_run_started()
    reason = manager._recycle_reason()

    assert reason is not None
    assert "episodes=2" in reason
    assert "limit=2" in reason


def test_sts2_process_manager_recycle_reason_after_rss_limit(tmp_path, monkeypatch):
    manager = StS2CliProcessManager(
        worker_dir=str(tmp_path / "sts2_worker_3"),
        cli_path="fake-sts2-cli",
        recycle_rss_mb=768.0,
    )
    manager._proc = FakeLiveProcess()
    monkeypatch.setattr(manager, "current_rss_mb", lambda: 812.25)

    reason = manager._recycle_reason()
    snapshot = manager.diagnostic_snapshot()

    assert reason is not None
    assert "rss_mb=812.2" in reason
    assert "limit=768.0" in reason
    assert snapshot["rss_mb"] == 812.2
    assert snapshot["recycle_limits"]["rss_mb"] == 768.0


def test_select_character_uses_sts2_roster_for_multi_character():
    assert select_character(3, {"multi_character": True}, "sts2") == "Necrobinder"


# ---------------------------------------------------------------------------
# SpireCodex Schema & Strict Mode Tests
# ---------------------------------------------------------------------------

def test_spire_codex_schema_version_constant():
    assert SpireCodex.SCHEMA_VERSION == "sts2_codex_v1"


def test_spire_codex_loads_all_entities():
    codex = SpireCodex()
    assert len(codex.card_ids) > 0, "card_ids must not be empty"
    assert len(codex.relic_ids) > 0, "relic_ids must not be empty"
    assert len(codex.potion_ids) > 0, "potion_ids must not be empty"
    assert len(codex.monster_ids) > 0, "monster_ids must not be empty"

    for entity, expected in SpireCodex.EXPECTED_COUNTS.items():
        actual = {
            "cards": len(codex.card_ids),
            "relics": len(codex.relic_ids),
            "potions": len(codex.potion_ids),
            "monsters": len(codex.monster_ids),
        }[entity]
        assert actual == expected, f"{entity}: expected {expected}, got {actual}"


def test_spire_codex_strict_fails_without_path(tmp_path):
    bogus_path = str(tmp_path / "nonexistent" / "path")
    with pytest.raises(RuntimeError, match="STRICT"):
        SpireCodex(localization_path=bogus_path, strict=True)


def test_spire_codex_expanded_card_metadata():
    codex = SpireCodex()
    strike_idx = codex.get_card_index("Strike")
    assert strike_idx is not None
    meta = codex.get_card_static_metadata(strike_idx)
    assert meta["is_attack"] == 1.0
    assert meta["is_skill"] == 0.0
    assert meta["cost"] == 1.0
    assert meta["rarity"] > 0.0
    assert meta["base_damage"] > 0.0
    assert "target" in meta
    assert "base_block" in meta
    assert "base_magic" in meta


# ---------------------------------------------------------------------------
# Flat Encoder Tests (experimental, 7231D)
# ---------------------------------------------------------------------------

def test_sts2_flat_encoder_schema_version():
    assert StS2StateEncoderFlat.SCHEMA_VERSION == "sts2_codex_flat_v1"


def test_sts2_flat_encoder_shape():
    encoder = StS2StateEncoderFlat()
    assert encoder.shape == (7231,)

    state = normalize_sts2_state(_combat_decision())
    encoded = encoder.encode(state)
    assert encoded.shape == (7231,)
    assert np.all(encoded >= -1.0)
    assert np.all(encoded <= 1.0)
