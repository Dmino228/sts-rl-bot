import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import SlayTheSpireEnv
from rllib.env_wrapper import select_character
from sts2.action_space import (
    BACK_ACTION,
    CHOICE_BASE,
    END_TURN_ACTION,
    StS2ActionMapper,
    StS2ActionMasker,
    TARGETED_PLAY_BASE,
)
from sts2.io import StS2StdIOOverlay
from sts2.process_manager import StS2CliProcessManager
from sts2.state_encoder import StS2StateEncoder, normalize_sts2_state


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


def test_sts2_state_adapter_and_encoder_fill_legacy_env_shape():
    state = normalize_sts2_state(_combat_decision())
    encoded = StS2StateEncoder().encode(state)

    assert state["in_game"] is True
    assert state["available_commands"] == ["play", "end", "state", "potion"]
    assert state["game_state"]["screen_type"] == "NONE"
    assert state["game_state"]["combat_state"]["monsters"][0]["current_hp"] == 30
    assert encoded.shape == (7231,)
    assert np.all(encoded >= -1.0)
    assert np.all(encoded <= 1.0)


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
    assert obs.shape == (7231,)
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


def test_select_character_uses_sts2_roster_for_multi_character():
    assert select_character(3, {"multi_character": True}, "sts2") == "Necrobinder"


def test_sts2_state_encoder_detailed_encoding():
    state = normalize_sts2_state(_combat_decision())
    # Set explicit name and powers to test lookup and power logic
    state["enemies"][0]["name"] = "Aeonglass"
    state["enemies"][0]["powers"] = [
        {"name": "Strength", "amount": 3},
        {"name": "Vulnerable", "amount": 2}
    ]
    encoder = StS2StateEncoder()
    encoded = encoder.encode(state)

    # Check shape
    assert encoded.shape == encoder.observation_space.shape
    assert encoded.shape == (7231,)

    # 1. Base 205 legacy features tests:
    # Gold is 12 (slot 11 = 12/1000 = 0.012)
    assert abs(encoded[11] - 0.012) < 1e-4
    # Player HP = 70, Max HP = 80 (slot 12 = 70/80 = 0.875)
    assert abs(encoded[12] - 0.875) < 1e-4
    # Player block = 3 (slot 16 = 3/100 = 0.03)
    assert abs(encoded[16] - 0.03) < 1e-4

    # Hand cards dynamic features (offset 40):
    # Slot 0 has Strike (cost=1, can_play=True, type=Attack, target=AnyEnemy, damage=6)
    assert encoded[40] == 1.0
    assert abs(encoded[41] - 0.2) < 1e-4
    assert encoded[42] == 1.0
    assert abs(encoded[43] - 0.1) < 1e-4
    assert abs(encoded[44] - 0.3) < 1e-4
    assert abs(encoded[45] - 0.06) < 1e-4

    # 2. Append Hand Cards Semantic Profiles (Offset 205)
    # Slot 0 card: Strike (ID: "STRIKE" or maps to Strike in codex)
    strike_idx = encoder.codex.get_card_index("Strike")
    assert strike_idx is not None
    assert encoded[205 + strike_idx] == 1.0
    # Metadata for Strike (Attack, cost 1, not X)
    assert abs(encoded[205 + encoder.card_count + 0] - 0.2) < 1e-4  # cost / 5.0 = 0.2
    assert encoded[205 + encoder.card_count + 1] == 0.0  # is_x = 0
    assert encoded[205 + encoder.card_count + 2] == 1.0  # is_attack = 1.0
    assert encoded[205 + encoder.card_count + 3] == 0.0  # is_skill = 0.0
    assert encoded[205 + encoder.card_count + 4] == 0.0  # is_power = 0.0

    # 3. Append Relics (relics_offset = 205 + 10 * 582 = 6025)
    # _combat_decision has relic "Burning Blood"
    burning_blood_idx = encoder.codex.get_relic_index("Burning Blood")
    assert burning_blood_idx is not None
    assert encoded[6025 + burning_blood_idx] == 1.0

    # 4. Append Potions (potions_offset = 6025 + 296 = 6321)
    # Potion slot 0: Block Potion
    block_potion_idx = encoder.codex.get_potion_index("Block Potion")
    assert block_potion_idx is not None
    assert encoded[6321 + block_potion_idx] == 1.0

    # 5. Append Monsters (monsters_offset = 6321 + 315 = 6636)
    # Enemy 0 name: "Aeonglass"
    monster_idx = encoder.codex.get_monster_index("Aeonglass")
    assert monster_idx is not None
    assert encoded[6636 + monster_idx] == 1.0
    # Active powers: Strength=3 (0.3), Vulnerable=2 (0.4)
    assert abs(encoded[6636 + encoder.monster_count + 0] - 0.3) < 1e-4
    assert abs(encoded[6636 + encoder.monster_count + 1] - 0.4) < 1e-4
    assert encoded[6636 + encoder.monster_count + 2] == 0.0
    assert encoded[6636 + encoder.monster_count + 3] == 0.0
