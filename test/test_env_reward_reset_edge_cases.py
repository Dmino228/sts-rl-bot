import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import (
    ACT_COMPLETION_REWARD,
    DEATH_PENALTY,
    FLOOR_REWARD,
    SlayTheSpireEnv,
)


class FakeProcessManager:
    def __init__(self, states=None):
        self.states = list(states or [])
        self.sent = []

    def read_state(self):
        if not self.states:
            raise RuntimeError("No fake states left to read")
        return self.states.pop(0)

    def send_command(self, command):
        self.sent.append(command)

    def stop(self):
        pass


def _deck(size=10, upgrades=0):
    return [{"id": f"Card{i}", "upgrades": upgrades} for i in range(size)]


def _relic(relic_id):
    return {"id": relic_id, "name": relic_id}


def _map_state(act, floor, hp=80, include_return=False):
    commands = ["choose", "key", "click", "wait", "state"]
    if include_return:
        commands.insert(1, "return")
    return {
        "in_game": True,
        "available_commands": commands,
        "game_state": {
            "screen_type": "MAP",
            "floor": floor,
            "act": act,
            "current_hp": hp,
            "max_hp": 80,
            "relics": [_relic("Burning Blood")],
            "deck": _deck(),
            "screen_state": {"next_nodes": [{"x": 1, "y": 0}]},
        },
    }


def _combat_state(act, floor, hp=80, monster_hp=40):
    return {
        "in_game": True,
        "available_commands": ["play", "end", "key", "click", "wait", "state"],
        "game_state": {
            "screen_type": "NONE",
            "floor": floor,
            "act": act,
            "current_hp": hp,
            "max_hp": 80,
            "relics": [_relic("Burning Blood")],
            "deck": _deck(),
            "combat_state": {
                "player": {"current_hp": hp, "max_hp": 80, "energy": 3},
                "hand": [],
                "monsters": [
                    {
                        "current_hp": monster_hp,
                        "max_hp": monster_hp,
                        "is_gone": False,
                        "half_dead": False,
                    }
                ],
            },
        },
    }


def _env_with_state(state):
    env = SlayTheSpireEnv()
    env.current_state = state
    env.process_manager = FakeProcessManager()
    return env


def _seed_tracking(env, act=1, floor=1, hp=80, deck_size=10):
    env.last_player_hp = hp
    env.last_max_hp = 80
    env.last_monster_total_hp = None
    env.last_floor = floor
    env.last_screen_type = "NONE"
    env.last_relic_ids = {"Burning Blood"}
    env.last_deck_size = deck_size
    env.last_upgraded_cards = 0
    env.last_act = act
    env.last_in_combat = False
    env.terminal_reward_given = False


def test_death_reward_is_one_shot_and_post_death_state_has_no_fake_removal():
    env = _env_with_state(
        {
            "in_game": True,
            "available_commands": ["proceed", "state"],
            "game_state": {
                "screen_type": "GAME_OVER",
                "floor": 1,
                "act": 1,
                "current_hp": 0,
                "max_hp": 80,
                "relics": [_relic("Burning Blood")],
                "deck": [],
            },
        }
    )
    _seed_tracking(env, act=1, floor=1, hp=16, deck_size=10)

    reward = env._calculate_reward()
    assert reward == pytest.approx(DEATH_PENALTY - 0.16)

    env.current_state = {
        "in_game": False,
        "available_commands": ["start", "state"],
        "game_state": {"screen_type": "NONE"},
    }

    assert env._calculate_reward() == pytest.approx(0.0)


def test_positive_hp_not_in_game_state_is_not_treated_as_death():
    env = _env_with_state(
        {
            "in_game": False,
            "available_commands": ["start", "state"],
            "game_state": {
                "screen_type": "VICTORY",
                "floor": 51,
                "act": 3,
                "current_hp": 44,
                "max_hp": 80,
                "relics": [_relic("Burning Blood")],
                "deck": _deck(),
            },
        }
    )
    _seed_tracking(env, act=3, floor=51, hp=44)

    assert env._calculate_reward() == pytest.approx(0.0)
    assert env.terminal_reward_given is False


def test_act2_soft_reset_does_not_try_to_reach_main_menu():
    state = _map_state(act=2, floor=17)
    env = _env_with_state(state)
    env.episode_ended_by_act_completion = True

    _, info = env.reset()

    assert env.process_manager.sent == []
    assert info["raw_state"] is state
    assert env.last_act == 2
    assert env.last_floor == 17
    assert env.last_player_hp == 80
    assert env.episode_ended_by_act_completion is False


def test_act_transition_terminates_once_then_act2_continues_normally():
    act2_map = _map_state(act=2, floor=17)
    act2_combat = _combat_state(act=2, floor=18)
    env = _env_with_state(
        {
            "in_game": True,
            "available_commands": ["proceed", "state"],
            "game_state": {
                "screen_type": "CHEST",
                "floor": 17,
                "act": 1,
                "current_hp": 22,
                "max_hp": 80,
                "relics": [_relic("Burning Blood")],
                "deck": _deck(),
            },
        }
    )
    env.process_manager = FakeProcessManager([act2_map, act2_combat])
    _seed_tracking(env, act=1, floor=17, hp=22)

    _, reward, terminated, _, _ = env.step(66)

    assert terminated is True
    assert reward == pytest.approx(ACT_COMPLETION_REWARD + 0.58)
    assert env.episode_ended_by_act_completion is True

    env.reset()
    assert env.last_act == 2

    _, reward, terminated, _, _ = env.step(68)

    assert terminated is False
    assert reward == pytest.approx(FLOOR_REWARD)


def test_act3_and_act4_transitions_are_soft_reset_safe():
    for target_act, target_floor in [(3, 34), (4, 51)]:
        env = _env_with_state(
            {
                "in_game": True,
                "available_commands": ["proceed", "state"],
                "game_state": {
                    "screen_type": "CHEST",
                    "floor": target_floor,
                    "act": target_act - 1,
                    "current_hp": 50,
                    "max_hp": 80,
                    "relics": [_relic("Burning Blood")],
                    "deck": _deck(),
                },
            }
        )
        env.process_manager = FakeProcessManager(
            [_map_state(act=target_act, floor=target_floor)]
        )
        _seed_tracking(env, act=target_act - 1, floor=target_floor, hp=50)

        _, reward, terminated, _, _ = env.step(66)
        assert terminated is True
        assert reward == pytest.approx(ACT_COMPLETION_REWARD + 0.30)

        env.reset()
        assert env.last_act == target_act
        assert env.last_floor == target_floor
        assert env.process_manager.sent == ["PROCEED"]


def test_relic_swap_is_rewarded_even_when_relic_count_does_not_change():
    env = _env_with_state(
        {
            "in_game": True,
            "available_commands": ["proceed", "state"],
            "game_state": {
                "screen_type": "BOSS_REWARD",
                "floor": 16,
                "act": 1,
                "current_hp": 40,
                "max_hp": 80,
                "relics": [_relic("New Boss Relic")],
                "deck": _deck(),
            },
        }
    )
    _seed_tracking(env, act=1, floor=16, hp=40)

    assert env._calculate_reward() == pytest.approx(1.0)


def test_monster_healing_is_negative_progress_not_zero_reward():
    env = _env_with_state(_combat_state(act=1, floor=3, hp=70, monster_hp=35))
    _seed_tracking(env, act=1, floor=3, hp=70)
    env.last_in_combat = True
    env.last_monster_total_hp = 20

    assert env._calculate_reward() == pytest.approx(-0.15)


def test_huge_hp_deltas_are_clipped_to_keep_reward_bounded():
    env = _env_with_state(_combat_state(act=1, floor=3, hp=70, monster_hp=0))
    _seed_tracking(env, act=1, floor=3, hp=70)
    env.last_in_combat = True
    env.last_monster_total_hp = 250

    assert env._calculate_reward() == pytest.approx(1.0)
