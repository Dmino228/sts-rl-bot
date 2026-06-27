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
        
    def terminate(self):
        pass
        
    def is_process_alive(self):
        return True


def _deck(size=10, upgrades=0):
    return [{"id": f"Card{i}", "upgrades": upgrades} for i in range(size)]


def _relic(relic_id):
    return {"id": relic_id, "name": relic_id}


def _combat_state(hp=80, mhp=100, relics=None, deck=None):
    return {
        "in_game": True,
        "available_commands": ["end"],
        "game_state": {
            "screen_type": "NONE",
            "room_phase": "COMBAT",
            "floor": 1,
            "act": 1,
            "current_hp": hp,
            "max_hp": 80,
            "relics": relics or [_relic("Burning Blood")],
            "deck": deck or _deck(),
            "combat_state": {
                "monsters": [
                    {"id": "JawWorm", "current_hp": mhp, "max_hp": 40, "is_gone": False}
                ],
                "player": {"energy": 3},
            },
        },
    }

def test_anti_stall_kill_switch():
    """Verify that exceeding MAX_COMBAT_STEPS instantly terminates the episode with DEATH_PENALTY."""
    import env as env_module
    
    # We will need MAX_COMBAT_STEPS + 2 states.
    # 1 for reset(), MAX_COMBAT_STEPS for steps, 1 for the final step that triggers the kill switch.
    total_steps = env_module.MAX_COMBAT_STEPS + 1
    
    states = [_combat_state(hp=80, mhp=40) for _ in range(total_steps + 1)]
    pm = FakeProcessManager(states)
    
    env = SlayTheSpireEnv()
    env.process_manager = pm
    
    env.current_state = states.pop(0)
    env._reset_reward_tracking(bootstrap_current=True)
    
    # Step until exactly the limit
    for _ in range(env_module.MAX_COMBAT_STEPS):
        obs, rew, terminated, truncated, info = env.step(0)
        assert not terminated
        
    # The next step should trigger the stall_failure
    obs, rew, terminated, truncated, info = env.step(0)
    assert terminated is True
    # Reward should include the death penalty plus the standard anti-stall penalty for that step
    assert env_module.DEATH_PENALTY + env_module.ANTI_STALL_PENALTY == pytest.approx(rew)
