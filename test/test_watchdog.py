import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import SlayTheSpireEnv

class CrashingProcessManager:
    def __init__(self, fail_on_read=False, fail_on_send=False, success_states=None):
        self.fail_on_read = fail_on_read
        self.fail_on_send = fail_on_send
        self.states = list(success_states or [])
        self.sent = []
        self.terminate_called = 0
        self.launch_called = 0
        self.signal_ready_called = 0
        self._proc = object()  # Mock process object not None initially

    def read_state(self):
        if self.fail_on_read:
            raise ConnectionResetError("Connection reset by peer")
        if not self.states:
            raise EOFError("Pipe broken")
        return self.states.pop(0)

    def send_command(self, command):
        if self.fail_on_send:
            raise TimeoutError("Send timed out")
        self.sent.append(command)

    def signal_ready(self):
        self.signal_ready_called += 1

    def launch_game(self):
        self.launch_called += 1
        self._proc = object()
        # Reset the failures so next attempt works
        self.fail_on_read = False
        self.fail_on_send = False

    def terminate(self):
        self.terminate_called += 1
        self._proc = None

    def stop(self):
        pass


def test_step_watchdog_trigger():
    # Setup: mock process manager that throws on next read_state
    fake_pm = CrashingProcessManager(fail_on_read=True)
    env = SlayTheSpireEnv()
    env.process_manager = fake_pm
    env.worker_dir = "mock_worker_dir"  # Ensure Python-as-parent mode is simulated
    env.current_state = {
        "in_game": True,
        "available_commands": ["end"],
        "game_state": {
            "screen_type": "NONE",
            "floor": 1,
            "act": 1,
            "current_hp": 80,
            "max_hp": 80,
            "relics": [],
            "deck": [],
        }
    }

    # Call step
    obs, reward, terminated, truncated, info = env.step(0)

    # Check that watchdog restart was triggered
    assert fake_pm.terminate_called == 1
    assert fake_pm.launch_called == 1
    assert fake_pm.signal_ready_called == 1
    assert env.current_state == {}
    
    # Check that dummy episode termination is returned
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["crashed"] is True
    assert np.all(obs == 0.0)


def test_reset_watchdog_recovery():
    # Setup: mock process manager that fails on first read, but then succeeds on relaunch
    success_states = [
        # Main Menu state (after relaunch)
        {
            "in_game": False,
            "available_commands": ["start"],
            "game_state": {"screen_type": "NONE"}
        },
        # Transition state
        {
            "in_game": False,
            "available_commands": ["state"],
            "game_state": {"screen_type": "NONE"}
        },
        # Neow room state
        {
            "in_game": True,
            "available_commands": ["choose"],
            "game_state": {
                "screen_type": "NEOW_ROOM",
                "floor": 0,
                "act": 1,
                "current_hp": 80,
                "max_hp": 80,
                "relics": [],
                "deck": [],
            }
        }
    ]
    fake_pm = CrashingProcessManager(fail_on_read=True, success_states=success_states)
    
    env = SlayTheSpireEnv()
    env.process_manager = fake_pm
    env.worker_dir = "mock_worker_dir"  # Simulate Python-as-parent mode
    env.current_state = {}

    obs, info = env.reset()

    # Verify that:
    # 1. First reset attempt failed, calling terminate
    assert fake_pm.terminate_called >= 1
    # 2. Re-launch was triggered
    assert fake_pm.launch_called >= 1
    # 3. reset completed successfully on subsequent attempt
    assert env.current_state["in_game"] is True
    assert fake_pm.sent == ["START IRONCLAD", "STATE"]
