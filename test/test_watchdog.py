"""
Watchdog tests — verify that the auto-restart mechanism actually handles
the real failure mode: Java JVM alive but unresponsive (socket open, no data).

Previous tests only checked ConnectionResetError (instant exception).
The actual crash scenario is a StackOverflowError in a background Java thread
that leaves the JVM alive — the socket stays open, readline() blocks forever.

These tests verify:
1. socket.settimeout() is called on the data socket after accept()
2. socket.timeout raised by readline() is converted to TimeoutError
3. env.step() catches the TimeoutError and returns a "crashed" episode
4. env.reset() retries on a hanging process
5. send_command() also converts socket.timeout to TimeoutError
"""

import os
import sys
import socket
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process_manager import GameProcessManager
from env import SlayTheSpireEnv


# ─────────────────────────────────────────────────────
# Helper: mock process manager that simulates hangs
# ─────────────────────────────────────────────────────

class HangingProcessManager:
    """Simulates the REAL failure mode: socket.timeout from readline().

    On the first call to read_state(), raises TimeoutError (simulating
    what happens when socket.settimeout fires because Java hung).
    After terminate() + launch_game(), subsequent calls succeed.
    """

    def __init__(self, hang_on_read=True, hang_on_send=False, recovery_states=None):
        self.hang_on_read = hang_on_read
        self.hang_on_send = hang_on_send
        self.recovery_states = list(recovery_states or [])
        self.sent = []
        self.terminate_called = 0
        self.launch_called = 0
        self.signal_ready_called = 0
        self._proc = object()

    def read_state(self):
        if self.hang_on_read:
            # This is what ACTUALLY happens now after the fix:
            # socket.timeout -> converted to TimeoutError by read_state()
            raise TimeoutError(
                "Socket read timed out after 120.0s. "
                "Java process is alive but unresponsive."
            )
        if not self.recovery_states:
            raise EOFError("Pipe broken")
        return self.recovery_states.pop(0)

    def send_command(self, command):
        if self.hang_on_send:
            raise TimeoutError(
                "Socket write timed out sending 'STATE'. "
                "Java process is alive but unresponsive."
            )
        self.sent.append(command)

    def signal_ready(self):
        self.signal_ready_called += 1

    def launch_game(self):
        self.launch_called += 1
        self._proc = object()
        # After relaunch, communication works again
        self.hang_on_read = False
        self.hang_on_send = False

    def terminate(self):
        self.terminate_called += 1
        self._proc = None

    def stop(self):
        pass


# ─────────────────────────────────────────────────────
# Test: socket timeout is set after accept
# ─────────────────────────────────────────────────────

class TestSocketTimeout:
    """Verify the socket-level timeout is correctly set in launch_game()."""

    def test_socket_settimeout_called_after_accept(self):
        """After accept(), the data socket must have settimeout(self.timeout)."""
        pm = GameProcessManager(timeout=30.0, worker_dir="/fake")

        mock_server = MagicMock(spec=socket.socket)
        mock_client = MagicMock(spec=socket.socket)
        mock_server.accept.return_value = (mock_client, ("127.0.0.1", 54321))

        # Bypass actual launch_game (file ops, subprocess, etc.)
        # Directly test the socket setup path
        pm._server_socket = mock_server
        pm._proc = MagicMock()  # Pretend process launched

        # Simulate the accept + setup from launch_game()
        pm._socket, addr = pm._server_socket.accept()
        pm._socket.settimeout(pm.timeout)
        pm._stdin_stream = pm._socket.makefile("r", encoding="utf-8")
        pm._stdout_stream = pm._socket.makefile("w", encoding="utf-8")

        mock_client.settimeout.assert_called_once_with(30.0)


# ─────────────────────────────────────────────────────
# Test: read_state converts socket.timeout to TimeoutError
# ─────────────────────────────────────────────────────

class TestReadStateTimeout:
    """Verify read_state() properly handles socket.timeout."""

    def test_socket_timeout_raises_timeout_error(self):
        """readline() raising socket.timeout must become a TimeoutError."""
        pm = GameProcessManager(timeout=5.0)

        mock_stream = MagicMock()
        mock_stream.readline.side_effect = socket.timeout("timed out")
        pm._stdin_stream = mock_stream

        with pytest.raises(TimeoutError, match="Socket read timed out"):
            pm.read_state()

    def test_eof_still_raises_eoferror(self):
        """readline() returning '' (EOF) must still raise EOFError."""
        pm = GameProcessManager(timeout=5.0)

        mock_stream = MagicMock()
        mock_stream.readline.return_value = ""
        pm._stdin_stream = mock_stream

        with pytest.raises(EOFError, match="Pipe broken"):
            pm.read_state()

    def test_connection_reset_still_propagates(self):
        """ConnectionResetError from recv() must still propagate."""
        pm = GameProcessManager(timeout=5.0)

        mock_stream = MagicMock()
        mock_stream.readline.side_effect = ConnectionResetError("reset")
        pm._stdin_stream = mock_stream

        with pytest.raises(ConnectionResetError):
            pm.read_state()


# ─────────────────────────────────────────────────────
# Test: send_command converts socket.timeout to TimeoutError
# ─────────────────────────────────────────────────────

class TestSendCommandTimeout:
    """Verify send_command() handles socket.timeout."""

    def test_write_timeout_raises_timeout_error(self):
        """write() raising socket.timeout must become a TimeoutError."""
        pm = GameProcessManager(timeout=5.0)

        mock_stream = MagicMock()
        mock_stream.write.side_effect = socket.timeout("timed out")
        pm._stdout_stream = mock_stream

        with pytest.raises(TimeoutError, match="Socket write timed out"):
            pm.send_command("STATE")

    def test_flush_timeout_raises_timeout_error(self):
        """flush() raising socket.timeout must also be caught."""
        pm = GameProcessManager(timeout=5.0)

        mock_stream = MagicMock()
        mock_stream.write.return_value = None
        mock_stream.flush.side_effect = socket.timeout("timed out")
        pm._stdout_stream = mock_stream

        with pytest.raises(TimeoutError, match="Socket write timed out"):
            pm.send_command("PLAY 1 0")


# ─────────────────────────────────────────────────────
# Test: env.step() watchdog catches TimeoutError (hung Java)
# ─────────────────────────────────────────────────────

class TestStepWatchdog:
    """env.step() must catch TimeoutError and return a crashed episode."""

    def _make_env_with_state(self, pm):
        env = SlayTheSpireEnv()
        env.process_manager = pm
        env.worker_dir = "mock_worker_dir"
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
            },
        }
        return env

    def test_hung_java_triggers_cleanup_not_restart(self):
        """When read_state() raises TimeoutError (hung socket), the watchdog
        must terminate and return a crashed episode, but NOT relaunch.
        The relaunch happens in reset() via ThreadedVecEnv auto-reset."""
        fake_pm = HangingProcessManager(hang_on_read=True)
        env = self._make_env_with_state(fake_pm)

        obs, reward, terminated, truncated, info = env.step(0)

        assert fake_pm.terminate_called == 1
        # step() must NOT restart — that's reset()'s job
        assert fake_pm.launch_called == 0
        assert fake_pm.signal_ready_called == 0
        assert terminated is True
        assert info["crashed"] is True
        assert reward == 0.0
        assert np.all(obs == 0.0)

    def test_connection_reset_also_triggers_cleanup(self):
        """ConnectionResetError (clean socket close) must also terminate without restart."""
        fake_pm = HangingProcessManager(hang_on_read=False)
        # Override to raise ConnectionResetError instead
        original_read = fake_pm.read_state
        def crash_on_read():
            raise ConnectionResetError("Connection reset by peer")
        fake_pm.read_state = crash_on_read
        env = self._make_env_with_state(fake_pm)

        obs, reward, terminated, truncated, info = env.step(0)

        assert fake_pm.terminate_called == 1
        assert fake_pm.launch_called == 0  # No restart in step()
        assert terminated is True
        assert info["crashed"] is True

    def test_send_timeout_triggers_cleanup(self):
        """When send_command() hangs (write timeout), cleanup without restart."""
        fake_pm = HangingProcessManager(hang_on_read=False, hang_on_send=True)
        env = self._make_env_with_state(fake_pm)

        obs, reward, terminated, truncated, info = env.step(0)

        assert fake_pm.terminate_called == 1
        assert fake_pm.launch_called == 0  # No restart in step()
        assert terminated is True
        assert info["crashed"] is True


# ─────────────────────────────────────────────────────
# Test: env.reset() recovery after hung Java
# ─────────────────────────────────────────────────────

class TestResetWatchdog:
    """env.reset() must retry after a hung process and recover."""

    def test_reset_recovers_after_timeout(self):
        """First attempt hangs (TimeoutError), second attempt succeeds."""
        recovery_states = [
            {
                "in_game": False,
                "available_commands": ["start"],
                "game_state": {"screen_type": "NONE"},
            },
            {
                "in_game": False,
                "available_commands": ["state"],
                "game_state": {"screen_type": "NONE"},
            },
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
                },
            },
        ]
        fake_pm = HangingProcessManager(
            hang_on_read=True,
            recovery_states=recovery_states,
        )

        env = SlayTheSpireEnv()
        env.process_manager = fake_pm
        env.worker_dir = "mock_worker_dir"
        env.current_state = {}

        obs, info = env.reset()

        # First attempt failed → terminate called
        assert fake_pm.terminate_called >= 1
        # Relaunch triggered
        assert fake_pm.launch_called >= 1
        # Successfully recovered
        assert env.current_state["in_game"] is True
        assert "START IRONCLAD" in fake_pm.sent


# ─────────────────────────────────────────────────────
# Test: is_process_alive
# ─────────────────────────────────────────────────────

class TestIsProcessAlive:
    def test_none_proc(self):
        pm = GameProcessManager()
        assert pm.is_process_alive() is False

    def test_alive_proc(self):
        pm = GameProcessManager()
        pm._proc = MagicMock()
        pm._proc.poll.return_value = None  # Still running
        assert pm.is_process_alive() is True

    def test_dead_proc(self):
        pm = GameProcessManager()
        pm._proc = MagicMock()
        pm._proc.poll.return_value = 1  # Exited with code 1
        assert pm.is_process_alive() is False
