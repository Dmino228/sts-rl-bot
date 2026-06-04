import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process_manager import GameProcessManager


def test_invalid_ram_usage():
    """Verify that passing an invalid ram_usage string raises ValueError."""
    with pytest.raises(ValueError):
        GameProcessManager(ram_usage="super_high")


def test_default_ram_usage():
    """Verify that GameProcessManager defaults to 'default' ram_usage."""
    pm = GameProcessManager()
    assert pm.ram_usage == "default"


@pytest.mark.parametrize(
    "ram_usage, expected_flags, unexpected_flags",
    [
        (
            "low",
            [
                "-Xmx256m",
                "-Xms128m",
                "-XX:MaxDirectMemorySize=128m",
                "-Xss1m",
                "-XX:ReservedCodeCacheSize=16m",
                "-XX:MaxMetaspaceSize=64m",
                "-XX:+UseSerialGC",
                "-Xint",
            ],
            ["-Xmx512m", "-Xms256m"],
        ),
        (
            "default",
            ["-Xmx256m", "-Xms128m"],
            [
                "-XX:MaxDirectMemorySize=128m",
                "-Xint",
                "-Xmx512m",
                "-Xms256m",
            ],
        ),
        (
            "safe",
            ["-Xmx512m", "-Xms256m"],
            [
                "-Xmx256m",
                "-Xms128m",
                "-XX:MaxDirectMemorySize=128m",
                "-Xint",
            ],
        ),
    ],
)
@patch("subprocess.Popen")
@patch("os.path.isfile")
def test_ram_usage_jvm_args(mock_isfile, mock_popen, ram_usage, expected_flags, unexpected_flags, tmp_path):
    """Verify that each ram_usage profile passes the correct flags to Java."""
    mock_isfile.return_value = True

    # Setup worker_dir structure in tmp_path
    worker_dir = tmp_path / "worker_0"
    worker_dir.mkdir()
    jre_bin = worker_dir / "jre" / "bin"
    jre_bin.mkdir(parents=True)
    java_exe = jre_bin / "java.exe"
    java_exe.touch()

    # Mock Popen return value
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    # Initialize GameProcessManager
    pm = GameProcessManager(worker_dir=str(worker_dir), ram_usage=ram_usage)
    
    # We must patch socket binding to avoid port conflicts or network errors in test environment
    with patch("socket.socket") as mock_socket:
        mock_sock_inst = MagicMock()
        mock_sock_inst.accept.return_value = (MagicMock(), ("127.0.0.1", 12345))
        mock_socket.return_value = mock_sock_inst
        
        # Call launch_game (it will call Popen)
        pm.launch_game()

    # Check Popen arguments
    assert mock_popen.called
    launch_args = mock_popen.call_args[0][0]

    # Verify expected JVM flags are present
    for flag in expected_flags:
        assert flag in launch_args, f"Expected flag {flag} not found in launch args: {launch_args}"

    # Verify unexpected JVM flags are absent
    for flag in unexpected_flags:
        assert flag not in launch_args, f"Unexpected flag {flag} found in launch args: {launch_args}"

    # Cleanup open file handles
    pm.stop()
