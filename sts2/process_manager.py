"""Process manager for the Slay the Spire 2 headless sts2-cli engine."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Optional, TextIO

from sts2.io import StS2StdIOOverlay

logger = logging.getLogger(__name__)


class StS2CliProcessManager:
    """Manage a native sts2-cli process over stdin/stdout JSON lines."""

    auto_launch = True

    def __init__(
        self,
        timeout: float = 120.0,
        worker_dir: Optional[str] = None,
        cli_path: str = "sts2-cli",
        cli_args: Optional[list[str]] = None,
        cli_cwd: Optional[str] = None,
        capture_stderr: bool = False,
    ) -> None:
        self.timeout = timeout
        self.worker_dir = worker_dir
        self.cli_path = cli_path
        self.cli_args = list(cli_args or [])
        self.cli_cwd = cli_cwd
        self.capture_stderr = capture_stderr
        self.io = StS2StdIOOverlay()

        self._proc: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_file: Optional[TextIO] = None
        self._last_state: Optional[dict[str, Any]] = None
        self._last_command: Optional[Any] = None
        self._last_command_at: Optional[float] = None

    def launch_game(self) -> None:
        """Start sts2-cli in headless mode."""
        self.stop()
        log_dir = os.path.abspath(self.worker_dir) if self.worker_dir else None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        stderr_target: TextIO | int = subprocess.DEVNULL
        if log_dir and self.capture_stderr:
            stderr_path = os.path.join(log_dir, "sts2-cli.stderr.log")
            self._stderr_file = open(stderr_path, "w", encoding="utf-8")
            stderr_target = self._stderr_file

        cmd = [self.cli_path, *self.cli_args]
        cwd = self._resolve_process_cwd()
        logger.info("[STS2] Starting sts2-cli: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            cwd=cwd,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._pump_stdout,
            name="sts2-cli-stdout",
            daemon=True,
        )
        self._reader_thread.start()

    def signal_ready(self) -> None:
        """Wait for the initial sts2-cli ready event."""
        state = self.read_state()
        if state.get("type") != "ready":
            raise RuntimeError(f"Expected sts2-cli ready event, got: {state!r}")

    def read_state(self) -> dict[str, Any]:
        """Read one JSON state emitted by sts2-cli."""
        start = time.time()
        while time.time() - start < self.timeout:
            remaining = max(0.01, self.timeout - (time.time() - start))
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    self._timeout_message("sts2-cli did not emit JSON state")
                ) from exc

            if line is None:
                raise EOFError("sts2-cli stdout closed.")

            try:
                state = self.io.decode_state_line(line)
            except ValueError:
                logger.warning("[STS2 NON-JSON]: %s", line.strip())
                continue
            if state is not None:
                self._last_state = state
                return state

        raise TimeoutError(self._timeout_message("No sts2-cli JSON state received"))

    def send_command(self, command: Any) -> None:
        """Send a single JSON command to sts2-cli stdin."""
        if self._proc is None or self._proc.stdin is None:
            raise EOFError("sts2-cli process is not running.")
        try:
            self._last_command = command
            self._last_command_at = time.time()
            self._proc.stdin.write(self.io.encode_command(command))
            self._proc.stdin.flush()
        except OSError:
            logger.exception("[STS2] Failed to send command: %s", command)
            raise

    def diagnostic_snapshot(self) -> dict[str, Any]:
        """Return lightweight process diagnostics for watchdog logs."""
        pid = self._proc.pid if self._proc is not None else None
        return {
            "pid": pid,
            "worker_id": self._worker_id_from_dir(),
            "last_command": self._last_command,
            "last_command_age_s": (
                None
                if self._last_command_at is None
                else round(time.time() - self._last_command_at, 3)
            ),
            "alive": self.is_process_alive(),
        }

    def _timeout_message(self, prefix: str) -> str:
        diag = self.diagnostic_snapshot()
        return (
            f"{prefix} within {self.timeout}s "
            f"(worker={diag['worker_id']} pid={diag['pid']} alive={diag['alive']} "
            f"last_command={diag['last_command']!r} "
            f"last_command_age_s={diag['last_command_age_s']})."
        )

    def _worker_id_from_dir(self) -> str:
        if not self.worker_dir:
            return "unknown"
        return os.path.basename(os.path.abspath(self.worker_dir))

    def _resolve_process_cwd(self) -> Optional[str]:
        if self.cli_cwd:
            cwd = os.path.abspath(self.cli_cwd)
            os.makedirs(cwd, exist_ok=True)
            return cwd

        project_path = self._find_project_path()
        if project_path:
            cursor = os.path.abspath(os.path.dirname(project_path))
            while True:
                if os.path.isfile(os.path.join(cursor, "global.json")):
                    return cursor
                parent = os.path.dirname(cursor)
                if parent == cursor:
                    break
                cursor = parent

        if self.worker_dir:
            return os.path.abspath(self.worker_dir)
        return None

    def _find_project_path(self) -> Optional[str]:
        for value in [self.cli_path, *self.cli_args]:
            candidate = str(value).strip('"')
            if candidate.lower().endswith(".csproj") and os.path.isfile(candidate):
                return candidate
        return None

    def is_process_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            finally:
                self._proc = None

        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

        self._stdout_queue = queue.Queue()
        self._reader_thread = None

    def terminate(self) -> None:
        self.stop()

    def _pump_stdout(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                self._stdout_queue.put(line)
        finally:
            self._stdout_queue.put(None)
