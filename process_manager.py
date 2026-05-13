"""
GameProcessManager for CommunicationMod protocol.

Supports two operating modes:
  1. **CommunicationMod-as-parent** (local dev):
     CommunicationMod launches this script. Communication via sys.stdin/stdout.
  2. **Python-as-parent** (Colab / SubprocVecEnv):
     Python calls launch_game() to spawn a JRE + ModTheSpire subprocess.
     Communication via subprocess pipes. Supports xvfb-run for headless.

Commands are plain text, NOT JSON. e.g. "START ironclad", "PLAY 1 0", "END"
"""

import sys
import os
import json
import time
import signal
import logging
import subprocess
from typing import Optional, Dict, Any, IO

logger = logging.getLogger(__name__)


class GameProcessManager:
    """Handles bidirectional communication with the CommunicationMod pipe.

    Protocol (both modes):
    1. Handshake: we write "ready\\n".
    2. Game writes JSON game state (one dict per line).
    3. We write plain-text commands.
    4. Repeat until game terminates.
    """

    def __init__(
        self,
        timeout: float = 120.0,
        worker_dir: Optional[str] = None,
        use_xvfb: bool = False,
    ) -> None:
        self.timeout = timeout
        self.worker_dir = worker_dir
        self.use_xvfb = use_xvfb
        self._last_state: Optional[Dict[str, Any]] = None

        # Subprocess handle (only set when Python launches the game)
        self._proc: Optional[subprocess.Popen] = None

        # I/O streams — default to sys pipes (CommunicationMod-as-parent mode)
        self._stdin_stream: Any = sys.stdin
        self._stdout_stream: Any = sys.__stdout__

    # ──────────────────────────────────────────────────────────────
    # COLAB / SUBPROCESS MODE
    # ──────────────────────────────────────────────────────────────

    def launch_game(self) -> None:
        """Spawn the game as a child process (Python-as-parent mode).

        Expects the following layout inside `self.worker_dir`:
            jre/bin/java
            ModTheSpire.jar
            desktop-1.0.jar
            mods/CommunicationMod.jar  (+ BaseMod, StSLib, etc.)
            preferences/

        CommunicationMod must be configured to launch this Python script.
        However in Colab mode we override CommunicationMod's command to
        point to a thin shim that just does the handshake relay.
        """
        if self.worker_dir is None:
            raise RuntimeError(
                "launch_game() requires worker_dir to be set."
            )

        game_dir = self.worker_dir
        java_bin = os.path.join(game_dir, "jre", "bin", "java")

        if not os.path.isfile(java_bin):
            raise FileNotFoundError(
                f"Java binary not found at {java_bin}. "
                f"Ensure sts_env_v1.zip was extracted correctly."
            )

        # Build the Java command
        java_cmd = [
            java_bin,
            "-jar", os.path.join(game_dir, "ModTheSpire.jar"),
            "--skip-launcher",
            "--mods", "basemod,CommunicationMod,StSLib,SuperFastMode",
        ]

        # Wrap in xvfb-run for headless Linux (Colab)
        if self.use_xvfb:
            launch_cmd = [
                "xvfb-run", "-a",
                "--server-args=-screen 0 1280x720x24",
            ] + java_cmd
        else:
            launch_cmd = java_cmd

        logger.info(
            "[LAUNCH] Starting game in %s\n  cmd: %s",
            game_dir, " ".join(launch_cmd),
        )

        self._proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=game_dir,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Override I/O to use subprocess pipes
        self._stdin_stream = self._proc.stdout  # game's stdout → our input
        self._stdout_stream = self._proc.stdin   # game's stdin  → our output

        logger.info("[LAUNCH] Game PID: %s", self._proc.pid)

    # ──────────────────────────────────────────────────────────────
    # PROTOCOL
    # ──────────────────────────────────────────────────────────────

    def signal_ready(self) -> None:
        """Send the 'ready' handshake to CommunicationMod."""
        self._stdout_stream.write("ready\n")
        self._stdout_stream.flush()
        logger.info("Sent 'ready' signal to CommunicationMod.")

    def read_state(self) -> Dict[str, Any]:
        """Read a JSON game state from the input pipe.

        Blocks until a valid JSON dict is received or timeout is reached.
        CommunicationMod sends one JSON object per line.
        """
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            try:
                line = self._stdin_stream.readline()
            except Exception as e:
                logger.error("Error reading stdin: %s", e)
                raise

            if not line:
                # EOF — CommunicationMod closed our stdin (game closed)
                raise EOFError("Pipe broken - game closed.")

            line = line.strip()
            if not line:
                continue

            try:
                state = json.loads(line)
                if isinstance(state, dict):
                    self._last_state = state
                    # Log a compact summary of the received state
                    screen = state.get("game_state", {}).get("screen_type", "?")
                    in_game = state.get("in_game", "?")
                    cmds = state.get("available_commands", [])
                    floor = state.get("game_state", {}).get("floor", "?")
                    hp = state.get("game_state", {}).get("current_hp", "?")
                    logger.info(
                        "[RECV] in_game=%s screen=%s floor=%s hp=%s cmds=%s",
                        in_game, screen, floor, hp, cmds,
                    )
                    return state
                else:
                    logger.debug("Ignored non-dict JSON: %s", line[:100])
                    continue
            except json.JSONDecodeError:
                # CommunicationMod might send error strings like "Invalid command"
                logger.warning("[CommunicationMod NON-JSON]: %s", line)
                continue

        raise TimeoutError(f"No JSON state received within {self.timeout}s.")

    def send_command(self, command: str) -> None:
        """Send a plain-text command to CommunicationMod via the output pipe.

        Commands are plain text like:
            START ironclad
            PLAY 1 0
            END
            CHOOSE 0
            PROCEED
            STATE
        """
        self._stdout_stream.write(command + "\n")
        self._stdout_stream.flush()
        logger.info("[SEND] %s", command)

    def stop(self) -> None:
        """Terminate the subprocess if we launched it."""
        if self._proc is not None:
            logger.info("[STOP] Terminating game process PID=%s", self._proc.pid)
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("[STOP] Force-killing game process.")
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception as e:
                logger.error("[STOP] Error during shutdown: %s", e)
            finally:
                self._proc = None
