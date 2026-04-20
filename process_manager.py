"""
GameProcessManager for CommunicationMod protocol.

CommunicationMod launches OUR script as a subprocess.
Communication happens via OUR stdin (game sends JSON) and OUR stdout (we send commands).
Commands are plain text, NOT JSON. e.g. "START ironclad", "PLAY 1 0", "END"
"""

import sys
import json
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GameProcessManager:
    """Handles communication with CommunicationMod via stdin/stdout.

    Protocol:
    1. CommunicationMod launches this script as a subprocess.
    2. We write "ready" to stdout to signal we're alive.
    3. CommunicationMod writes JSON game state to our stdin.
    4. We write plain-text commands to our stdout.
    5. CommunicationMod executes them, then writes the next JSON state.
    """

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout
        self._last_state: Optional[Dict[str, Any]] = None

    def signal_ready(self) -> None:
        """Send the 'ready' handshake to CommunicationMod."""
        sys.stdout.write("ready\n")
        sys.stdout.flush()
        logger.info("Sent 'ready' signal to CommunicationMod.")

    def read_state(self) -> Dict[str, Any]:
        """Read a JSON game state from stdin (sent by CommunicationMod).

        Blocks until a valid JSON dict is received or timeout is reached.
        CommunicationMod sends one JSON object per line.
        """
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            try:
                line = sys.stdin.readline()
            except Exception as e:
                logger.error(f"Error reading stdin: {e}")
                raise

            if not line:
                # EOF — CommunicationMod closed our stdin (game closed)
                raise RuntimeError("stdin closed by CommunicationMod (game likely exited).")

            line = line.strip()
            if not line:
                continue

            try:
                state = json.loads(line)
                if isinstance(state, dict):
                    self._last_state = state
                    return state
                else:
                    logger.debug(f"Ignored non-dict JSON: {line[:100]}")
                    continue
            except json.JSONDecodeError:
                # CommunicationMod might send error strings like "Invalid command"
                logger.warning(f"Non-JSON from CommunicationMod: {line}")
                print(f"[CommunicationMod]: {line}", file=sys.stderr)
                continue

        raise TimeoutError(f"No JSON state received within {self.timeout}s.")

    def send_command(self, command: str) -> None:
        """Send a plain-text command to CommunicationMod via stdout.

        Commands are plain text like:
            START ironclad
            PLAY 1 0
            END
            CHOOSE 0
            PROCEED
            STATE
        """
        sys.stdout.write(command + "\n")
        sys.stdout.flush()
        logger.debug(f"Sent command: {command}")

    def stop(self) -> None:
        """No-op for the stdin/stdout model. Game manages our lifecycle."""
        pass
