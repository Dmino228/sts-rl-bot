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
# Remove cv2 from sys.path to prevent it from hijacking standard imports like 'typing'
sys.path = [p for p in sys.path if not p.endswith('cv2') and not p.endswith('cv2/')]
import typing
import os
import json
import time
import signal
import logging
import subprocess
import socket
from typing import Optional, Dict, Any, IO

logger = logging.getLogger(__name__)

# Environment variable overrides for headless software rendering on Linux
if sys.platform != "win32":
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"



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

        # Subprocess handle, logs, and sockets (only set when Python launches the game)
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_file: Optional[IO] = None
        self._stderr_file: Optional[IO] = None
        self._server_socket: Optional[socket.socket] = None
        self._socket: Optional[socket.socket] = None

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
        """
        if self.worker_dir is None:
            raise RuntimeError(
                "launch_game() requires worker_dir to be set."
            )

        game_dir = self.worker_dir
        game_dir_abs = os.path.abspath(game_dir)

        if sys.platform == "win32":
            java_bin = os.path.join(game_dir, "jre", "bin", "java.exe")
            if not os.path.isfile(java_bin):
                raise FileNotFoundError(
                    f"Java binary not found at {java_bin}. "
                    f"Ensure sts_env_v1.zip was extracted correctly."
                )
        else:
            # Headless Java OpenJDK 8 on Linux (Docker)
            java_bin = "java"


        # Extract worker_id from worker_dir (default to 0 if not found)
        worker_id = 0
        dirname = os.path.basename(os.path.normpath(game_dir_abs))
        if "_" in dirname:
            try:
                worker_id = int(dirname.split("_")[-1])
            except ValueError:
                pass

        # Create TCP socket server
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = 12340 + worker_id
        self._server_socket.bind(("127.0.0.1", port))
        self._server_socket.listen(1)
        logger.info("[LAUNCH] Worker %d socket server listening on 127.0.0.1:%d", worker_id, port)

        # Write agent_shim.py dynamically into the worker directory
        shim_path = os.path.join(game_dir_abs, "agent_shim.py")
        shim_content = f'''import sys
import socket
import threading

def main():
    port = {port}

    # Connect to the Python worker process's socket server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("127.0.0.1", port))
    except Exception as e:
        sys.stderr.write(f"agent_shim failed to connect to port {{port}}: {{e}}\\n")
        sys.stderr.flush()
        sys.exit(1)

    # Read from socket (commands from Python worker) and write to stdout (Java)
    def socket_to_stdout():
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                sys.stdout.write(data.decode("utf-8"))
                sys.stdout.flush()
        except Exception:
            pass

    t = threading.Thread(target=socket_to_stdout, daemon=True)
    t.start()

    # Read from stdin (JSON states from Java) and send to socket (Python worker)
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            sock.sendall(line.encode("utf-8"))
    except Exception:
        pass
    finally:
        sock.close()

if __name__ == "__main__":
    main()
'''
        with open(shim_path, "w", encoding="utf-8") as f:
            f.write(shim_content)

        # Add the directory containing the current Python executable to the FRONT of PATH env var.
        # This allows us to use just the Python binary name (e.g. python or python3) in config.properties,
        # avoiding issues with spaces in the absolute path to Python on Windows.
        python_dir = os.path.dirname(os.path.abspath(sys.executable))
        python_bin_name = os.path.basename(sys.executable)

        # Override user home and local appdata environment variables for complete isolation
        env = os.environ.copy()
        env["HOME"] = game_dir_abs
        env["USERPROFILE"] = game_dir_abs
        env["LOCALAPPDATA"] = os.path.join(game_dir_abs, "LocalAppData")
        env["APPDATA"] = os.path.join(game_dir_abs, "AppData")
        if "PATH" in env:
            env["PATH"] = python_dir + os.pathsep + env["PATH"]
        else:
            env["PATH"] = python_dir

        if sys.platform != "win32":
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
            env["MESA_GLSL_VERSION_OVERRIDE"] = "330"
            env["GALLIUM_DRIVER"] = "softpipe"

        # Write config.properties inside local appdata/config directories to force
        # CommunicationMod to run at startup with our agent_shim.py command
        config_dirs = [
            os.path.join(game_dir_abs, "LocalAppData", "ModTheSpire", "CommunicationMod"),
            os.path.join(game_dir_abs, ".config", "ModTheSpire", "CommunicationMod"),
        ]
        config_content = (
            f"command={python_bin_name} agent_shim.py\n"
            f"runAtStartup=true\n"
            f"runAtGameStart=true\n"
        )
        for cfg_dir in config_dirs:
            os.makedirs(cfg_dir, exist_ok=True)
            cfg_file = os.path.join(cfg_dir, "config.properties")
            with open(cfg_file, "w", encoding="utf-8") as f:
                f.write(config_content)

        # Write SuperFastModeConfig.properties to maximize game speed during training.
        # deltaMultiplier=10.0 is the maximum supported value.
        sfm_dirs = [
            os.path.join(game_dir_abs, "LocalAppData", "ModTheSpire", "SuperFastMode"),
            os.path.join(game_dir_abs, ".config", "ModTheSpire", "SuperFastMode"),
        ]
        sfm_content = (
            "isDeltaMultiplied=true\n"
            "deltaMultiplier=10.0\n"
            "EXISTS=YES INDEED I EXIST\n"
            "isInstantLerp=true\n"
        )
        for sfm_dir in sfm_dirs:
            os.makedirs(sfm_dir, exist_ok=True)
            sfm_file = os.path.join(sfm_dir, "SuperFastModeConfig.properties")
            with open(sfm_file, "w", encoding="utf-8") as f:
                f.write(sfm_content)

        display_config = os.path.join(game_dir_abs, "info.displayconfig")
        optimized_display_config = (
            "EXPLICIT_FULLSCREEN=false\n"
            "HEIGHT=480\n"
            "WIDTH=640\n"
            "MAX_FPS=60\n"
            "W_FULLSCREEN=false\n"
        )
        try:
            with open(display_config, "w", encoding="utf-8") as f:
                f.write(optimized_display_config)
            logger.info("[LAUNCH] Wrote optimized info.displayconfig for VRAM saving.")
        except Exception as e:
            logger.warning("[LAUNCH] Could not write display config: %s", e)

        if sys.platform == "win32":
            java_cmd = [
                java_bin,
                f"-Duser.home={game_dir_abs}",
                "-Xmx256m", "-Xms128m",         # Heap
                "-XX:MaxDirectMemorySize=128m", # Native Memory (Textures/Audio)
                "-Xss256k",                     # Thread Stacks
                "-XX:ReservedCodeCacheSize=16m",# Code Cache
                "-XX:MaxMetaspaceSize=64m",     # Metadata (classes)
                "-XX:+UseSerialGC",             # Garbage Collector
                "-Xint",
                "-jar", os.path.join(game_dir, "ModTheSpire.jar"),
                "nogui",
                "--skip-launcher",
                "--mods", "basemod,CommunicationMod,stslib,superfastmode",
            ]
        else:
            # HEADLESS LINUX/DOCKER
            java_cmd = [
                java_bin,
                f"-Duser.home={game_dir_abs}",
                "-Xmx256m", "-Xms128m",         # Heap
                "-XX:MaxDirectMemorySize=128m", # Native Memory (Textures/Audio)
                "-Xss256k",                     # Thread Stacks
                "-XX:ReservedCodeCacheSize=16m",# Code Cache
                "-XX:MaxMetaspaceSize=64m",     # Metadata (classes)
                "-XX:+UseSerialGC",             # Garbage Collector
                "-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true",
                "-Dorg.lwjgl.opengl.Display.enableHighDPI=false",
                "-Dorg.lwjgl.util.NoChecks=true",
                # "-Xint",
                "-Dorg.lwjgl.openal.libname=/dev/null",
                "-jar", os.path.join(game_dir, "ModTheSpire.jar"),
                "nogui",
                "--skip-launcher",
                "--mods", "basemod,CommunicationMod,stslib,superfastmode",
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

        # Open stdout and stderr log files in the worker directory to prevent deadlocks
        stdout_log_path = os.path.join(game_dir, "stdout.log")
        stderr_log_path = os.path.join(game_dir, "stderr.log")
        self._stdout_file = open(stdout_log_path, "w", encoding="utf-8")
        self._stderr_file = open(stderr_log_path, "w", encoding="utf-8")

        self._proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.PIPE,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
            cwd=game_dir,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )

        logger.info("[LAUNCH] Game PID: %s. Waiting for agent_shim.py to connect on port %d...", self._proc.pid, port)

        # Wait for agent_shim.py to connect via TCP
        self._server_socket.settimeout(300.0)  # 300 seconds timeout
        try:
            self._socket, addr = self._server_socket.accept()
            logger.info("[LAUNCH] Connected to agent_shim on %s", addr)
            # Wrap socket as text streams for readline and write
            self._stdin_stream = self._socket.makefile("r", encoding="utf-8")
            self._stdout_stream = self._socket.makefile("w", encoding="utf-8")
        except socket.timeout:
            logger.error("[LAUNCH] Timed out waiting for agent_shim.py to connect on port %d", port)
            self.stop()
            raise TimeoutError(f"Timed out waiting for agent_shim.py to connect on port {port}")

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
                    if logger.isEnabledFor(logging.DEBUG):
                        screen = state.get("game_state", {}).get("screen_type", "?")
                        in_game = state.get("in_game", "?")
                        cmds = state.get("available_commands", [])
                        floor = state.get("game_state", {}).get("floor", "?")
                        hp = state.get("game_state", {}).get("current_hp", "?")
                        logger.debug(
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
        logger.debug("[SEND] %s", command)

    def stop(self) -> None:
        """Terminate the subprocess if we launched it."""
        # 1. Close communication sockets
        if hasattr(self, "_stdin_stream") and self._stdin_stream is not None:
            try:
                self._stdin_stream.close()
            except Exception:
                pass
            self._stdin_stream = sys.stdin

        if hasattr(self, "_stdout_stream") and self._stdout_stream is not None:
            try:
                self._stdout_stream.close()
            except Exception:
                pass
            self._stdout_stream = sys.__stdout__

        if hasattr(self, "_socket") and self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if hasattr(self, "_server_socket") and self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        # 2. Terminate subprocess
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

        # 3. Close log files
        if hasattr(self, "_stdout_file") and self._stdout_file is not None:
            try:
                self._stdout_file.close()
            except Exception as e:
                logger.error("[STOP] Error closing stdout file: %s", e)
            self._stdout_file = None

        if hasattr(self, "_stderr_file") and self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception as e:
                logger.error("[STOP] Error closing stderr file: %s", e)
            self._stderr_file = None
