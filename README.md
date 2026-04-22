# Slay the Spire RL Bot

A Reinforcement Learning environment and bot for Slay the Spire, built on top of `gymnasium` and the [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) API.

## Project Architecture (Phase 1 MVP)

This project uses an **inverse communication protocol** governed by CommunicationMod. 
Unlike typical Python scripts that spawn external binaries, Slay the Spire (via CommunicationMod) acts as the parent process and **spawns this Python script as a subprocess**.

### The Communication Loop:
1. Slay the Spire starts up, CommunicationMod reads `config.properties`, and spawns `main.py`.
2. Python writes `"ready"` to its own `sys.stdout` to inform the mod it is alive.
3. CommunicationMod streams the game's state (as JSON lines) down to Python's `sys.stdin`.
4. Python reads the state from `sys.stdin`, makes a decision, and writes a plain-text command (e.g., `PLAY 1`, `END`) to `sys.stdout`.

> **⚠️ CRITICAL: Standard Output vs Standard Error**
> Because `sys.stdout` is strictly reserved for sending plain-text commands back to the game, **all developer logs, debug prints, and error messages MUST go natively to `sys.stderr`**. If Python accidentally prints text to `sys.stdout`, CommunicationMod will attempt to execute it as a game command and likely crash.

## Prerequisites
- Slay the Spire (Steam Version)
- Python 3.12+
- `gymnasium`
- Steam Workshop Mods:
  - BaseMod
  - ModTheSpire
  - StSLib
  - CommunicationMod

## Installation and Setup

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure CommunicationMod:**
   You must tell CommunicationMod to launch this python script upon startup. Edit the CommunicationMod configuration file located at:
   - Windows: `%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`
   - Linux: `~/.config/ModTheSpire/CommunicationMod/config.properties`

   Add the command to launch the `main.py` entry point (make sure to properly escape special characters depending on your OS, standard Java properties file formatting applies):
   ```properties
   command=py "C\:\\dev\\sts rl bot\\main.py"
   ```

## Usage

**Do not manually run `py main.py` in your terminal.**

To start the agent:
1. Open Slay the Spire via Steam (Launch with mods).
2. Ensure CommunicationMod is checked in the ModTheSpire menu.
3. Click "Play". 
4. The game will automatically boot up `main.py` in the background, send states, and begin playing through the game using the Random Agent (Phase 1).
5. You can view the agent's output logic in the ModTheSpire standard developer console (press `~`) or within the game's log files.

## Project Structure
- `docs/` - Architecture and planning rules.
- `env.py` - Custom `gymnasium.Env` wrapper.
- `process_manager.py` - Core IO pipe logic parsing `sys.stdin` and communicating via `sys.stdout`.
- `main.py` - The agent loop that plays the simulation.
