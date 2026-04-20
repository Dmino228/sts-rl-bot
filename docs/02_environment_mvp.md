# 02_ENVIRONMENT_MVP: THE "BLIND & DEAF" RANDOM AGENT (PHASE 1)

## ROLE & CONTEXT
You are implementing the foundational infrastructure connecting a Python script to the game *Slay the Spire* via the *CommunicationMod* API. 
At this stage, we are building the Minimum Viable Product (MVP). There is NO Machine Learning yet. Our goal is strictly to establish a stable communication loop, define a basic `gymnasium.Env` wrapper, and run a "Random Agent" that clicks valid actions until it dies or wins Act 1.

## 1. SCOPE AND CONSTRAINTS
- **Character:** Ironclad ONLY.
- **Duration:** Act 1 ONLY (If the bot reaches the Act 1 Boss and wins/loses, the episode ends).
- **Brain:** A Random Agent. The bot will simply pick a random choice from the list of valid, available commands provided by the game state.
- **No ML, No Tensors:** Do not import PyTorch in this module yet. We are purely focusing on process management and JSON parsing.

## 2. COMMUNICATION PROTOCOL (GAME PROCESS MANAGEMENT)
- **CRITICAL ARCHITECTURE SHIFT:** *CommunicationMod* works by spawning the Python script as a subprocess, not the other way around.
- You must create a dedicated `GameProcessManager` class that:
  1. Writes the initial `"ready"` handshake to its own `sys.stdout` to inform the mod it is alive.
  2. Reads the incoming JSON state line-by-line from its own `sys.stdin` automatically provided by CommunicationMod.
  3. Sends plain-text commands (e.g., `START ironclad`, `PLAY 1 0`, `END`) to its own `sys.stdout`.
- **Log Management:** All Python print statements and logs intended for the developer must be routed to `sys.stderr`, as `sys.stdout` is strictly reserved for the CommunicationMod protocol.

## 3. THE GYMNASIUM WRAPPER STUB
Create a class `SlayTheSpireEnv(gymnasium.Env)`. For this MVP, implement the bare minimum:
- `__init__()`: Initialize the `GameProcessManager`. Set `action_space` and `observation_space` to simple, temporary dummy spaces (e.g., `Discrete(100)` and `Box()`), as we will define the strict ML spaces in Phase 2.
- `reset()`: Signal readiness to the CommunicationMod. Wait for the initial JSON state on stdin. If at the main menu, send the command `START ironclad`. Return a dummy observation.
- `step(action)`: 
  1. Forward the plain-text string command to the `GameProcessManager` which sends it to stdout.
  2. Block to read the new JSON state from stdin.
  3. Return `(obs, reward, terminated, truncated, info)`. For now, `reward` is 0, and `terminated` is True if the JSON indicates death or the end of Act 1.

## 4. THE RANDOM EXECUTION LOOP
Write a `main.py` script that instantiates `SlayTheSpireEnv` and runs a `while` loop. In each step, the agent parses the `available_commands` array from the raw JSON state, uses it to construct a valid plain-text command (e.g., picking a random target for a PLAY command), and passes it to `env.step()`. This script must be specified in the CommunicationMod `config.properties` file so the game launches it automatically.