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
- *CommunicationMod* works by reading from standard input (`stdin`) and writing to standard output (`stdout`).
- You must create a dedicated `GameProcessManager` class that:
  1. Spawns the *Slay the Spire* game as a subprocess.
  2. Reads the incoming JSON state line-by-line from the game's `stdout` without blocking indefinitely (handle timeouts).
  3. Sends JSON commands (e.g., `{"command": "PLAY", "index": 1}`) to the game's `stdin`.
- **Error Handling:** If the game process crashes, the environment must gracefully catch the exception, close the subprocess, and allow for a clean restart.

## 3. THE GYMNASIUM WRAPPER STUB
Create a class `SlayTheSpireEnv(gymnasium.Env)`. For this MVP, implement the bare minimum:
- `__init__()`: Initialize the `GameProcessManager`. Set `action_space` and `observation_space` to simple, temporary dummy spaces (e.g., `Discrete(100)` and `Box()`), as we will define the strict ML spaces in Phase 2.
- `reset()`: Send the command to start a new run with Ironclad. Wait for the initial state JSON and return a dummy observation.
- `step(action)`: 
  1. Translate the dummy action into a valid text command based on the `available_commands` from the current JSON state.
  2. Send the command to the game.
  3. Read the new JSON state.
  4. Return `(obs, reward, terminated, truncated, info)`. For now, `reward` is 0, and `terminated` is True if the JSON indicates death or the end of Act 1.

## 4. THE RANDOM EXECUTION LOOP
Write a `main.py` script that instantiates `SlayTheSpireEnv` and runs a `while` loop. In each step, the agent parses the `available_commands` array from the raw JSON state, uses `random.choice()` to pick one, and passes it to `env.step()`.