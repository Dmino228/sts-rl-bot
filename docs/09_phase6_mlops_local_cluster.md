# MLOps Phase 6.3: Resilience & Watchdog Architecture (Local Windows)

## Strategic Pivot: Embracing Instability
We are officially abandoning the `mlops_docker_cluster` and `mlops_colab_cluster` initiatives. Overnight stress testing on the local Windows cluster (`mlops_local_cluster`) revealed that long-running training sessions are interrupted not by Python or OS limits, but by internal Java exceptions within the heavily modded game engine itself (e.g., `log4j StackOverflowError`). 

Moving a fundamentally unstable Java stack to a Docker container or Linux VPS will not solve internal JVM crashes. Instead, we are adopting a "Crash-Only / Watchdog" architectural philosophy. We will build resilience at the Python `VecEnv` layer. If a Java worker crashes, Python will catch the resulting socket disconnection, automatically restart that specific game instance, and seamlessly resume training without crashing the overarching Stable Baselines3 process.

## Agent Task & Refactoring Blueprint

Below is the structured functional specification to be provided directly to the AI development agent.

***

### 🤖 Prompt for the AI Agent (Copy & Paste)

**Role:** Expert MLOps & Reinforcement Learning Systems Engineer.

**Context:** We are abandoning Docker/Linux deployment efforts. We are remaining strictly on the local Windows architecture (`sys.platform == "win32"` using `ThreadedVecEnv`). Overnight training crashed because one of the ModTheSpire Java instances threw a fatal internal `StackOverflowError` in `log4j`, breaking the TCP socket and crashing the entire SB3 training loop. Furthermore, the SB3 logger is overwriting our `progress.csv` and flattening our TensorBoard metrics on each resume.

**Primary Objective:** Implement an Auto-Restart Watchdog mechanism inside the Python environment to recover from Java crashes, fix the logging overwrites, and ensure metrics accurately track total timesteps across multiple resumed runs.

**Requirements:**

1. **The Watchdog Auto-Restart (`env.py` / `process_manager.py`):**
   * Wrap the network communication logic (e.g., `step()`, socket `recv()`, and `send()`) inside `try-except` blocks catching `ConnectionResetError`, `EOFError`, `TimeoutError`, and generic `Exception`.
   * If an exception occurs indicating the game process died, the environment must NOT crash the main Python script. Instead, it must trigger a soft restart:
     a) Terminate the dead/zombie Java process (using `self.process_manager.terminate()`).
     b) Relaunch the Java process (`self.process_manager.launch_game()`).
     c) Re-establish the socket connection.
     d) Treat the crash as the end of an episode: return a dummy observation, `reward=0`, `terminated=True`, `truncated=False`, and `info={"crashed": True}`.

2. **Fix SB3 Logger Overwrites (`train.py` / `train_cluster.py`):**
   * SB3's `configure()` method overwrites `progress.csv` by default. Modify the logging setup so that each training run is saved in a unique subfolder (e.g., `logs/ppo_run_{TIMESTAMP}`) OR configure the CSV/Tensorboard loggers to strictly *append* to the existing files.
   * Stop deleting old logs. Remove any code that wipes out old log files or TensorBoard directories at the start of the script.

3. **Expose Total Timesteps:**
   * When loading a pre-existing model (resuming training), explicitly print the model's current total steps to the console before calling `model.learn()`. (e.g., `logger.info(f"Resuming training. Current total timesteps: {model.num_timesteps}")`).

4. **Verify Windows Parity:**
   * Ensure `ThreadedVecEnv` and all JVM optimization flags (`-Xint`, etc.) are kept perfectly intact for the Windows environment. Do not introduce Linux-specific Xvfb commands.

Please output the exact modifications required for `env.py`, `process_manager.py`, and the main training script. Ensure the Watchdog gracefully handles socket disconnects without dropping the SB3 training loop.
***