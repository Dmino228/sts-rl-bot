# Phase 7: RLlib Experimental Integration & Architectural Restructuring

## 1. Project Objective
We are moving beyond the limitations of Stable Baselines3 (SB3) to explore **Ray / RLlib**. The goal is to leverage RLlib's superior asynchronous execution, advanced multi-processing, and highly scalable configuration, which is better suited for a heavy, Java-based environment like Slay the Spire. 

This branch (`rllib-experimental`) is dedicated to successfully integrating Ray/RLlib while keeping our existing, battle-tested components (like the watchdog, IPC manager, and state encoders) fully intact.

## 2. Architectural Restructuring (Separation of Concerns)
Our codebase has grown significantly. Before implementing RLlib, the agent MUST restructure the project to prevent namespace pollution and configuration clashes between SB3 and RLlib. 

### Target Directory Structure:
- `/sb3/` -> Move all files specifically tied to Stable Baselines3 here. This includes `train_cluster.py`, `threaded_vec_env.py`, `mask_cache_vec_env.py`, and any custom SB3 callbacks.
- `/rllib/` -> Create this new directory for all RLlib-specific scripts, custom environment wrappers required by Ray, and RLlib training launch scripts.
- `/ (Project Root)` -> Core engine files shared between BOTH frameworks must remain here. This includes:
  - `env.py` (The Gym/Gymnasium interface)
  - `process_manager.py` (Java IPC and Watchdog)
  - `action_space.py`, `state_encoder.py` (Logic & Encoding)

*Constraint:* The core files in the root must remain framework-agnostic. They should not import SB3 or RLlib directly.

## 3. RLlib Integration Requirements & Assumptions

**A. Environment Registration:**
RLlib requires strict Gymnasium environment registration. The agent must create a proper environment creator function and register `SlayTheSpireEnv` with Ray. 

**B. Action Masking:**
RLlib handles action masking differently than SB3 (it typically requires a dictionary observation space with `action_mask` and `observations` keys, coupled with a custom PyTorch model). The agent must implement a custom RLlib ModelV2/TorchModelV2 to handle our existing action masks correctly.

**C. Asynchronous Workers & Port Collisions:**
We are keeping the local Windows cluster approach. The agent must ensure that Ray's environment workers do not cause port collisions (TCP ports 12340+). RLlib spawns workers differently than our old `ThreadedVecEnv`; the agent must ensure `process_manager.py` correctly allocates unique worker IDs and directories under Ray's multiprocessing.

**D. Fault Tolerance (Watchdog Compatibility):**
Our `process_manager.py` has a robust auto-restart Watchdog for Java crashes. The agent must ensure that RLlib's worker exception handling respects our soft-resets (returning `terminated=True` and recreating the JVM) without crashing the entire Ray cluster.

## 4. Expected Output
The agent is responsible for:
1. Reorganizing the files into `sb3/`, `rllib/`, and root.
2. Writing a new training script (e.g., `rllib/train_rllib.py`).
3. Implementing the custom RLlib Action Masking model.
4. Ensuring the environment successfully initializes and takes at least one optimization step via Ray.