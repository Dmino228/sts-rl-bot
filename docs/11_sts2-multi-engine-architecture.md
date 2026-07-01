# Multi-Engine Architecture: Slay the Spire 1 & 2

## Introduction
This document outlines the architectural evolution of the project, aimed at simultaneously supporting two distinct game engines within a single Ray RLlib training cluster.
The transition to a "Multi-Engine" architecture is driven by the drastic differences in how both games are launched and communicated with, while maintaining the same core objective: providing unified observation spaces (ObservationSpace) and action spaces (ActionSpace) for the PPO model.

## 1. Directory Topology

To avoid "spaghetti code" and simplify dependency management, the project adopts a strict separation between the core and domain-specific (game) modules.

* `/` (Root): Contains base interfaces, startup scripts, and the Gymnasium environment abstraction.
* `/rllib`: Contains universal wrappers for the Ray framework, the PyTorch neural network architecture (action masking), and cluster configuration (game-agnostic).
* `/sts1`: Encapsulated knowledge regarding the first game. Contains JSON parsers, definitions of the 205-element state vector and 100-element action vector, as well as ModTheSpire handling logic.
* `/sts2`: Encapsulated knowledge regarding the second game. Communicates with the native headless engine (`sts2-cli`), and features a completely new, specific state space encoder architecture based on structures extracted from `.pck` files (spire-codex).

## 2. I/O Communication Paradigms

The most significant architectural difference, hidden behind the `ProcessManager` abstraction, concerns the process lifecycle:

### Slay the Spire 1 (Legacy)
* **Engine:** Java Virtual Machine (JVM).
* **Interface:** Graphical UI game launched via ModTheSpire with CommunicationMod installed.
* **Bottlenecks:** High RAM usage (Java memory leaks, necessity of using `-Xss4m` and `-Xmx512m`), slow Garbage Collector blocking threads, risk of exceeding mathematical limits (requires `[-np.inf, np.inf]` space bounds).
* **Management:** The Watchdog must rigorously monitor timeouts and dump error logs (`stdout`/`stderr`) before performing a hard kill on the game window.

### Slay the Spire 2 (Headless)
* **Engine:** C# / .NET Core (via `sts2-cli`).
* **Interface:** Fully headless, no GUI (Godot Stubs).
* **Advantages:** Runs exclusively the pure logic engine. Radically lower RAM and VRAM consumption. Direct communication via native IPC pipes (`stdin`/`stdout`). All game content is unlocked at the source code level.
* **Management:** Fast startup times (instant environment reset). The lack of graphical overhead allows for an increased limit of concurrent workers on the same hardware. For long RLlib runs, the Python process manager may recycle `sts2-cli` between runs using episode, step, or RSS limits to contain upstream `Sts2Headless` memory growth without interrupting an active decision.

## 3. Environment Abstraction Goal (Env)

The main environment class (`StSEnv`) in the root directory now serves exclusively as a **Router**. Its responsibilities are:
1. Initialize the process pipeline.
2. Catch the "raw" JSON from standard output.
3. Route this JSON to the appropriate module (`sts1` or `sts2`) to construct the dictionary: `{"observations": np.array, "action_mask": np.array}`.
4. Return the unified rewards (Reward Shaping) and state to the RLlib workers.

As a result, the neural network (PyTorch) itself is completely unaware of which version of the game it is currently playing – it only sees normalized tensor matrices.
