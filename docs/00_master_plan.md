# 00_MASTER_PLAN: SLAY THE SPIRE RL ROADMAP

## PROJECT OVERVIEW
We are building an autonomous Reinforcement Learning (RL) agent in Python 3.12+ to play *Slay the Spire* using the `CommunicationMod` interface.

## DOCUMENTATION MANDATE
**CRITICAL RULE FOR AI AGENTS:** Whenever generating code interacting with the game state or sending commands, you MUST base your assumptions on the official CommunicationMod documentation (https://github.com/ForgottenArbiter/CommunicationMod). Do not hallucinate commands. Valid commands include: `START`, `POTION`, `PLAY`, `END`, `CHOOSE`, `PROCEED`, `RETURN`, `KEY`, `CLICK`, `WAIT`, `STATE`.

## PHASES
- **[DONE] Phase 1: Environment MVP.** Establish the subprocess pipe connection. Forward `sys.stdin` to `sys.stdout`. Random Agent execution loop.
- **[CURRENT] Phase 2: State & Action Encoding.** Translate the dynamic JSON state into fixed-size PyTorch-friendly Tensors (`gymnasium.spaces.Box`) and Discrete Actions (`gymnasium.spaces.Discrete`). Ensure it works for BOTH combat and out-of-combat screens (Map, Events, Rewards).
- **Phase 3: Gymnasium Integration & Algorithm.** Wrap the encoder in a strict `gymnasium.Env` and attach the `Stable Baselines3` PPO algorithm.
- **Phase 4: Reward Shaping.** Design the reward function (e.g., damage dealt, climbing floors, penalties for losing HP) to prevent the agent from stalling.
- **Phase 5: Cloud Training.** Headless, multi-process training over millions of steps.