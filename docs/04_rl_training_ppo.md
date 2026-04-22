# 04_RL_TRAINING_PPO: STABLE BASELINES3 INTEGRATION (PHASE 3)

## ROLE & CONTEXT
We are now entering Phase 3. Our Environment (`SlayTheSpireEnv`) successfully communicates with the game, and our State/Action encoders are working flawlessly. Your task is to implement the Reinforcement Learning algorithm using `Stable Baselines3` (SB3). We will use the **PPO (Proximal Policy Optimization)** algorithm.

## 1. VECTORIZED ENVIRONMENTS
SB3 requires environments to be wrapped in a Vectorized Environment class.
- **For Debugging/MVP:** Use `DummyVecEnv`. It runs a single instance of the environment in the main thread. This is crucial for verifying that the model can interact with the environment without crashing.
- **For Scaling (Future):** Write the code so it can easily be toggled to use `SubprocVecEnv` (spawning multiple independent Python subprocesses, each managing its own `GameProcessManager` and JVM).

## 2. PPO MODEL SETUP
- **Algorithm:** Use `sb3.PPO`.
- **Policy:** Use `"MlpPolicy"` (Multi-Layer Perceptron), as our Observation Space is a flattened 1D `Box` array (no convolutions needed).
- **Action Masking:** Standard PPO in SB3 *does not* natively support Action Masking out of the box. You MUST use the `MaskablePPO` implementation from the `sb3-contrib` package.
  - *Dependency:* `pip install sb3-contrib`
  - *Implementation:* The environment must expose an `action_masks()` method that returns the binary mask generated in Phase 2. `MaskablePPO` will automatically call this during `predict()` and training to ignore illegal actions.

## 3. THE TRAINING LOOP (`train.py`)
Create a new entry point script `train.py`:
1. Initialize the `SlayTheSpireEnv`.
2. Wrap it in a `DummyVecEnv` (or `ActionMasker` wrapper if required by sb3-contrib).
3. Initialize `MaskablePPO` with the environment.
4. Set up a basic `CheckpointCallback` to save the PyTorch model (`.zip` format) every N steps.
5. Call `model.learn(total_timesteps=100_000)`.
6. Gracefully handle KeyboardInterrupt (`Ctrl+C`) to save the model before exiting.

## 4. LOGGING (TENSORBOARD)
- Enable TensorBoard logging in the PPO initialization (`tensorboard_log="./ppo_sts_tensorboard/"`). This is mandatory for the Architect to monitor episode length, cumulative rewards, and entropy.