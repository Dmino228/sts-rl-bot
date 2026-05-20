# 08_PHASE 5: MLOPS & LOCAL CLUSTER ARCHITECTURE

## ROLE AND OBJECTIVE
The reward system (V3.2) and the StateEncoder are verified. We are now scaling the environment from a single process to a multi-processed local cluster using `stable_baselines3.common.vec_env.SubprocVecEnv`. 

## 1. DYNAMIC ENVIRONMENT INITIALIZATION
`SlayTheSpireEnv` must be refactored to accept two new parameters:
- `worker_dir` (str): The isolated directory where this specific Java subprocess will run.
- `character_class` (str): The class to play ("IRONCLAD", "SILENT", "DEFECT", "WATCHER").

**Execution isolation:** The `subprocess.Popen` call that launches the Java ModTheSpire `.jar` MUST set the `cwd` (Current Working Directory) argument to `worker_dir`. This prevents multiple instances from corrupting shared `saves/` or `preferences/` files.
**Character Selection:** After establishing the CommunicationMod pipe, send the command `START <CHARACTER_CLASS>` instead of relying on the default run.

## 2. THE CLUSTER MANAGER (`train_cluster.py`)
Create a new entry-point script for training.
- **CLI Arguments:** Use `argparse` to accept `--workers` (default 4, recommended multiples of 4) and `--timesteps` (default 1_000_000).
- **Environment Cloning:** For each worker ID (0 to N-1):
  1. Define a target path: `./cluster_workers/worker_{id}`.
  2. If the directory doesn't exist, strictly copy the contents of the master portable STS environment (the extracted `sts_env_v1`) into it using `shutil`.
- **Character Assignment (Round-Robin):**
  Assign characters dynamically based on modulo math:
  `chars = ["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]`
  `assigned_char = chars[worker_id % 4]`
- **Vectorization:**
  Wrap the environment creation logic in `make_env(worker_id, assigned_char, worker_dir)` functions and pass them to `SubprocVecEnv`.
- **Training Initiation:**
  Initialize the PPO model and call `.learn(total_timesteps=args.timesteps)`. Ensure TensorBoard logging remains centralized in `./logs/ppo_sts_cluster/`.

## 3. GRACEFUL SHUTDOWN
Ensure that when `train_cluster.py` receives a KeyboardInterrupt (Ctrl+C), it cleanly closes `SubprocVecEnv`, which in turn should terminate all underlying Java subprocesses to prevent orphan memory leaks.