import sys
import os
import glob

# 1. IMMEDIATE HANDSHAKE (Beat the CommunicationMod timeout)
sys.__stdout__.write("ready\n")
sys.__stdout__.flush()

# 2. SILENCE STABLE BASELINES3 (Protect the pipe from ASCII progress bars)
sys.stdout = sys.stderr

# 3. HEAVY IMPORTS (Now we can safely take 15 seconds to load PyTorch)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import configure
from env import SlayTheSpireEnv

def mask_fn(env: SlayTheSpireEnv):
    return env.get_action_mask()

def main():
    print("Phase 3: Stable Baselines3 MaskablePPO Training", file=sys.stderr)
    
    # 1. Initialize environment
    def make_env():
        env = SlayTheSpireEnv()
        return ActionMasker(env, mask_fn)
        
    vec_env = DummyVecEnv([make_env])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    tensorboard_path = os.path.join(BASE_DIR, "ppo_sts_tensorboard")
    models_path = os.path.join(BASE_DIR, "models")
    
    # 2. Setup Checkpoint Callback
    os.makedirs(models_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_path,
        name_prefix="ppo_sts",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # 3. Initialize or Load Model
    final_model_path = os.path.join(models_path, "ppo_sts_final.zip")
    checkpoints = glob.glob(os.path.join(models_path, "ppo_sts_*_steps.zip"))
    
    if os.path.exists(final_model_path):
        latest_model = final_model_path
    elif checkpoints:
        latest_model = max(checkpoints, key=os.path.getmtime)
    else:
        latest_model = None

    if latest_model:
        print(f"Loading model from {latest_model}...", file=sys.stderr)
        model = MaskablePPO.load(
            latest_model,
            env=vec_env,
            custom_objects={"n_steps": 2048},
            tensorboard_log=tensorboard_path
        )
    else:
        print("Initializing new model...", file=sys.stderr)
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            n_steps=2048,
            tensorboard_log=tensorboard_path
        )
    
    # Completely bypass stdout to protect the CommunicationMod pipe
    custom_logger = configure(tensorboard_path, ["tensorboard", "csv"])
    model.set_logger(custom_logger)
    
    # 4. Training Loop with graceful exit
    total_timesteps = 100_000
    try:
        print(f"Starting training for {total_timesteps} timesteps...", file=sys.stderr)
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually (Ctrl+C). Saving model...", file=sys.stderr)
        model.save(os.path.join(models_path, "ppo_sts_latest"))
    except EOFError:
        print("\nPipe broken - game closed (Alt+F4). Saving model...", file=sys.stderr)
        model.save(os.path.join(models_path, "ppo_sts_latest"))
    except Exception as e:
        print(f"\nTraining crashed: {e}", file=sys.stderr)
        raise e
    finally:
        model.save(os.path.join(models_path, "ppo_sts_final"))
        vec_env.close()
        print("Model saved to " + os.path.join(models_path, "ppo_sts_final"), file=sys.stderr)

if __name__ == "__main__":
    main()
