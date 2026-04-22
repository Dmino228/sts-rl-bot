import sys
import os

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
    
    # 2. Setup Checkpoint Callback
    os.makedirs("./models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="ppo_sts",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # 3. Initialize Model
    # tensorboard_log for Architect monitoring
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log="./ppo_sts_tensorboard/"
    )
    
    # Completely bypass stdout to protect the CommunicationMod pipe
    custom_logger = configure("./ppo_sts_tensorboard/", ["tensorboard", "csv"])
    model.set_logger(custom_logger)
    
    # 4. Training Loop with graceful exit
    total_timesteps = 100_000
    try:
        print(f"Starting training for {total_timesteps} timesteps...", file=sys.stderr)
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually (Ctrl+C). Saving model...", file=sys.stderr)
    except Exception as e:
        print(f"\nTraining crashed: {e}", file=sys.stderr)
        raise e
    finally:
        model.save("./models/ppo_sts_final")
        vec_env.close()
        print("Model saved to ./models/ppo_sts_final.zip", file=sys.stderr)

if __name__ == "__main__":
    main()
