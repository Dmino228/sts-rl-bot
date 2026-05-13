import sys
import os
import glob
import datetime
import logging
import traceback

# ──────────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

tensorboard_path = os.path.join(BASE_DIR, "ppo_sts_tensorboard")
models_path = os.path.join(BASE_DIR, "models")

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOGS_DIR, f"training_{TIMESTAMP}.log")

os.makedirs(tensorboard_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# TEE LOGGER — duplicates writes to stderr AND a log file
# CommunicationMod pipe (sys.__stdout__) is NEVER touched.
# ──────────────────────────────────────────────────────────────
class TeeLogger:
    """Writes every message to both the original stream and a log file."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "a", encoding="utf-8")

    def write(self, msg):
        if msg:  # skip empty writes
            self.stream.write(msg)
            self.file.write(msg)
            self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return False


# Redirect both stdout and stderr to the tee (file + real stderr).
# sys.__stdout__ stays untouched → CommunicationMod pipe is safe.
_tee = TeeLogger(sys.__stderr__, log_file)
sys.stdout = _tee
sys.stderr = _tee

# ──────────────────────────────────────────────────────────────
# PYTHON LOGGING CONFIG — captures SB3, process_manager, etc.
# ──────────────────────────────────────────────────────────────
_file_handler = logging.FileHandler(log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)

_stderr_handler = logging.StreamHandler(sys.__stderr__)
_stderr_handler.setLevel(logging.DEBUG)

_formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_file_handler.setFormatter(_formatter)
_stderr_handler.setFormatter(_formatter)

# Root logger — catches everything (SB3 uses "stable_baselines3.*" loggers)
logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(_file_handler)
logging.root.addHandler(_stderr_handler)

train_logger = logging.getLogger("train")

# ──────────────────────────────────────────────────────────────
# HANDSHAKE — must happen BEFORE heavy imports
# ──────────────────────────────────────────────────────────────
sys.__stdout__.write("ready\n")
sys.__stdout__.flush()
train_logger.info("Sent 'ready' handshake to CommunicationMod")

# ──────────────────────────────────────────────────────────────
# HEAVY IMPORTS
# ──────────────────────────────────────────────────────────────
train_logger.info("Loading PyTorch + Stable Baselines3 ...")
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import configure
from env import SlayTheSpireEnv
train_logger.info("Imports complete.")


def mask_fn(env: SlayTheSpireEnv):
    return env.get_action_mask()


def main():
    train_logger.info("=" * 60)
    train_logger.info("Phase 3: MaskablePPO Training — session %s", TIMESTAMP)
    train_logger.info("Log file: %s", log_file)
    train_logger.info("=" * 60)

    # 1. Initialize environment
    def make_env():
        env = SlayTheSpireEnv()
        return ActionMasker(env, mask_fn)

    vec_env = DummyVecEnv([make_env])

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
        train_logger.info("Loading model from %s ...", latest_model)
        model = MaskablePPO.load(
            latest_model,
            env=vec_env,
            custom_objects={"n_steps": 2048},
            tensorboard_log=tensorboard_path
        )
    else:
        train_logger.info("Initializing new model ...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,  # verbose=1 → SB3 prints training stats to stderr
            n_steps=2048,
            tensorboard_log=tensorboard_path
        )

    # SB3 custom logger: tensorboard + csv + stdout
    # "stdout" here uses sys.stdout which is our TeeLogger (file + real stderr).
    # sys.__stdout__ (CommunicationMod pipe) is never touched.
    custom_logger = configure(tensorboard_path, ["tensorboard", "csv", "stdout"])
    model.set_logger(custom_logger)

    # 4. Training Loop with graceful exit
    total_timesteps = 1_000_000
    try:
        train_logger.info("Starting training for %d timesteps ...", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )
        train_logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        train_logger.warning("Training interrupted manually (Ctrl+C). Saving model...")
        model.save(os.path.join(models_path, "ppo_sts_latest"))
    except EOFError:
        train_logger.warning("Pipe broken — game closed (Alt+F4). Saving model...")
        model.save(os.path.join(models_path, "ppo_sts_latest"))
    except Exception as e:
        train_logger.error("Training crashed: %s", e, exc_info=True)
        raise
    finally:
        model.save(os.path.join(models_path, "ppo_sts_final"))
        vec_env.close()
        train_logger.info("Model saved to %s", os.path.join(models_path, "ppo_sts_final"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Last-resort catch: log the full traceback to file even if main() re-raises
        train_logger.critical("FATAL — unhandled exception:\n%s", traceback.format_exc())
        sys.exit(1)
