# ═══════════════════════════════════════════════════════════════
# STS RL Bot — Google Colab Setup & Training Notebook
# ═══════════════════════════════════════════════════════════════
#
# This file contains the commands to paste into Colab cells.
# Each section delimited by "# ── CELL N ──" is a separate cell.
#
# Prerequisites:
#   - sts_env_v1.zip uploaded to your Google Drive root
#   - Colab runtime with GPU enabled (optional, CPU works for PPO)
#
# ═══════════════════════════════════════════════════════════════

# ── CELL 1: Mount Google Drive ──────────────────────────────────
# %% [markdown]
# ## 1. Mount Google Drive & System Setup

# %%
from google.colab import drive  # type: ignore
drive.mount('/content/drive')

# Verify the zip exists
import os
ZIP_PATH = '/content/drive/MyDrive/sts_env_v1.zip'
assert os.path.exists(ZIP_PATH), f"ERROR: {ZIP_PATH} not found. Upload sts_env_v1.zip to your Drive root."
print(f"✓ Found {ZIP_PATH} ({os.path.getsize(ZIP_PATH) / 1e6:.1f} MB)")


# ── CELL 2: Install system dependencies ────────────────────────
# %%
# Install xvfb (X virtual framebuffer) — required for LibGDX headless rendering.
# Without this, the Java process crashes with "No X11 DISPLAY variable" on Colab.
# Also install supporting X11 libraries that LibGDX/LWJGL needs.

!apt-get update -qq
!apt-get install -y -qq xvfb libxrender1 libxtst6 libxi6 libxrandr2 \
    libxcursor1 libxcomposite1 libasound2 libatk1.0-0 libgtk-3-0 \
    libgl1-mesa-glx libgl1-mesa-dri mesa-utils > /dev/null 2>&1
!which xvfb-run && echo "✓ xvfb-run installed"


# ── CELL 3: Extract portable environment ───────────────────────
# %%
import time
import shutil

BASE_ENV_DIR = '/content/sts_env'
WORKSPACE_DIR = '/content/workers'

# Extract fresh base environment (skip if already exists)
if not os.path.isdir(BASE_ENV_DIR):
    print(f"Extracting {ZIP_PATH} → {BASE_ENV_DIR} ...")
    t0 = time.time()
    shutil.unpack_archive(ZIP_PATH, BASE_ENV_DIR)
    print(f"✓ Extracted in {time.time() - t0:.1f}s")
else:
    print(f"✓ {BASE_ENV_DIR} already exists, skipping extraction.")

# Verify critical files
java_bin = os.path.join(BASE_ENV_DIR, 'jre', 'bin', 'java')
mts_jar = os.path.join(BASE_ENV_DIR, 'ModTheSpire.jar')
comm_mod = os.path.join(BASE_ENV_DIR, 'mods', 'CommunicationMod.jar')

for f in [java_bin, mts_jar, comm_mod]:
    assert os.path.exists(f), f"MISSING: {f}"
    print(f"  ✓ {os.path.basename(f)}")

# Make Java executable
os.chmod(java_bin, 0o755)
print("✓ Base environment ready.")


# ── CELL 4: Clone the bot repo ─────────────────────────────────
# %%
REPO_DIR = '/content/sts-rl-bot'

if not os.path.isdir(REPO_DIR):
    !git clone https://github.com/Dmino228/sts-rl-bot.git {REPO_DIR}
else:
    print(f"✓ {REPO_DIR} already exists.")

os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")


# ── CELL 5: Install Python dependencies ────────────────────────
# %%
!pip install -q gymnasium numpy stable-baselines3 sb3-contrib torch tensorboard


# ── CELL 6: Initialize cluster & verify ────────────────────────
# %%
import sys
sys.path.insert(0, REPO_DIR)

from sb3.cluster_manager import ClusterManager

NUM_WORKERS = 2  # Colab free tier: keep ≤2, Pro: try 4

cluster = ClusterManager(
    base_env_dir=BASE_ENV_DIR,
    workspace_dir=WORKSPACE_DIR,
    num_workers=NUM_WORKERS,
    character_class='IRONCLAD',
    use_xvfb=True,
    python_command='python3',
)

dirs = cluster.initialize_workers(force_rebuild=False)

print(f"\n{'='*50}")
print(f"Cluster initialized: {len(dirs)} workers")
for d in dirs:
    print(f"  → {d}")
print(f"{'='*50}")

# Print status
import json
print(json.dumps(cluster.status(), indent=2))


# ── CELL 7: Quick xvfb smoke test ──────────────────────────────
# %%
# Verify Java can run under xvfb
import subprocess

worker_0 = dirs[0]
java_bin = os.path.join(worker_0, 'jre', 'bin', 'java')

result = subprocess.run(
    ['xvfb-run', '-a', java_bin, '-version'],
    capture_output=True, text=True, timeout=15,
)
print("Java version output:")
print(result.stderr or result.stdout)
assert result.returncode == 0, f"Java failed under xvfb! RC={result.returncode}"
print("✓ Java runs successfully under xvfb-run.")


# ── CELL 8: Start training ─────────────────────────────────────
# %%
# Option A: Run via train_colab.py CLI (recommended for long runs)
!python {REPO_DIR}/sb3/train_colab.py \
    --base-env-dir {BASE_ENV_DIR} \
    --workspace-dir {WORKSPACE_DIR} \
    --num-workers {NUM_WORKERS} \
    --character IRONCLAD \
    --timesteps 500000 \
    --save-freq 10000

# Option B: Multi-character generalization
# !python {REPO_DIR}/sb3/train_colab.py \
#     --base-env-dir {BASE_ENV_DIR} \
#     --workspace-dir {WORKSPACE_DIR} \
#     --num-workers 4 \
#     --multi-character \
#     --timesteps 1000000


# ── CELL 9: TensorBoard (optional) ─────────────────────────────
# %%
# %load_ext tensorboard
# %tensorboard --logdir {REPO_DIR}/ppo_sts_tensorboard


# ── CELL 10: Save model back to Drive ──────────────────────────
# %%
import shutil

DRIVE_MODELS = '/content/drive/MyDrive/sts_rl_models'
os.makedirs(DRIVE_MODELS, exist_ok=True)

src_model = os.path.join(REPO_DIR, 'models', 'ppo_sts_final.zip')
if os.path.exists(src_model):
    dst = os.path.join(DRIVE_MODELS, f'ppo_sts_final_{time.strftime("%Y%m%d_%H%M")}.zip')
    shutil.copy2(src_model, dst)
    print(f"✓ Model saved to Drive: {dst}")
else:
    print("⚠ No final model found. Check training logs.")

# Also copy latest checkpoint
import glob
checkpoints = glob.glob(os.path.join(REPO_DIR, 'models', 'ppo_sts_colab_*_steps.zip'))
if checkpoints:
    latest = max(checkpoints, key=os.path.getmtime)
    dst = os.path.join(DRIVE_MODELS, os.path.basename(latest))
    shutil.copy2(latest, dst)
    print(f"✓ Latest checkpoint saved: {dst}")
