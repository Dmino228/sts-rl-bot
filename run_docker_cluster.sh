#!/bin/bash
set -e

# Start persistent Xvfb framebuffer in the background on port :99
echo "[BOOTSTRAP] Starting persistent Xvfb display server on DISPLAY=:99..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension RANDR &
XVFB_PID=$!

export DISPLAY=:99

# Let Xvfb spin up
sleep 2

# Check if Xvfb started successfully
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "[BOOTSTRAP] [ERROR] Xvfb failed to start. Exiting."
    exit 1
fi

echo "[BOOTSTRAP] Xvfb successfully launched with PID $XVFB_PID."

# Run a quick Mesa renderer diagnostic check
echo "[BOOTSTRAP] Diagnostic: Checking OpenGL/Mesa software rendering info..."
if command -v glxinfo >/dev/null 2>&1; then
    glxinfo | grep -E "OpenGL vendor|OpenGL renderer|OpenGL version" || true
else
    echo "[BOOTSTRAP] Warning: glxinfo command not found."
fi

# Clean up background process when the main python run ends
cleanup() {
    echo "[BOOTSTRAP] Shutting down Xvfb process (PID: $XVFB_PID)..."
    kill $XVFB_PID || true
    wait $XVFB_PID 2>/dev/null || true
    echo "[BOOTSTRAP] Cleanup finished."
}
trap cleanup EXIT

# Invoke train_cluster.py with all forwarded CLI arguments
echo "[BOOTSTRAP] Invoking python3 train_cluster.py $@"
python3 train_cluster.py "$@"
