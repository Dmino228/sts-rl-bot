# MLOps Phase 6.2: Cloud Conquest (Google Colab Cluster)

## Architecture Status
We have successfully optimized the local cluster (MLOps Local Cluster) to the absolute limits of the JVM engine and the Stable Baselines3 framework.
- RAM usage per Java process: **~750 MB** (thanks to the `-Xint` flag and off-heap reductions).
- CPU overhead eliminated by using Python's built-in `makefile` buffers for TCP sockets.
- Missing `ep_rew_mean` logs fixed by correctly wrapping the environment in the `Monitor` class (bypassing the `__getattr__` bug in Gymnasium).

## Strategy for Google Colab
The Linux environment in Google Colab solves our final problem—the VRAM (Graphics Card) bottleneck. Windows forced interface allocation for every window, even with `--nogui` arguments. On Linux, we will utilize **Xvfb (X Virtual Framebuffer)**.

`Xvfb` creates a virtual display matrix directly in the system's RAM. The game "thinks" it is rendering an image on a monitor, but the physical graphics card (e.g., the powerful Tesla T4 with 15GB VRAM in Colab) remains 100% free for the PyTorch neural network!

### Initialization Commands for the Notebook (.ipynb)
In Colab, you do not need to modify the individual Java execution commands inside your Python code (e.g., in `cluster_manager.py`). You simply need to run the main `train_colab.py` script through the virtual display. As a result, all Java subprocesses will automatically inherit this virtual, safe environment without throwing OpenGL errors.

**Cell 1: Install Linux system dependencies**
```bash
!apt-get update
!apt-get install -y openjdk-11-jre-headless xvfb
```

**Cell 2: Run the training cluster via X Virtual Framebuffer**
```bash
!xvfb-run -a python train_colab.py --num-workers 8 --character IRONCLAD --timesteps 1000000
```