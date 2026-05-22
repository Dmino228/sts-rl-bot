import os
import sys
import shutil
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cluster_manager import ClusterManager


@pytest.fixture
def temp_env_setup():
    # Set up temporary directories for testing
    base_dir = tempfile.mkdtemp()
    workspace_dir = tempfile.mkdtemp()

    # Create dummy critical files in base env
    os.makedirs(os.path.join(base_dir, "mods"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "jre", "bin"), exist_ok=True)

    # Write dummy jar files
    with open(os.path.join(base_dir, "ModTheSpire.jar"), "w") as f:
        f.write("dummy mts")
    with open(os.path.join(base_dir, "desktop-1.0.jar"), "w") as f:
        f.write("dummy desktop")
    with open(os.path.join(base_dir, "mods", "CommunicationMod.jar"), "w") as f:
        f.write("dummy comm")

    # Write dummy java executable based on platform
    java_ext = ".exe" if sys.platform == "win32" else ""
    java_bin_path = os.path.join(base_dir, "jre", "bin", f"java{java_ext}")
    with open(java_bin_path, "w") as f:
        f.write("dummy java")

    yield base_dir, workspace_dir

    # Cleanup after test
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.rmtree(workspace_dir, ignore_errors=True)


def test_cluster_manager_worker_validation_and_rebuild(temp_env_setup):
    base_dir, workspace_dir = temp_env_setup

    manager = ClusterManager(
        base_env_dir=base_dir,
        workspace_dir=workspace_dir,
        num_workers=2,
        character_class="IRONCLAD",
        use_xvfb=False,
    )

    # 1. Initialize workers first time (copies everything)
    dirs = manager.initialize_workers(force_rebuild=False)
    assert len(dirs) == 2
    assert manager._is_worker_valid(dirs[0]) is True
    assert manager._is_worker_valid(dirs[1]) is True

    # 2. Corrupt worker 0 by deleting a critical jar file
    mts_path = os.path.join(dirs[0], "ModTheSpire.jar")
    os.remove(mts_path)
    assert manager._is_worker_valid(dirs[0]) is False

    # 3. Initialize workers again with force_rebuild=False.
    # It should detect worker 0 is invalid, rebuild it (copy fresh), and skip worker 1 (already valid).
    dirs2 = manager.initialize_workers(force_rebuild=False)
    assert len(dirs2) == 2
    assert os.path.exists(mts_path) is True  # Recreated!
    assert manager._is_worker_valid(dirs2[0]) is True
    assert manager._is_worker_valid(dirs2[1]) is True
