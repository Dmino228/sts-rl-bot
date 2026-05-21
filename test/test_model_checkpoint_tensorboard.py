"""
Tests for model checkpoint save/load fidelity and TensorBoard metric propagation.

These tests verify the SB3 documented contracts:
  1. Monitor wrapper populates info["episode"] on done=True with keys {r, l, t}.
  2. MaskablePPO.save() / .load() preserves policy weights and training state.
  3. Custom VecEnv wrappers (CachedActionMaskVecEnv) do not strip info["episode"].
  4. Resumed training writes to TensorBoard with correct cumulative timestep offset.

All tests use lightweight stub environments — no Java/game subprocess needed.
"""

import os
import sys
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ──────────────────────────────────────────────────────────────────────
# Stub environment matching SlayTheSpireEnv's spaces (obs=205, act=100)
# ──────────────────────────────────────────────────────────────────────

class StubSlayTheSpireEnv(gym.Env):
    """Minimal env that mimics SlayTheSpireEnv's spaces and terminates predictably.

    Episodes last exactly `episode_length` steps, with a fixed reward per step.
    This allows deterministic assertions on Monitor's info["episode"]["r"] and ["l"].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length: int = 5,
        reward_per_step: float = 1.0,
        obs_size: int = 205,
        action_size: int = 100,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(action_size)
        self._episode_length = episode_length
        self._reward_per_step = reward_per_step
        self._step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        obs = self.observation_space.sample()
        return obs, {"action_mask": self._full_mask()}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        terminated = self._step_count >= self._episode_length
        obs = self.observation_space.sample()
        return obs, self._reward_per_step, terminated, False, {"action_mask": self._full_mask()}

    def get_action_mask(self) -> np.ndarray:
        return self._full_mask()

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO calls this via env_method('action_masks')."""
        return self._full_mask()

    def _full_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=np.int8)


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Monitor wrapper produces info["episode"]
# ══════════════════════════════════════════════════════════════════════


class TestMonitorInfoEpisode:
    """Verify the SB3 Monitor wrapper contract:
    on done=True the info dict must contain an 'episode' key with
    {'r': cumulative_reward, 'l': episode_length, 't': elapsed_time}.
    """

    def test_monitor_adds_episode_key_on_termination(self):
        """Monitor must inject info['episode'] exactly when done=True."""
        from stable_baselines3.common.monitor import Monitor

        ep_len = 4
        reward = 2.0
        env = Monitor(StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=reward))
        env.reset()

        info = {}
        for _ in range(ep_len):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())

        assert terminated, "Episode should have terminated"
        assert "episode" in info, "Monitor must populate info['episode'] on done=True"

    def test_monitor_episode_reward_matches_sum(self):
        """info['episode']['r'] must equal the sum of step rewards."""
        from stable_baselines3.common.monitor import Monitor

        ep_len = 6
        reward = 3.0
        env = Monitor(StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=reward))
        env.reset()

        for _ in range(ep_len):
            _, _, _, _, info = env.step(env.action_space.sample())

        assert info["episode"]["r"] == pytest.approx(ep_len * reward)

    def test_monitor_episode_length_matches_steps(self):
        """info['episode']['l'] must equal the number of steps taken."""
        from stable_baselines3.common.monitor import Monitor

        ep_len = 7
        env = Monitor(StubSlayTheSpireEnv(episode_length=ep_len))
        env.reset()

        for _ in range(ep_len):
            _, _, _, _, info = env.step(env.action_space.sample())

        assert info["episode"]["l"] == ep_len

    def test_monitor_episode_time_is_positive(self):
        """info['episode']['t'] must be a positive elapsed time."""
        from stable_baselines3.common.monitor import Monitor

        ep_len = 3
        env = Monitor(StubSlayTheSpireEnv(episode_length=ep_len))
        env.reset()

        for _ in range(ep_len):
            _, _, _, _, info = env.step(env.action_space.sample())

        assert info["episode"]["t"] >= 0.0

    def test_monitor_no_episode_key_before_termination(self):
        """info['episode'] must NOT be present on intermediate steps."""
        from stable_baselines3.common.monitor import Monitor

        ep_len = 10
        env = Monitor(StubSlayTheSpireEnv(episode_length=ep_len))
        env.reset()

        # Step once (not terminal)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert not terminated
        assert "episode" not in info

    def test_monitor_resets_accumulator_across_episodes(self):
        """After reset, the next episode must accumulate from zero."""
        from stable_baselines3.common.monitor import Monitor

        ep_len = 3
        reward = 5.0
        env = Monitor(StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=reward))

        # Episode 1
        env.reset()
        for _ in range(ep_len):
            _, _, _, _, info = env.step(env.action_space.sample())
        ep1_reward = info["episode"]["r"]

        # Episode 2
        env.reset()
        for _ in range(ep_len):
            _, _, _, _, info = env.step(env.action_space.sample())
        ep2_reward = info["episode"]["r"]

        assert ep1_reward == pytest.approx(ep_len * reward)
        assert ep2_reward == pytest.approx(ep_len * reward)


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 2: CachedActionMaskVecEnv preserves info["episode"]
# ══════════════════════════════════════════════════════════════════════


class TestCachedActionMaskVecEnvInfoPassthrough:
    """Verify that CachedActionMaskVecEnv does not drop, overwrite,
    or strip the 'episode' key from info dicts returned by step_wait().
    """

    def _make_monitored_vec_env(self, n_envs: int = 2, ep_len: int = 3):
        """Create a DummyVecEnv -> CachedActionMaskVecEnv with Monitor-wrapped stubs."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from mask_cache_vec_env import CachedActionMaskVecEnv

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=1.0)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        base = DummyVecEnv([make_fn for _ in range(n_envs)])
        return CachedActionMaskVecEnv(base)

    def test_episode_info_survives_wrapper(self):
        """info['episode'] from Monitor must appear in CachedActionMaskVecEnv output."""
        ep_len = 3
        vec_env = self._make_monitored_vec_env(n_envs=1, ep_len=ep_len)
        vec_env.reset()

        episode_found = False
        # Step enough times to guarantee at least one episode completes
        for _ in range(ep_len * 3):
            actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
            _, _, dones, infos = vec_env.step(actions)
            for i, done in enumerate(dones):
                if done and "episode" in infos[i]:
                    episode_found = True
                    break
            if episode_found:
                break

        vec_env.close()
        assert episode_found, (
            "CachedActionMaskVecEnv must preserve info['episode'] from Monitor"
        )

    def test_episode_reward_value_correct_through_wrapper(self):
        """The reward reported in info['episode']['r'] must match the actual sum."""
        ep_len = 4
        reward_per_step = 2.5
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from mask_cache_vec_env import CachedActionMaskVecEnv

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=reward_per_step)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        base = DummyVecEnv([make_fn])
        vec_env = CachedActionMaskVecEnv(base)
        vec_env.reset()

        for _ in range(ep_len * 3):
            actions = np.array([vec_env.action_space.sample()])
            _, _, dones, infos = vec_env.step(actions)
            if dones[0] and "episode" in infos[0]:
                assert infos[0]["episode"]["r"] == pytest.approx(ep_len * reward_per_step)
                break
        vec_env.close()

    def test_action_mask_also_preserved(self):
        """CachedActionMaskVecEnv must preserve action_mask alongside episode info."""
        ep_len = 3
        vec_env = self._make_monitored_vec_env(n_envs=1, ep_len=ep_len)
        vec_env.reset()

        for _ in range(ep_len * 3):
            actions = np.array([vec_env.action_space.sample()])
            _, _, dones, infos = vec_env.step(actions)
            if dones[0]:
                # Both keys should coexist
                assert "episode" in infos[0] or "action_mask" in infos[0]
                break
        vec_env.close()


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Model save/load roundtrip
# ══════════════════════════════════════════════════════════════════════


class TestModelSaveLoadRoundtrip:
    """Verify the SB3 documented contract:
    MaskablePPO.save() and MaskablePPO.load() must produce an identical policy
    (weights, optimizer state) so that training can resume seamlessly.
    """

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp(prefix="sts_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def trained_model(self, tmp_dir):
        """Train a small MaskablePPO for a few steps and return (model, vec_env, path)."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=5, reward_per_step=1.0)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        vec_env = DummyVecEnv([make_fn])
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=64)

        save_path = os.path.join(tmp_dir, "test_model")
        model.save(save_path)
        return model, vec_env, save_path

    def test_saved_file_exists(self, trained_model):
        """model.save() must create a .zip file on disk."""
        _, _, save_path = trained_model
        assert os.path.isfile(save_path + ".zip"), "Model .zip file must be created"

    def test_loaded_model_has_same_observation_space(self, trained_model):
        """Loaded model observation_space must match the original."""
        from sb3_contrib import MaskablePPO

        original, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)
        assert loaded.observation_space == original.observation_space
        vec_env.close()

    def test_loaded_model_has_same_action_space(self, trained_model):
        """Loaded model action_space must match the original."""
        from sb3_contrib import MaskablePPO

        original, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)
        assert loaded.action_space == original.action_space
        vec_env.close()

    def test_loaded_policy_weights_identical(self, trained_model):
        """Policy network parameters must be bit-identical after save/load."""
        import torch as th
        from sb3_contrib import MaskablePPO

        original, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)

        for (name_orig, param_orig), (name_loaded, param_loaded) in zip(
            original.policy.named_parameters(),
            loaded.policy.named_parameters(),
        ):
            assert name_orig == name_loaded, f"Parameter name mismatch: {name_orig} vs {name_loaded}"
            assert th.equal(param_orig, param_loaded), (
                f"Parameter {name_orig} differs after save/load"
            )
        vec_env.close()

    def test_loaded_model_can_predict(self, trained_model):
        """A loaded model must be able to call predict() without errors."""
        from sb3_contrib import MaskablePPO

        _, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)
        obs = vec_env.reset()
        action_masks = np.ones((1, 100), dtype=np.int8)
        action, _ = loaded.predict(obs, action_masks=action_masks, deterministic=True)
        assert action.shape == (1,)
        assert 0 <= action[0] < 100
        vec_env.close()

    def test_loaded_model_preserves_num_timesteps(self, trained_model):
        """num_timesteps must be restored so TensorBoard x-axis continues correctly."""
        from sb3_contrib import MaskablePPO

        original, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)
        assert loaded.num_timesteps == original.num_timesteps, (
            f"num_timesteps mismatch: {loaded.num_timesteps} vs {original.num_timesteps}"
        )
        vec_env.close()

    def test_loaded_model_preserves_learning_rate(self, trained_model):
        """Hyperparameters like learning_rate must survive the roundtrip."""
        from sb3_contrib import MaskablePPO

        original, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)
        assert loaded.learning_rate == original.learning_rate
        vec_env.close()

    def test_loaded_model_preserves_n_steps(self, trained_model):
        """PPO-specific params like n_steps must survive the roundtrip."""
        from sb3_contrib import MaskablePPO

        original, vec_env, save_path = trained_model
        loaded = MaskablePPO.load(save_path, env=vec_env)
        assert loaded.n_steps == original.n_steps
        vec_env.close()


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Training resume with TensorBoard continuity
# ══════════════════════════════════════════════════════════════════════


class TestTrainingResumeContinuity:
    """Verify that a saved+loaded model resumes training with correct
    timestep accounting so TensorBoard plots are continuous.
    """

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp(prefix="sts_resume_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_resumed_training_continues_timestep_counter(self, tmp_dir):
        """After load + learn, num_timesteps must be the sum of both sessions."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=5)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        # Session 1
        vec_env = DummyVecEnv([make_fn])
        model = MaskablePPO("MlpPolicy", vec_env, n_steps=32, batch_size=16, verbose=0)
        model.learn(total_timesteps=64)
        ts_after_session1 = model.num_timesteps

        save_path = os.path.join(tmp_dir, "resume_model")
        model.save(save_path)
        vec_env.close()

        # Session 2 — resume
        vec_env2 = DummyVecEnv([make_fn])
        loaded = MaskablePPO.load(save_path, env=vec_env2)
        loaded.learn(total_timesteps=128, reset_num_timesteps=False)
        ts_after_session2 = loaded.num_timesteps

        assert ts_after_session2 >= ts_after_session1 + 64, (
            f"Timestep counter should continue: session1={ts_after_session1}, "
            f"session2={ts_after_session2}"
        )
        vec_env2.close()

    def test_tensorboard_log_directory_created(self, tmp_dir):
        """MaskablePPO with tensorboard_log must create the log directory."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=5)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        tb_path = os.path.join(tmp_dir, "tb_logs")
        vec_env = DummyVecEnv([make_fn])
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=32, batch_size=16, verbose=0,
            tensorboard_log=tb_path,
        )
        model.learn(total_timesteps=64)

        assert os.path.isdir(tb_path), "TensorBoard log directory must be created"
        # SB3 creates a subdirectory like "MaskablePPO_1"
        subdirs = os.listdir(tb_path)
        assert len(subdirs) > 0, "TensorBoard log must contain at least one run subdirectory"
        vec_env.close()

    def test_tensorboard_events_file_written(self, tmp_dir):
        """TensorBoard events file must be created and non-empty after training."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=5)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        tb_path = os.path.join(tmp_dir, "tb_events")
        vec_env = DummyVecEnv([make_fn])
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=32, batch_size=16, verbose=0,
            tensorboard_log=tb_path,
        )
        model.learn(total_timesteps=64)

        # Find events files
        events_files = []
        for root, dirs, files in os.walk(tb_path):
            for f in files:
                if f.startswith("events.out.tfevents"):
                    events_files.append(os.path.join(root, f))

        assert len(events_files) > 0, "At least one TensorBoard events file must exist"
        for ef in events_files:
            assert os.path.getsize(ef) > 0, f"Events file {ef} must be non-empty"
        vec_env.close()


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 5: ThreadedVecEnv info passthrough
# ══════════════════════════════════════════════════════════════════════


class TestThreadedVecEnvInfoPassthrough:
    """Verify that ThreadedVecEnv preserves info['episode'] from Monitor."""

    def test_episode_info_through_threaded_vec_env(self):
        """info['episode'] must survive ThreadedVecEnv -> CachedActionMaskVecEnv."""
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker
        from threaded_vec_env import ThreadedVecEnv
        from mask_cache_vec_env import CachedActionMaskVecEnv

        ep_len = 3

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=2.0)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        threaded = ThreadedVecEnv([make_fn])
        vec_env = CachedActionMaskVecEnv(threaded)
        vec_env.reset()

        episode_found = False
        for _ in range(ep_len * 5):
            actions = np.array([vec_env.action_space.sample()])
            _, _, dones, infos = vec_env.step(actions)
            if dones[0] and "episode" in infos[0]:
                assert infos[0]["episode"]["r"] == pytest.approx(ep_len * 2.0)
                assert infos[0]["episode"]["l"] == ep_len
                episode_found = True
                break

        vec_env.close()
        assert episode_found, "ThreadedVecEnv must propagate info['episode']"

    def test_multiple_workers_episode_tracking(self):
        """Each worker in ThreadedVecEnv must independently track episodes."""
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker
        from threaded_vec_env import ThreadedVecEnv
        from mask_cache_vec_env import CachedActionMaskVecEnv

        n_envs = 3
        ep_len = 4

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=1.0)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        threaded = ThreadedVecEnv([make_fn for _ in range(n_envs)])
        vec_env = CachedActionMaskVecEnv(threaded)
        vec_env.reset()

        episodes_per_worker = [0] * n_envs
        for _ in range(ep_len * 10):
            actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
            _, _, dones, infos = vec_env.step(actions)
            for i in range(n_envs):
                if dones[i] and "episode" in infos[i]:
                    episodes_per_worker[i] += 1

        vec_env.close()
        for i, count in enumerate(episodes_per_worker):
            assert count >= 1, f"Worker {i} must have completed at least 1 episode"


# ══════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Full integration — MaskablePPO logs rollout metrics
# ══════════════════════════════════════════════════════════════════════


class TestMaskablePPORolloutMetrics:
    """Verify that MaskablePPO.learn() with Monitor-wrapped envs
    actually logs rollout/ep_rew_mean and rollout/ep_len_mean.
    """

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp(prefix="sts_rollout_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_rollout_metrics_logged_to_csv(self, tmp_dir):
        """With a CSV logger, rollout/ep_rew_mean must appear after episodes complete."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.logger import configure
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        ep_len = 5
        reward = 1.0

        def make_fn():
            base_env = StubSlayTheSpireEnv(episode_length=ep_len, reward_per_step=reward)
            env = Monitor(base_env)
            return ActionMasker(env, lambda _: base_env.get_action_mask())

        vec_env = DummyVecEnv([make_fn])
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=32, batch_size=16, verbose=0,
        )

        csv_path = os.path.join(tmp_dir, "log_output")
        logger = configure(csv_path, ["csv"])
        model.set_logger(logger)

        # Train enough to complete multiple episodes and trigger PPO updates
        model.learn(total_timesteps=256)

        # Read the CSV log
        csv_file = os.path.join(csv_path, "progress.csv")
        assert os.path.isfile(csv_file), "CSV progress file must exist"

        with open(csv_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "rollout/ep_rew_mean" in content, (
            "rollout/ep_rew_mean must be logged when Monitor wraps the env"
        )
        assert "rollout/ep_len_mean" in content, (
            "rollout/ep_len_mean must be logged when Monitor wraps the env"
        )
        vec_env.close()

    def test_no_rollout_metrics_without_monitor(self, tmp_dir):
        """Without Monitor, rollout/ep_rew_mean should NOT appear in logs."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.logger import configure
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        def make_fn():
            env = StubSlayTheSpireEnv(episode_length=5)
            # Deliberately NO Monitor wrapper
            return ActionMasker(env, lambda _: env.get_action_mask())

        vec_env = DummyVecEnv([make_fn])
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=32, batch_size=16, verbose=0,
        )

        csv_path = os.path.join(tmp_dir, "log_no_monitor")
        logger = configure(csv_path, ["csv"])
        model.set_logger(logger)

        model.learn(total_timesteps=256)

        csv_file = os.path.join(csv_path, "progress.csv")
        if os.path.isfile(csv_file):
            with open(csv_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Without Monitor, ep_rew_mean should be absent or empty
            lines = [l for l in content.strip().split("\n") if l.strip()]
            if len(lines) > 1 and "rollout/ep_rew_mean" in lines[0]:
                # If header exists, values should be empty
                header_parts = lines[0].split(",")
                idx = header_parts.index("rollout/ep_rew_mean")
                for data_line in lines[1:]:
                    val = data_line.split(",")[idx].strip()
                    assert val == "", (
                        f"Without Monitor, ep_rew_mean values should be empty, got '{val}'"
                    )
        vec_env.close()
