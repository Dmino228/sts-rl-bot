import os
import sys
import gymnasium as gym
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3.threaded_vec_env import ThreadedVecEnv

class MockEnv(gym.Env):
    def __init__(self):
        self.reset_called = 0
        self.step_called = 0
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        self.reset_called += 1
        return [0.0], {}

    def step(self, action):
        self.step_called += 1
        # Returns a normal step
        return [1.0], 1.0, False, False, {}

def test_step_env_lazy_reset():
    """Verify that _step_env uses lazy reset when info['crashed'] is true."""
    env = MockEnv()
    
    # Mock a dummy ThreadedVecEnv without starting threads
    vec_env = MagicMock()
    
    # Scenario 1: Normal episode end -> auto-reset immediately
    env.step = MagicMock(return_value=([1.0], 1.0, True, False, {}))
    obs, rew, done, info, reset_info = ThreadedVecEnv._step_env(vec_env, env, 0)
    assert done is True
    assert env.reset_called == 1
    assert not getattr(env, "needs_lazy_reset", False)
    
    # Scenario 2: Crash -> lazy reset deferred
    env.reset_called = 0
    env.step = MagicMock(return_value=([0.0], 0.0, True, False, {"crashed": True}))
    obs, rew, done, info, reset_info = ThreadedVecEnv._step_env(vec_env, env, 0)
    assert done is True
    # Reset is NOT called yet!
    assert env.reset_called == 0
    # Flag is set
    assert getattr(env, "needs_lazy_reset", False) is True

    # Scenario 3: Next step cycle -> resets FIRST, then steps
    env.reset_called = 0
    env.step = MagicMock(return_value=([2.0], 1.0, False, False, {}))
    obs, rew, done, info, reset_info = ThreadedVecEnv._step_env(vec_env, env, 0)
    
    # Reset was called at the top of _step_env
    assert env.reset_called == 1
    # Flag was cleared
    assert getattr(env, "needs_lazy_reset", False) is False
    # Then step was called
    assert env.step.called
    assert obs == [2.0]
