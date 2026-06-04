"""Thread-backed VecEnv for local I/O-bound game environments."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


class ThreadedVecEnv(VecEnv):
    """Run each env in a thread instead of a Python subprocess.

    This keeps SB3's synchronous VecEnv contract but avoids the Windows
    multiprocessing pickle/pipe overhead that can make the parent Python
    process burn a full core while the child env processes sit idle.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]) -> None:
        self.envs = [_patch_env(env_fn()) for env_fn in env_fns]
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.envs),
            thread_name_prefix="sts-env",
        )
        self.waiting = False
        self.closed = False
        self._pending_futures: list[Future] | None = None
        self.reset_infos: tuple[dict[str, Any], ...] = tuple({} for _ in self.envs)

        super().__init__(
            len(self.envs),
            self.envs[0].observation_space,
            self.envs[0].action_space,
        )

    def step_async(self, actions: np.ndarray) -> None:
        if self.waiting:
            raise RuntimeError("Already waiting for pending step results.")
        self._pending_futures = [
            self.executor.submit(self._step_env, env, action)
            for env, action in zip(self.envs, actions, strict=True)
        ]
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        if not self.waiting or self._pending_futures is None:
            raise RuntimeError("step_wait() called before step_async().")

        # Use timeout so KeyboardInterrupt can be delivered on Windows.
        # Without timeout, Condition.wait() is not interruptible by signals.
        results = [future.result(timeout=300) for future in self._pending_futures]
        self._pending_futures = None
        self.waiting = False

        obs, rews, dones, infos, reset_infos = zip(*results, strict=True)
        self.reset_infos = reset_infos
        return (
            self._stack_obs(obs),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def reset(self) -> VecEnvObs:
        futures = [
            self.executor.submit(env.reset, seed=self._seeds[idx], **(
                {"options": self._options[idx]} if self._options[idx] else {}
            ))
            for idx, env in enumerate(self.envs)
        ]
        results = [future.result(timeout=300) for future in futures]
        obs, reset_infos = zip(*results, strict=True)
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return self._stack_obs(obs)

    def close(self) -> None:
        if self.closed:
            return
        # Kill all env subprocesses first — this unblocks any worker
        # thread stuck in launch_game()'s server_socket.accept().
        for env in self.envs:
            env.close()
        # Now drain any pending futures (they should finish quickly
        # since we just killed all Java processes and closed sockets).
        if self.waiting and self._pending_futures is not None:
            for future in self._pending_futures:
                try:
                    future.result(timeout=10)
                except Exception:
                    pass
            self.waiting = False
        self.executor.shutdown(wait=True)
        self.closed = True

    def get_images(self) -> list[np.ndarray | None]:
        return [env.render() for env in self.envs]

    def has_attr(self, attr_name: str) -> bool:
        try:
            self.envs[0].get_wrapper_attr(attr_name)
            return True
        except AttributeError:
            return False

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [
            self.envs[idx].get_wrapper_attr(attr_name)
            for idx in self._get_indices(indices)
        ]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for idx in self._get_indices(indices):
            setattr(self.envs[idx], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        return [
            self.envs[idx].get_wrapper_attr(method_name)(*method_args, **method_kwargs)
            for idx in self._get_indices(indices)
        ]

    def env_is_wrapped(
        self,
        wrapper_class: type[gym.Wrapper],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        return [
            is_wrapped(self.envs[idx], wrapper_class)
            for idx in self._get_indices(indices)
        ]

    def _step_env(self, env: gym.Env, action: Any) -> tuple[Any, float, bool, dict[str, Any], dict[str, Any]]:
        if getattr(env, "needs_lazy_reset", False):
            # If the env crashed in the previous cycle, we deferred the reset
            # to give the main thread a window to process Ctrl+C. Now we must
            # reset it before stepping to satisfy SB3's Monitor wrapper.
            env.needs_lazy_reset = False
            env.reset()

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reset_info: dict[str, Any] = {}
        info["TimeLimit.truncated"] = truncated and not terminated
        if done:
            info["terminal_observation"] = observation
            if info.get("crashed"):
                # Defer the expensive Java restart to the next step cycle.
                # This prevents worker threads from immediately launching Java
                # when Ctrl+C kills processes, blocking the main thread from
                # exiting cleanly.
                env.needs_lazy_reset = True
            else:
                observation, reset_info = env.reset()
        return observation, reward, done, info, reset_info

    @staticmethod
    def _stack_obs(obs: tuple[Any, ...]) -> VecEnvObs:
        if isinstance(obs[0], dict):
            return {
                key: np.stack([single_obs[key] for single_obs in obs])
                for key in obs[0].keys()
            }
        if isinstance(obs[0], tuple):
            return tuple(np.stack(items) for items in zip(*obs, strict=True))
        return np.stack(obs)
