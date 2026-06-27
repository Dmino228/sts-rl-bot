"""VecEnv wrapper that reuses action masks returned by worker steps."""

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


class CachedActionMaskVecEnv(VecEnvWrapper):
    """Avoid a per-step env_method("action_masks") IPC roundtrip.

    MaskablePPO asks vectorized envs for masks via env_method("action_masks")
    before every policy call. Our workers already compute the next mask after
    reset/step, so this wrapper serves that cached mask from the main process.
    """

    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)
        self._last_action_masks: np.ndarray | None = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self._last_action_masks = self._masks_from_reset_infos()
        if self._last_action_masks is None:
            self._last_action_masks = np.stack(self.venv.env_method("action_masks"))
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_action_masks = self._masks_from_infos(infos, dones)
        if self._last_action_masks is None:
            self._last_action_masks = np.stack(self.venv.env_method("action_masks"))
        return obs, rewards, dones, infos

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if (
            method_name == "action_masks"
            and not method_args
            and not method_kwargs
            and self._last_action_masks is not None
        ):
            return [self._last_action_masks[i].copy() for i in self._get_indices(indices)]
        return self.venv.env_method(
            method_name,
            *method_args,
            indices=indices,
            **method_kwargs,
        )

    def _masks_from_reset_infos(self) -> np.ndarray | None:
        reset_infos = getattr(self.venv, "reset_infos", None)
        if reset_infos is None:
            return None
        return self._stack_masks(reset_infos)

    def _masks_from_infos(
        self,
        infos: tuple[dict[str, Any], ...],
        dones: np.ndarray,
    ) -> np.ndarray | None:
        reset_infos = getattr(self.venv, "reset_infos", None)
        mask_sources: list[dict[str, Any]] = []
        for idx, info in enumerate(infos):
            if dones[idx] and reset_infos is not None:
                reset_info = reset_infos[idx]
                if isinstance(reset_info, dict) and "action_mask" in reset_info:
                    mask_sources.append(reset_info)
                    continue
            mask_sources.append(info)
        return self._stack_masks(mask_sources)

    @staticmethod
    def _stack_masks(infos: Any) -> np.ndarray | None:
        masks = []
        for info in infos:
            if not isinstance(info, dict) or "action_mask" not in info:
                return None
            masks.append(np.asarray(info["action_mask"], dtype=np.int8))
        return np.stack(masks)
