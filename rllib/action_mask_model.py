"""RLlib TorchModelV2 that applies the STS action mask to policy logits."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN


ACTION_MASK_MODEL = "sts_torch_action_mask_model"


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """Mask invalid discrete actions by adding -inf to their logits."""

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: dict[str, Any],
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", obs_space)
        if not isinstance(original_space, gym.spaces.Dict):
            raise TypeError(
                "TorchActionMaskModel requires a Dict observation space with "
                "'observations' and 'action_mask' keys."
            )
        observation_space = original_space["observations"]
        self.internal_model = FullyConnectedNetwork(
            observation_space,
            action_space,
            num_outputs,
            model_config,
            f"{name}_fc",
        )

    def forward(
        self,
        input_dict: dict[str, Any],
        state: list[Any],
        seq_lens: Any,
    ) -> tuple[torch.Tensor, list[Any]]:
        obs = input_dict["obs"]
        observations = obs["observations"]
        action_mask = obs["action_mask"].to(dtype=torch.float32)

        logits, state = self.internal_model(
            {"obs": observations},
            state,
            seq_lens,
        )
        inf_mask = torch.where(
            action_mask > 0.0,
            torch.zeros_like(action_mask),
            torch.full_like(action_mask, FLOAT_MIN),
        )
        return logits + inf_mask, state

    def value_function(self) -> torch.Tensor:
        return self.internal_model.value_function()


def register_action_mask_model() -> None:
    """Register the model under the stable name used by train_rllib.py."""
    ModelCatalog.register_custom_model(ACTION_MASK_MODEL, TorchActionMaskModel)
