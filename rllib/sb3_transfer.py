"""Best-effort SB3-to-RLlib policy weight initialization helpers."""

from __future__ import annotations

import logging
from typing import Any


def try_transfer_sb3_policy(
    algo: Any,
    sb3_model_path: str,
    logger: logging.Logger,
) -> bool:
    """Copy compatible SB3 MaskablePPO policy tensors into the RLlib model.

    SB3 and RLlib do not share optimizer state, rollout buffers, or exactly the
    same module layout. When layer shapes match, this provides a warm start for
    the RLlib policy. If they do not, training continues from RLlib init.
    """
    try:
        import torch
        from sb3_contrib import MaskablePPO
    except Exception as exc:
        logger.warning("SB3 transfer skipped; sb3_contrib/torch unavailable: %s", exc)
        return False

    try:
        sb3_model = MaskablePPO.load(sb3_model_path, device="cpu")
        source_state = sb3_model.policy.state_dict()
        rllib_model = algo.get_policy().model
        target_state = rllib_model.state_dict()
    except Exception as exc:
        logger.warning("SB3 transfer skipped; could not load models: %s", exc)
        return False

    copied: list[tuple[str, str]] = []
    used_sources: set[str] = set()
    new_target_state = dict(target_state)

    explicit_pairs = _explicit_policy_pairs(source_state, target_state)
    for source_key, target_key in explicit_pairs:
        tensor = source_state[source_key]
        if tensor.shape == target_state[target_key].shape:
            new_target_state[target_key] = tensor.detach().clone()
            used_sources.add(source_key)
            copied.append((source_key, target_key))

    already_copied_targets = {target for _, target in copied}
    for target_key, target_tensor in target_state.items():
        if target_key in already_copied_targets:
            continue
        source_key = _find_unused_same_shape_source(
            source_state,
            target_tensor.shape,
            used_sources,
        )
        if source_key is None:
            continue
        new_target_state[target_key] = source_state[source_key].detach().clone()
        used_sources.add(source_key)
        copied.append((source_key, target_key))

    if not copied:
        logger.warning(
            "SB3 transfer found no shape-compatible tensors. "
            "RLlib will train from a fresh initialization."
        )
        return False

    rllib_model.load_state_dict(new_target_state, strict=False)
    logger.info("Transferred %d SB3 tensor(s) into RLlib policy.", len(copied))
    if logger.isEnabledFor(logging.DEBUG):
        for source_key, target_key in copied:
            logger.debug("SB3 transfer: %s -> %s", source_key, target_key)
    del torch
    return True


def _explicit_policy_pairs(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    source_candidates = [
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    ]
    target_candidates = [
        key
        for key in target_state
        if "internal_model" in key
        and ("_hidden_layers" in key or "_logits" in key or "logits" in key)
    ]
    for source_key, target_key in zip(source_candidates, target_candidates):
        if source_key in source_state and target_key in target_state:
            pairs.append((source_key, target_key))
    return pairs


def _find_unused_same_shape_source(
    source_state: dict[str, Any],
    target_shape: Any,
    used_sources: set[str],
) -> str | None:
    preferred_fragments = ("policy_net", "action_net", "mlp_extractor")
    for source_key, tensor in source_state.items():
        if source_key in used_sources:
            continue
        if not any(fragment in source_key for fragment in preferred_fragments):
            continue
        if tensor.shape == target_shape:
            return source_key
    return None
