"""Run folder management for reproducible experiment tracking.

Each training run gets a timestamped directory under ``runs/`` containing:

- ``config.resolved.yaml`` — full resolved configuration snapshot
- ``train.log`` — training log (file handler target)
- ``metrics.jsonl`` — one JSON line per training iteration
- ``crashes/`` — crash debug bundles (on-demand)
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import deque
from typing import Any


def create_run_folder(
    base_dir: str,
    experiment_name: str,
    resolved_config: dict[str, Any],
) -> "RunFolder":
    """Create a timestamped run directory and save the resolved config."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_path_component(experiment_name) or "experiment"
    folder_name = f"{timestamp}_{safe_name}"
    run_dir = os.path.join(os.path.abspath(base_dir), folder_name)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.resolved.yaml")
    _save_yaml(config_path, resolved_config)

    return RunFolder(run_dir)


class RunFolder:
    """Handle to a run directory with convenience writers."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(self.path, exist_ok=True)

    @property
    def train_log_path(self) -> str:
        return os.path.join(self.path, "train.log")

    @property
    def metrics_jsonl_path(self) -> str:
        return os.path.join(self.path, "metrics.jsonl")

    def save_metrics_line(self, data: dict[str, Any]) -> None:
        """Append one JSON-lines record to ``metrics.jsonl``."""
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, default=_json_default))
            fh.write("\n")

    def save_crash_bundle(
        self,
        *,
        ring_buffer: deque[dict[str, Any]] | list[dict[str, Any]] | None = None,
        last_observation: Any | None = None,
        stderr_tail: str = "",
        resolved_config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Save a crash debug bundle and return its directory path."""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crash_dir = os.path.join(self.path, "crashes", timestamp)
        os.makedirs(crash_dir, exist_ok=True)

        if ring_buffer is not None:
            actions_path = os.path.join(crash_dir, "last_actions.jsonl")
            with open(actions_path, "w", encoding="utf-8") as fh:
                for entry in ring_buffer:
                    fh.write(json.dumps(entry, default=_json_default))
                    fh.write("\n")

        if last_observation is not None:
            state_path = os.path.join(crash_dir, "last_state.json")
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(
                    _observation_to_serializable(last_observation),
                    fh,
                    indent=2,
                    default=_json_default,
                )
                fh.write("\n")

        if stderr_tail:
            stderr_path = os.path.join(crash_dir, "stderr_tail.txt")
            with open(stderr_path, "w", encoding="utf-8") as fh:
                fh.write(stderr_tail)

        if resolved_config is not None:
            config_path = os.path.join(crash_dir, "config.resolved.yaml")
            _save_yaml(config_path, resolved_config)

        if extra:
            extra_path = os.path.join(crash_dir, "extra.json")
            with open(extra_path, "w", encoding="utf-8") as fh:
                json.dump(extra, fh, indent=2, default=_json_default)
                fh.write("\n")

        return crash_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_path_component(value: str) -> str:
    text = str(value or "").strip().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in text).strip("._")


def _save_yaml(path: str, data: dict[str, Any]) -> None:
    import yaml

    serializable = _make_serializable(data)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(serializable, fh, default_flow_style=False, sort_keys=True)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable types for YAML/JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy types and others."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def _observation_to_serializable(obs: Any) -> Any:
    """Convert an observation (possibly dict of arrays) to JSON-safe form."""
    if isinstance(obs, dict):
        return {str(k): _observation_to_serializable(v) for k, v in obs.items()}
    try:
        import numpy as np

        if isinstance(obs, np.ndarray):
            if obs.size > 200:
                return {
                    "shape": list(obs.shape),
                    "dtype": str(obs.dtype),
                    "sample_first_20": obs.flat[:20].tolist(),
                }
            return obs.tolist()
    except ImportError:
        pass
    return str(obs)
