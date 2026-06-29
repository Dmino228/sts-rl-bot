"""RLlib callbacks for run-progress metrics."""

from __future__ import annotations

from typing import Any

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except ImportError:  # pragma: no cover - older Ray fallback
    from ray.rllib.agents.callbacks import DefaultCallbacks  # type: ignore


class ProgressMetricsCallback(DefaultCallbacks):
    """Aggregate floor and act progress from env info dictionaries."""

    def on_episode_start(self, *, episode: Any, **kwargs: Any) -> None:
        episode.user_data["floor_values"] = []
        episode.user_data["max_floor"] = 0
        episode.user_data["boss_reached"] = 0.0
        episode.user_data["boss_killed"] = 0.0
        episode.user_data["act2"] = 0.0

    def on_episode_step(self, *, episode: Any, **kwargs: Any) -> None:
        self._record_progress(episode, _last_episode_info(episode))

    def on_episode_end(self, *, episode: Any, **kwargs: Any) -> None:
        self._record_progress(episode, _last_episode_info(episode))
        floors = episode.user_data.get("floor_values") or [0]
        episode.custom_metrics["floor"] = float(max(floors))
        episode.custom_metrics["max_floor"] = float(max(floors))
        episode.custom_metrics["boss_reached_pct"] = 100.0 * float(
            episode.user_data.get("boss_reached", 0.0)
        )
        episode.custom_metrics["boss_killed_pct"] = 100.0 * float(
            episode.user_data.get("boss_killed", 0.0)
        )
        episode.custom_metrics["act2_pct"] = 100.0 * float(
            episode.user_data.get("act2", 0.0)
        )

    @staticmethod
    def _record_progress(episode: Any, info: Any) -> None:
        if not isinstance(info, dict):
            return
        progress = info.get("progress_metrics")
        if not isinstance(progress, dict):
            return

        floor = _safe_float(progress.get("floor"), 0.0)
        episode.user_data.setdefault("floor_values", []).append(floor)
        episode.user_data["max_floor"] = max(
            float(episode.user_data.get("max_floor", 0.0)),
            floor,
        )
        for key in ("boss_reached", "boss_killed", "act2"):
            episode.user_data[key] = max(
                float(episode.user_data.get(key, 0.0)),
                _safe_float(progress.get(key), 0.0),
            )


def _last_episode_info(episode: Any) -> Any:
    getter = getattr(episode, "last_info_for", None)
    if callable(getter):
        for args in ((), ("agent0",), ("__default_policy__",)):
            try:
                info = getter(*args)
            except TypeError:
                continue
            if info is not None:
                return info

    infos = getattr(episode, "_last_infos", None)
    if isinstance(infos, dict) and infos:
        return next(iter(infos.values()))
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
