"""Training console output with compact/verbose/quiet modes.

``compact`` mode uses ``rich`` for a live-updating progress bar and metrics
table that replaces itself in-place (no scroll spam).

``verbose`` mode uses standard ``logging.info()`` lines with all metrics
(original behaviour).

``quiet`` mode suppresses iteration output to console; everything still
goes to the log file.
"""

from __future__ import annotations

import logging
import time
from typing import Any


class TrainingConsole:
    """Mode-aware training output manager.

    Instantiate once before the training loop; call ``on_iteration`` after
    each ``algo.train()`` return.
    """

    def __init__(
        self,
        *,
        mode: str = "compact",
        curriculum_mode: str = "full_run",
        target_steps: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.mode = mode.strip().lower()
        self.curriculum_mode = curriculum_mode.strip().lower()
        self.target_steps = max(0, target_steps)
        self.logger = logger or logging.getLogger("train_rllib")
        self._started_at = time.perf_counter()
        self._last_metrics: dict[str, Any] = {}

        # Rich live display (compact mode only)
        self._live: Any | None = None
        self._progress: Any | None = None
        self._task_id: Any | None = None

        if self.mode == "compact":
            self._init_rich()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_iteration(
        self,
        *,
        iteration: int,
        current_steps: int,
        step_delta: int,
        iteration_seconds: float,
        reward_mean: Any,
        progress_metrics: dict[str, str],
        combat_metrics: dict[str, str],
        grouped_combat_metrics: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Process one training iteration and return the metrics dict
        that should be written to ``metrics.jsonl``."""
        sps = step_delta / iteration_seconds if iteration_seconds > 0 else 0.0

        metrics_line = {
            "iteration": iteration,
            "env_steps": current_steps,
            "step_delta": step_delta,
            "iter_seconds": round(iteration_seconds, 2),
            "steps_per_sec": round(sps, 1),
            "reward_mean": _safe_number(reward_mean),
            **{f"progress_{k}": v for k, v in progress_metrics.items()},
            **{f"combat_{k}": v for k, v in combat_metrics.items()},
        }
        if grouped_combat_metrics:
            metrics_line.update(
                {f"grouped_{k}": v for k, v in grouped_combat_metrics.items()}
            )
        self._last_metrics = metrics_line

        if self.mode == "compact":
            self._update_compact(
                iteration=iteration,
                current_steps=current_steps,
                sps=sps,
                reward_mean=reward_mean,
                progress_metrics=progress_metrics,
                combat_metrics=combat_metrics,
                grouped_combat_metrics=grouped_combat_metrics,
            )
        elif self.mode == "verbose":
            self._log_verbose(
                iteration=iteration,
                current_steps=current_steps,
                step_delta=step_delta,
                iteration_seconds=iteration_seconds,
                sps=sps,
                reward_mean=reward_mean,
                progress_metrics=progress_metrics,
                combat_metrics=combat_metrics,
                grouped_combat_metrics=grouped_combat_metrics,
            )
        # quiet: nothing to console

        return metrics_line

    def on_eval(self, eval_type: str, metrics: dict[str, Any]) -> None:
        """Show eval results according to current mode."""
        if self.mode == "quiet":
            return

        if self.mode == "compact" and self._live is not None:
            self._live.console.print(
                _format_eval_compact(eval_type, metrics),
                highlight=False,
            )
        else:
            self.logger.info(
                "%s eval: %s",
                eval_type,
                _format_eval_verbose(metrics),
            )

    def on_finish(self, summary: dict[str, Any]) -> None:
        """Print final training summary."""
        self.close()
        elapsed = time.perf_counter() - self._started_at
        self.logger.info(
            "Training complete. Total time: %.1fs  Final steps: %s  Checkpoint: %s",
            elapsed,
            summary.get("total_steps", "n/a"),
            summary.get("checkpoint_path", "n/a"),
        )

    def close(self) -> None:
        """Stop the rich live display if active."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    # ------------------------------------------------------------------
    # Rich compact mode
    # ------------------------------------------------------------------

    def _init_rich(self) -> None:
        try:
            from rich.live import Live
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            from rich.table import Table
            from rich.console import Group

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("steps"),
                TextColumn("[green]({task.fields[sps]:.0f} steps/s)"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(),
            )
            self._task_id = self._progress.add_task(
                "Training",
                total=self.target_steps if self.target_steps > 0 else None,
                sps=0.0,
            )
            self._rich_table = self._make_empty_table()
            self._rich_Group = Group
            self._live = Live(
                Group(self._progress, self._rich_table),
                refresh_per_second=1,
                auto_refresh=False,
                screen=True,
                transient=True,
                vertical_overflow="crop",
            )
            self._live.start()
        except ImportError:
            self.logger.warning(
                "rich not available; falling back to verbose console mode."
            )
            self.mode = "verbose"

    def _make_empty_table(self) -> Any:
        from rich.table import Table

        table = Table(
            show_header=True,
            header_style="bold cyan",
            expand=False,
            padding=(0, 1),
        )
        if self.curriculum_mode == "combat":
            for col in [
                "iter", "steps", "reward",
                "train_win", "train_loss", "train_tmout",
                "avg_hp_lost", "avg_steps",
                "weak_wr", "normal_wr", "elite_wr", "boss_wr",
            ]:
                table.add_column(col, justify="right")
        else:
            for col in [
                "iter", "steps", "reward",
                "floor", "max_floor", "boss_r%", "boss_k%", "act2%",
            ]:
                table.add_column(col, justify="right")
        return table

    def _update_compact(
        self,
        *,
        iteration: int,
        current_steps: int,
        sps: float,
        reward_mean: Any,
        progress_metrics: dict[str, str],
        combat_metrics: dict[str, str],
        grouped_combat_metrics: dict[str, str] | None,
    ) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=current_steps, sps=sps)

        from rich.table import Table
        from rich.console import Group

        table = self._make_empty_table()

        if self.curriculum_mode == "combat":
            gm = grouped_combat_metrics or {}
            table.add_row(
                str(iteration),
                f"{current_steps:,}",
                _fmt(reward_mean),
                combat_metrics.get("combat_win_rate", "n/a"),
                combat_metrics.get("combat_loss_rate", "n/a"),
                combat_metrics.get("combat_timeout_rate", "n/a"),
                combat_metrics.get("avg_hp_lost", "n/a"),
                combat_metrics.get("avg_combat_steps", "n/a"),
                gm.get("weak_win_rate", "n/a"),
                gm.get("normal_win_rate", "n/a"),
                gm.get("elite_win_rate", "n/a"),
                gm.get("boss_win_rate", "n/a"),
            )
        else:
            table.add_row(
                str(iteration),
                f"{current_steps:,}",
                _fmt(reward_mean),
                progress_metrics.get("floor_mean", "n/a"),
                progress_metrics.get("max_floor", "n/a"),
                progress_metrics.get("boss_reached_pct", "n/a"),
                progress_metrics.get("boss_killed_pct", "n/a"),
                progress_metrics.get("act2_pct", "n/a"),
            )

        self._rich_table = table
        if self._live is not None:
            self._live.update(Group(self._progress, table), refresh=True)

    # ------------------------------------------------------------------
    # Verbose mode (original log lines)
    # ------------------------------------------------------------------

    def _log_verbose(
        self,
        *,
        iteration: int,
        current_steps: int,
        step_delta: int,
        iteration_seconds: float,
        sps: float,
        reward_mean: Any,
        progress_metrics: dict[str, str],
        combat_metrics: dict[str, str],
        grouped_combat_metrics: dict[str, str] | None,
    ) -> None:
        if self.curriculum_mode == "combat":
            parts = [
                f"iter={iteration}",
                f"steps={current_steps} (+{step_delta})",
                f"iter_s={iteration_seconds:.2f}",
                f"sps={sps:.1f}",
                f"reward={_fmt(reward_mean)}",
                f"win={combat_metrics.get('combat_win_rate', 'n/a')}",
                f"loss={combat_metrics.get('combat_loss_rate', 'n/a')}",
                f"timeout={combat_metrics.get('combat_timeout_rate', 'n/a')}",
                f"hp_lost={combat_metrics.get('avg_hp_lost', 'n/a')}",
                f"boss_hp_left_loss={combat_metrics.get('avg_boss_hp_remaining_on_loss', 'n/a')}",
                f"boss_hp_removed={combat_metrics.get('avg_boss_hp_fraction_removed', 'n/a')}",
                f"dmg_total={combat_metrics.get('avg_damage_dealt_total', 'n/a')}",
                f"turns={combat_metrics.get('avg_turns_survived', 'n/a')}",
                f"combat_steps={combat_metrics.get('avg_combat_steps', 'n/a')}",
            ]
            if grouped_combat_metrics:
                for key in ("weak_win_rate", "normal_win_rate", "elite_win_rate", "boss_win_rate"):
                    val = grouped_combat_metrics.get(key, "n/a")
                    parts.append(f"{key}={val}")
            self.logger.info(" | ".join(parts))
        else:
            parts = [
                f"iter={iteration}",
                f"steps={current_steps} (+{step_delta})",
                f"iter_s={iteration_seconds:.2f}",
                f"sps={sps:.1f}",
                f"reward={_fmt(reward_mean)}",
                f"floor={progress_metrics.get('floor_mean', 'n/a')}",
                f"max_floor={progress_metrics.get('max_floor', 'n/a')}",
                f"boss_r%={progress_metrics.get('boss_reached_pct', 'n/a')}",
                f"boss_k%={progress_metrics.get('boss_killed_pct', 'n/a')}",
                f"act2%={progress_metrics.get('act2_pct', 'n/a')}",
            ]
            self.logger.info(" | ".join(parts))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_eval_compact(eval_type: str, metrics: dict[str, Any]) -> str:
    """One-line eval summary for compact mode."""
    parts = [f"[bold]{eval_type}[/bold]"]
    for key in ("combat_win_rate", "avg_hp_lost", "avg_combat_steps"):
        val = metrics.get(key)
        if val is not None:
            parts.append(f"{key}={_fmt(val)}")

    grouped = metrics.get("grouped_metrics")
    if isinstance(grouped, dict):
        for cat in ("weak", "normal", "elite", "boss"):
            wr = grouped.get(f"{cat}_win_rate")
            if wr is not None:
                parts.append(f"{cat}_wr={_fmt(wr)}")

    return " | ".join(parts)


def _format_eval_verbose(metrics: dict[str, Any]) -> str:
    """Multi-field eval string for verbose mode."""
    parts = []
    for key, val in sorted(metrics.items()):
        if key in ("grouped_metrics",):
            continue
        parts.append(f"{key}={_fmt(val)}")
    return " ".join(parts)
