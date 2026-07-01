"""Benchmark the real sts2-cli Gym pipeline without Ray/PPO overhead."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env import SlayTheSpireEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real sts2-cli env throughput")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--sts2-lang", default="en")
    parser.add_argument("--sts2-cli-path", default="dotnet")
    parser.add_argument("--sts2-cli-cwd", default=r"C:\dev\sts2-cli")
    parser.add_argument(
        "--sts2-cli-arg",
        action="append",
        default=[],
        dest="sts2_cli_args",
        help="Extra argument passed to sts2-cli. Use --sts2-cli-arg=--flag for values starting with --.",
    )
    parser.add_argument("--workspace-dir", default=os.path.join(PROJECT_ROOT, "rllib_workers"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sts2-capture-stderr", action="store_true")
    parser.add_argument("--sts2-recycle-every-episodes", type=int, default=250)
    parser.add_argument("--sts2-recycle-every-steps", type=int, default=0)
    parser.add_argument("--sts2-recycle-rss-mb", type=float, default=768.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_args = args.sts2_cli_args or [
        "run",
        "--no-build",
        "--project",
        os.path.join(args.sts2_cli_cwd, "src", "Sts2Headless", "Sts2Headless.csproj"),
    ]
    rng = np.random.default_rng(42)
    envs = [_make_env(args, cli_args, index) for index in range(args.envs)]
    infos: list[dict[str, Any]] = []
    resets = 0
    reset_times: list[float] = []
    step_times: list[float] = []
    steps = 0

    start = time.perf_counter()
    try:
        for index, env in enumerate(envs):
            t0 = time.perf_counter()
            _obs, info = env.reset(seed=index)
            reset_times.append(time.perf_counter() - t0)
            infos.append(info)
            resets += 1

        while steps < args.steps:
            env_index = steps % len(envs)
            env = envs[env_index]
            info = infos[env_index]
            valid = np.flatnonzero(np.asarray(info["action_mask"]) > 0)
            action = int(rng.choice(valid))

            t0 = time.perf_counter()
            _obs, _reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - t0)
            steps += 1

            if terminated or truncated:
                t0 = time.perf_counter()
                _obs, info = env.reset(seed=steps + env_index)
                reset_times.append(time.perf_counter() - t0)
                resets += 1

            infos[env_index] = info
    finally:
        for env in envs:
            env.close()

    elapsed = time.perf_counter() - start
    p50 = float(np.percentile(step_times, 50)) if step_times else 0.0
    p95 = float(np.percentile(step_times, 95)) if step_times else 0.0
    print(f"envs:          {args.envs}")
    print(f"steps:         {steps}")
    print(f"resets:        {resets}")
    print(f"elapsed_s:     {elapsed:.3f}")
    print(f"steps/sec:     {steps / elapsed:.2f}")
    print(f"avg_step_ms:   {1000.0 * sum(step_times) / max(len(step_times), 1):.2f}")
    print(f"p50_step_ms:   {1000.0 * p50:.2f}")
    print(f"p95_step_ms:   {1000.0 * p95:.2f}")
    print(f"avg_reset_s:   {sum(reset_times) / max(len(reset_times), 1):.3f}")


def _make_env(args: argparse.Namespace, cli_args: list[str], index: int) -> SlayTheSpireEnv:
    return SlayTheSpireEnv(
        character_class=args.character,
        worker_dir=os.path.join(args.workspace_dir, f"sts2_bench_{index}"),
        worker_id=index,
        include_raw_state_in_info=False,
        include_action_mask_in_info=True,
        game_version=2,
        process_timeout=args.timeout,
        sts2_cli_path=args.sts2_cli_path,
        sts2_cli_args=cli_args,
        sts2_cli_cwd=args.sts2_cli_cwd,
        sts2_capture_stderr=args.sts2_capture_stderr,
        sts2_recycle_every_episodes=args.sts2_recycle_every_episodes,
        sts2_recycle_every_steps=args.sts2_recycle_every_steps,
        sts2_recycle_rss_mb=args.sts2_recycle_rss_mb,
        sts2_ascension=args.ascension,
        sts2_lang=args.sts2_lang,
    )


if __name__ == "__main__":
    main()
