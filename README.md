# Slay the Spire RL Bot

Reinforcement Learning environment for training agents on Slay the Spire-style
games through Gymnasium, Ray RLlib, PyTorch, and action masking.

The project started as a CommunicationMod bot for Slay the Spire 1 and has since
grown into a multi-engine training stack:

- **StS1**: Slay the Spire + ModTheSpire + CommunicationMod, launched as isolated
  JVM workers.
- **StS2**: Slay the Spire 2 headless pipeline through
  [`sts2-cli`](https://github.com/wuhao21/sts2-cli), launched as lightweight
  .NET processes over stdin/stdout JSON.

The current primary training path is **Ray RLlib PPO**. Stable Baselines3 support
is still present under `sb3/` as the legacy/local-cluster path.

## Current Architecture

The root environment is intentionally game-agnostic. `SlayTheSpireEnv` in
`env.py` delegates game-specific work to an engine strategy selected by
`--game-version`.

```text
env.py / engine.py
  shared Gymnasium API, reward shaping, watchdog hooks

sts1/
  StS1 state encoder, action masker, CommunicationMod/JVM process manager

sts2/
  StS2 state encoder, JSON command action mapper, sts2-cli process manager,
  SpireCodex-derived card/relic/potion/monster metadata

rllib/
  train_rllib.py    – main training orchestrator (PPO loop, checkpoints, eval)
  config.py         – CLI arg parsing, config resolution, env config builder
  presets.py        – built-in experiment presets & YAML config loader
  console.py        – TrainingConsole with compact/verbose/quiet display modes
  run_folder.py     – timestamped run dirs, metrics.jsonl, crash bundles
  preflight.py      – pre-Ray validation checks & experiment summary
  progress_metrics.py – RLlib callback for grouped combat & full-run metrics
  action_mask_model.py – custom Torch model with action masking
  env_wrapper.py    – RLlib env registration, ring buffer, crash diagnostics
  benchmark_sts2_env.py – standalone STS2 Gym benchmark (no Ray)

sb3/
  legacy Stable Baselines3 training scripts and VecEnv implementations

docs/
  architecture notes and historical design decisions
```

The neural network sees only normalized tensors and action masks. It should not
need to know whether the backing game is StS1 or StS2.

## Features

- Gymnasium-compatible environment with fixed discrete action space.
- Binary action masks for combat, map, rewards, shops, events, rest sites, and
  STS2 card-selection decisions.
- Ray RLlib PPO training with a custom Torch action-mask model.
- **Preset system** – launch common experiments with `--preset <name>` instead
  of writing 12+ CLI flags.
- **YAML config files** – define reusable custom configurations in a `.yaml`
  file and load with `--config`.
- **Console output modes** – `compact` (Rich live progress bar + table),
  `verbose` (full log lines), `quiet` (errors only).
- **Run folders** – each run creates a timestamped directory under `runs/` with
  `config.resolved.yaml`, `train.log`, and `metrics.jsonl`.
- **Grouped combat metrics** – win rates split by encounter category (weak,
  normal, elite, boss).
- **Crash debug bundles** – ring buffer of last 50 actions, state snapshot,
  stderr tail, config saved on crash.
- Per-game RLlib checkpoints in `models/rllib/sts1` and `models/rllib/sts2`.
- Watchdog-style recovery for crashed/stalled game processes.
- StS2 process recycling by episode count, step count, or RSS threshold to
  survive long runs while upstream headless memory leaks are being chased.
  Default: recycle every 15 000 episodes (~1 hour), or at 768 MB RSS.

## Requirements

General:

- Windows is the best-tested target for local training.
- Python 3.12+.
- PyTorch, Ray RLlib, Gymnasium, Stable Baselines3, and test dependencies from
  `requirements.txt`.

For StS1:

- Slay the Spire.
- ModTheSpire, BaseMod, StSLib, CommunicationMod.
- A prepared local/portable game directory for worker cloning, usually
  `SlayTheSpire/`.

For StS2:

- .NET 9 SDK.
- A local checkout of `sts2-cli`, commonly `C:\dev\sts2-cli`.
- `sts2-cli` / `Sts2Headless` buildable from:
  `C:\dev\sts2-cli\src\Sts2Headless\Sts2Headless.csproj`.

## Setup

Create or activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Verify the Python-side test suite:

```powershell
python -m pytest test -q -k "not smoke_training"
```

For StS2, also verify the headless project:

```powershell
dotnet build C:\dev\sts2-cli\src\Sts2Headless\Sts2Headless.csproj
```

## RLlib Training

### Quick Start with Presets

The fastest way to launch an experiment is with a built-in preset. Only the
`--sts2-cli-path` flag is required alongside the preset:

```powershell
# Smoke test (2 workers, 50k steps, fixed encounter)
python rllib\train_rllib.py --preset combat_smoke_fixed `
  --sts2-cli-path C:\dev\sts2-cli\src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.exe

# Real training (8 workers, 1M steps, act1 mixed pool)
python rllib\train_rllib.py --preset combat_train_act1_mixed `
  --sts2-cli-path C:\dev\sts2-cli\src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.exe

# Debug mode (1 worker, verbose logging, debug episodes)
python rllib\train_rllib.py --preset combat_debug_fixed `
  --sts2-cli-path C:\dev\sts2-cli\src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.exe

# Full run with hard non-combat heuristic
python rllib\train_rllib.py --preset fullrun_ironclad_heuristic_hard `
  --sts2-cli-path C:\dev\sts2-cli\src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.exe
```

CLI flags override preset values:

```powershell
python rllib\train_rllib.py --preset combat_train_act1_mixed `
  --sts2-cli-path <path> --workers 4 --timesteps 500000
```

### Available Presets

| Preset | Mode | Pool | Workers | Steps | Console |
|--------|------|------|---------|-------|---------|
| `combat_smoke_fixed` | combat | fixed | 2 | 50K | compact |
| `combat_debug_fixed` | combat | fixed | 1 | 10K | verbose |
| `combat_train_act1_mixed` | combat | act1_mixed | 8 | 1M | compact |
| `combat_train_act1_mixed_random_deck` | combat | act1_mixed | 8 | 1M | compact |
| `combat_train_all_mixed_random_deck` | combat | all_mixed | 8 | 1M | compact |
| `combat_eval_act1_mixed` | combat | act1_mixed | 1 | 0 | compact |
| `fullrun_ironclad` | full_run | n/a | 8 | 10M | compact |
| `fullrun_ironclad_heuristic_hard` | full_run | n/a | 8 | 10M | compact |

List presets and inspect resolved config without starting training:

```powershell
python rllib\train_rllib.py --list-presets
python rllib\train_rllib.py --preset combat_smoke_fixed --sts2-cli-path <path> --dry-run
```

### YAML Config Files

For reusable custom configurations, define a YAML file with underscored key
names matching the CLI destinations:

```yaml
# my_experiment.yaml
game_version: "2"
sts2_curriculum_mode: combat
sts2_reward_mode: combat_sparse
sts2_combat_enemy_pool: act1_mixed
workers: 8
timesteps: 2000000
eval_combat_episodes: 500
eval_combat_freq: 10
```

```powershell
python rllib\train_rllib.py --config my_experiment.yaml --sts2-cli-path <path>
```

Resolution order: **preset defaults < YAML file < CLI flags**. CLI always wins.

### Console Output Modes

Control training output with `--console <mode>`:

- **compact** (default for presets): Rich live-updating progress bar with
  steps/s display, plus a metrics table that replaces itself in-place.
  No scroll spam.
- **verbose**: Full log lines with all metrics printed per iteration.
- **quiet**: Only warnings and errors to console; full detail goes to log files.

Compact combat layout example:

```
⠋ Training ━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━ 15,200/50,000 steps (250 steps/s) 00:01:05 ETA: 00:02:30
┌──────┬────────┬────────┬───────────┬────────────┬────────────┬───────────┬───────────┬─────────┬───────────┬──────────┬─────────┐
│ iter │  steps │ reward │ train_win │ train_loss │ train_tmout│ avg_hp_lost│ avg_steps│ weak_wr │ normal_wr │ elite_wr │ boss_wr │
├──────┼────────┼────────┼───────────┼────────────┼────────────┼───────────┼───────────┼─────────┼───────────┼──────────┼─────────┤
│   15 │ 15,200 │   0.96 │      0.98 │       0.02 │       0.00 │     31.13 │     22.04 │    0.98 │      0.00 │     0.00 │    0.00 │
└──────┴────────┴────────┴───────────┴────────────┴────────────┴───────────┴───────────┴─────────┴───────────┴──────────┴─────────┘
```

### Run Folders

Each training run creates a timestamped directory under `runs/`:

```
runs/20260702_203800_combat_c1_ironclad_starter_act1_mixed/
  config.resolved.yaml   # full config snapshot
  train.log              # complete training log
  metrics.jsonl          # one JSON line per iteration
  crashes/               # crash debug bundles (on-demand)
```

### Manual Training (Advanced)

The original full-flag syntax still works for fine-grained control:

```powershell
python rllib\train_rllib.py `
  --game-version 2 `
  --sts2-cli-path C:\dev\sts2-cli\src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.exe `
  --workers 8 `
  --envs-per-worker 1 `
  --character Ironclad `
  --training-stage combat_c0_ironclad_starter_act1 `
  --deck-mode starter `
  --enemy-pool act1 `
  --sts2-curriculum-mode combat `
  --sts2-reward-mode combat_sparse `
  --sts2-combat-enemy-pool act1_hallway `
  --eval-random-baseline 500 `
  --eval-combat-episodes 500 `
  --eval-combat-freq 10 `
  --eval-combat-deterministic `
  --timesteps 1000000
```

### Combat Curriculum

In combat curriculum mode (`--sts2-curriculum-mode combat`), one episode is
exactly one fight. The episode terminates immediately on combat win, loss, or
timeout.

Reward modes:

- `full_v3_2`: legacy full-run reward, preserved for compatibility.
- `combat_sparse`: `+1` win, `−1` loss/timeout, `0` otherwise.
- `combat_dense`: terminal sparse reward plus small configurable damage dealt,
  HP lost, and action-penalty shaping.

Encounter pools:

| Pool | Description |
|------|-------------|
| `fixed` | Single encounter via `--sts2-combat-encounter`; smoke tests only |
| `act1_hallway` | Sampled Act 1 hallway combats |
| `act1_elite` | Sampled Act 1 elites |
| `act1_boss` | Sampled Act 1 bosses observed from `get_map` |
| `act1_hallway_elite` | Hallway-heavy pool with elites |
| `act1_mixed` | Hallway-heavy mix with elites and bosses |

### Heuristic Modes

`--heuristic-mode hard` is the combat-focused curriculum: PPO controls combat
while a deterministic STS2 heuristic hard-selects non-combat actions through the
action mask. `--heuristic-mode mask --heuristic-top-k N` keeps the top N
heuristic-ranked non-combat actions, which is the planned bridge toward PPO
learning strategic choices directly.

### StS1 RLlib Training

StS1 uses the legacy Java/CommunicationMod process manager behind the same RLlib
entrypoint:

```powershell
python rllib\train_rllib.py `
  --game-version 1 `
  --base-env-dir C:\dev\sts rl bot\SlayTheSpire `
  --workers 4 `
  --envs-per-worker 1 `
  --character IRONCLAD `
  --timesteps 1000000
```

StS1 workers are heavier. Start with fewer workers and scale after confirming
that worker directories and ports are isolated correctly.

### Checkpoints and Resume

By default, RLlib writes checkpoints per game:

```text
models/rllib/sts1/
models/rllib/sts2/
```

Training auto-resumes from the latest checkpoint in the default directory unless
`--no-auto-resume` is passed.

When `--training-stage` is provided and `--checkpoint-dir` is not, RLlib writes
to `models/rllib/<game>/<stage>/`. Each saved RLlib checkpoint also receives a
`checkpoint_metadata.json` recording the stage, character, deck/enemy labels,
preset name, total steps, source checkpoint, heuristic mode, worker/batch
settings, and StS2 process recycling settings.

Useful checkpoint flags:

```powershell
--checkpoint-freq 10
--training-stage combat_c0_ironclad_starter_act1
--deck-mode starter
--enemy-pool act1
--run-notes "Starter combat-only baseline"
--checkpoint-dir C:\path\to\custom\checkpoint_dir
--resume-from C:\path\to\specific\checkpoint
--no-auto-resume
--init-from-sb3 C:\path\to\old_sb3_model.zip
```

### Process Recycling

StS2 workers are recycled between episodes to guard against upstream memory
leaks. Recycling is checked at `reset()`.

| Flag | Default | Description |
|------|---------|-------------|
| `--sts2-recycle-every-episodes` | 15 000 | Recycle after N episodes (~1 hour) |
| `--sts2-recycle-every-steps` | 0 (off) | Recycle after N env steps |
| `--sts2-recycle-rss-mb` | 768 | Recycle when process RSS exceeds this |

Set any limit to `0` to disable it. Recycling logs a short one-line notice
(PID + reason) to the console. Full diagnostics are saved only in crash bundles.

### Eval and Benchmarks

```powershell
--eval-random-baseline 500       # random baseline episodes (runs at startup)
--eval-random-baseline-freq N    # re-run random baseline every N iterations
--eval-combat-episodes 500       # PPO eval episodes per eval round
--eval-combat-freq 10            # evaluate every N train iterations
--eval-combat-deterministic      # use deterministic eval actions
```

The random baseline uses valid random actions on the same pool. PPO combat eval
runs outside the training sampler and logs per-encounter win rate, HP lost, and
combat steps. In STS2 combat curriculum, PPO eval and checkpointing default to
every 10 train iterations.

## STS2 Benchmark Helper

To test the real STS2 Gym pipeline without Ray/PPO overhead:

```powershell
python rllib\benchmark_sts2_env.py `
  --envs 4 `
  --steps 2000 `
  --sts2-cli-path dotnet `
  --sts2-cli-cwd C:\dev\sts2-cli
```

This is useful when separating RLlib overhead from IPC/headless engine overhead.

## Legacy CommunicationMod Mode

The original `main.py` CommunicationMod loop still exists for simple/manual StS1
integration. In that mode CommunicationMod owns the parent process and launches
Python from its config file.

CommunicationMod stdout/stderr rule:

- `stdout` is reserved for game commands.
- developer logs must go to `stderr`.

Example config entry:

```properties
command=py "C\:\\dev\\sts rl bot\\main.py"
```

Most current training work should use `rllib/train_rllib.py` instead.

## Development Workflow

Before changing architecture or engine behavior, read the relevant files under
`docs/`. They are the source of truth for current design intent.

Common checks:

```powershell
python -m py_compile env.py engine.py sts1\engine.py sts2\engine.py sts2\process_manager.py rllib\train_rllib.py
python -m pytest test\test_sts2_json_protocol.py -q
python -m pytest test\test_rllib_integration.py -q
python -m pytest test\test_presets.py -q
python -m pytest test -q -k "not smoke_training"
```

## Notes

- The shared action space is size `100`; STS2 currently uses the same external
  action count while mapping commands to JSON.
- Current STS1 observations are `205` floats; current STS2 observations are
  `349` floats.
- Reward shaping is V3.2-style: floor progress, combat victory, relics,
  upgrades/removals, signed HP deltas, anti-stall pressure, death penalty, and
  act-completion reward.
- On an 8C/16T CPU, 8 STS2 workers has been a safer baseline than immediately
  jumping to 16 workers. Use logs and the progress bar `steps/s` to tune from
  there.
