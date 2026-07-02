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
  Ray environment registration, action-mask model, PPO training script,
  progress metrics, smoke tests, STS2 benchmark helper

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
- Per-game RLlib checkpoints in `models/rllib/sts1` and `models/rllib/sts2`.
- Training logs in `logs/`.
- Progress metrics in RLlib logs:
  `steps`, `reward_mean`, `floor_mean`, `max_floor`, `boss_reached%`,
  `boss_killed%`, and `act2%`.
- Watchdog-style recovery for crashed/stalled game processes.
- StS2 process recycling by episode count, step count, or RSS threshold to
  survive long runs while upstream headless memory leaks are being chased.

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

### StS2 Headless Training

This is the fastest and currently most active path.

```powershell
python rllib\train_rllib.py `
  --game-version 2 `
  --sts2-cli-path dotnet `
  --sts2-cli-cwd C:\dev\sts2-cli `
  --sts2-cli-arg=run `
  --sts2-cli-arg=--no-build `
  --sts2-cli-arg=--project `
  --sts2-cli-arg=C:\dev\sts2-cli\src\Sts2Headless\Sts2Headless.csproj `
  --workers 8 `
  --envs-per-worker 1 `
  --character Ironclad `
  --timesteps 1000000
```

Important PowerShell detail: when a repeated argument value starts with `--`,
pass it as `--sts2-cli-arg=--no-build`, not `--sts2-cli-arg --no-build`.

Useful StS2 flags:

```powershell
--multi-character
--character Ironclad
--character Necrobinder
--ascension 0
--num-gpus 1
--heuristic-mode hard
--heuristic-mode mask
--heuristic-top-k 2
--sts2-curriculum-mode combat
--sts2-combat-encounter SHRINKER_BEETLE_WEAK
--sts2-capture-stderr
--sts2-recycle-every-episodes 250
--sts2-recycle-every-steps 0
--sts2-recycle-rss-mb 768
```

`--heuristic-mode hard` is the first curriculum experiment: PPO controls combat,
while a deterministic STS2 heuristic hard-selects non-combat actions through the
action mask. `--heuristic-mode mask --heuristic-top-k N` keeps the top N
heuristic-ranked non-combat actions, which is the planned bridge toward PPO
learning strategic choices directly.

The recycle limits are checked between runs, during `reset()`. Set any recycle
limit to `0` to disable it.

For combat-only curriculum runs, keep the normal StS2 launch arguments and add
the curriculum labels:

```powershell
python rllib\train_rllib.py `
  --game-version 2 `
  --sts2-cli-path dotnet `
  --sts2-cli-cwd C:\dev\sts2-cli `
  --sts2-cli-arg=run `
  --sts2-cli-arg=--no-build `
  --sts2-cli-arg=--project `
  --sts2-cli-arg=C:\dev\sts2-cli\src\Sts2Headless\Sts2Headless.csproj `
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
  --eval-combat-deterministic
```

This still uses the official `Sts2Headless` engine. It starts a normal run, then
uses the headless JSON command `enter_room` to jump into one combat encounter.
In `combat` curriculum mode the episode ends as soon as that fight is won, lost,
or times out. `combat_sparse` gives `+1` for a win, `-1` for a loss/timeout, and
`0` for intermediate combat steps. `combat_dense` keeps the same terminal
signals and adds small configurable damage/hp/action shaping.

`--sts2-combat-enemy-pool fixed` with `--sts2-combat-encounter
SHRINKER_BEETLE_WEAK` is now only the C0a smoke test. Use `act1_hallway`,
`act1_hallway_elite`, `act1_elite`, `act1_boss`, or `act1_mixed` for meaningful
combat benchmarks.
The random baseline and PPO eval run outside the RLlib training sampler and log
per-encounter win rates, HP lost, and combat steps on the same pool.

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

Useful checkpoint flags:

```powershell
--checkpoint-freq 1
--training-stage combat_c0_ironclad_starter_act1
--deck-mode starter
--enemy-pool act1
--run-notes "Starter combat-only baseline"
--checkpoint-dir C:\path\to\custom\checkpoint_dir
--resume-from C:\path\to\specific\checkpoint
--no-auto-resume
--init-from-sb3 C:\path\to\old_sb3_model.zip
```

When `--training-stage` is provided and `--checkpoint-dir` is not, RLlib writes
to `models/rllib/<game>/<stage>/`. Each saved RLlib checkpoint also receives a
`checkpoint_metadata.json` file containing the stage, character, deck/enemy
labels, total steps, source checkpoint, heuristic mode, worker/batch settings,
and StS2 process recycling settings. This is the intended path for staged STS2
curriculum runs such as `combat_c0_ironclad_starter_act1` followed by full-run
fine-tuning.

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
  jumping to 16 workers. Use logs and `steps/sec` to tune from there.
