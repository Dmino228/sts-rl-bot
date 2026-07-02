# STS2 Curriculum and Strategic Heuristics

## Motivation

Long StS2 full-run training can plateau because PPO is asked to learn too much
at once:

- combat tactics,
- card/relic/potion rewards,
- map routing,
- events,
- shops,
- rest-site choices,
- long-term credit assignment across the whole act.

The current priority is to make staged experiments reproducible first, then use
those stages to test whether combat curriculum and non-combat heuristics improve
Act 1 progression.

## Checkpoint Metadata

RLlib checkpoints can now be scoped by a curriculum stage:

```text
--training-stage combat_c0_ironclad_starter_act1
```

When `--checkpoint-dir` is not set, the stage becomes part of the default path:

```text
models/rllib/sts2/combat_c0_ironclad_starter_act1/
```

Every saved RLlib checkpoint gets a `checkpoint_metadata.json` file next to the
RLlib checkpoint data. The metadata records:

- `training_stage`
- `character`
- `multi_character`
- `deck_mode`
- `enemy_pool`
- `total_steps`
- `source_checkpoint`
- `heuristic_mode`
- core RLlib batch/worker settings
- StS2 process recycle settings

Useful flags:

```powershell
--training-stage combat_c0_ironclad_starter_act1 `
--deck-mode starter `
--enemy-pool act1 `
--run-notes "Starter combat-only baseline"
```

## Minimal Combat Curriculum Hook

The first STS2 curriculum profile is intentionally small and disabled by
default:

This uses the official `Sts2Headless` JSON protocol:

```text
start_run -> enter_room(type="combat", encounter="<id>")
```

### Quick Start with Presets

The most common experiments can now be launched with a single `--preset` flag:

```powershell
# Smoke test (2 workers, 50k steps, fixed encounter)
python rllib\train_rllib.py --preset combat_smoke_fixed --sts2-cli-path <path>

# Real training (8 workers, 1M steps, act1 mixed pool)
python rllib\train_rllib.py --preset combat_train_act1_mixed --sts2-cli-path <path>

# Debug mode (1 worker, verbose logging, debug episodes)
python rllib\train_rllib.py --preset combat_debug_fixed --sts2-cli-path <path>

# Full run with heuristic
python rllib\train_rllib.py --preset fullrun_ironclad_heuristic_hard --sts2-cli-path <path>
```

CLI flags override preset values:

```powershell
python rllib\train_rllib.py --preset combat_train_act1_mixed --sts2-cli-path <path> --workers 4
```

List available presets:

```powershell
python rllib\train_rllib.py --list-presets
```

Inspect resolved config without starting training:

```powershell
python rllib\train_rllib.py --preset combat_smoke_fixed --sts2-cli-path <path> --dry-run
```

### YAML Config Files

For reusable custom configurations:

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

Resolution order: preset defaults < YAML file < CLI flags.

### Manual Example (verbose flags)

The original full-flag syntax still works:

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
  --sts2-combat-enemy-pool act1_hallway
```

In combat curriculum mode one episode is exactly one fight. The episode
terminates immediately on combat win, loss, or timeout, and does not continue to
map/card rewards/full-run macro decisions.

Reward modes:

- `full_v3_2`: legacy full-run reward, preserved for compatibility.
- `combat_sparse`: `+1` win, `-1` loss/timeout, `0` otherwise.
- `combat_dense`: terminal sparse reward plus small configurable damage dealt,
  HP lost, and action-penalty shaping. It never includes floor/relic/card/act
  completion rewards.

Debugging:

```powershell
--sts2-debug-episodes 3
```

logs detailed per-step combat state for the first N episodes per worker.

Encounter pools:

- `fixed`: use `--sts2-combat-encounter`; C0a smoke tests only.
- `act1_hallway`: sampled Act 1 hallway combats.
- `act1_elite`: sampled Act 1 elites.
- `act1_boss`: sampled Act 1 bosses observed from `get_map`.
- `act1_hallway_elite`: hallway-heavy pool with elites, for C0c.
- `act1_mixed`: hallway-heavy mix with elites and bosses.

Benchmark helpers:

```powershell
--eval-random-baseline 500 `
--eval-combat-episodes 500 `
--eval-combat-freq 10 `
--eval-combat-deterministic
```

The random baseline uses valid random actions on the same pool. PPO combat eval
runs outside the training sampler and logs per-encounter win rate, HP lost, and
combat steps. The random baseline runs once at startup by default; set
`--eval-random-baseline-freq N` only when you explicitly want to rerun it during
training. In STS2 combat curriculum, PPO eval and checkpointing default to every
10 train iterations.

Crash reproduction:

```powershell
--sts2-combat-enemy-pool fixed `
--sts2-combat-encounter ENTOMANCER_ELITE `
--seed 12345 `
--sts2-debug-episodes 1
```

Crash diagnostics include the selected encounter, turn/combat step, hand card
IDs/names/costs/playability, selected card/target, a bounded recent trace, and
the latest stderr tail.

Deck randomization and per-character C1/C2 schedules are still later stages.

## Console Output Modes

The `--console` flag controls what appears on stdout during training:

- **compact** (default for presets): Uses `rich` for a live-updating progress
  bar and metrics table that replaces itself in-place. No scroll spam.
- **verbose**: Full log lines with all metrics printed per iteration.
- **quiet**: Only warnings and errors to console; everything goes to log files.

In combat curriculum mode, compact shows: iter, steps, sps, reward,
train win/loss/timeout, avg hp lost, avg combat steps, and grouped
weak/normal/elite/boss win rates.

In full-run mode, compact shows: iter, steps, sps, reward, floor_mean,
max_floor, boss_reached%, boss_killed%, act2%.

Full details always go to `train.log` and `metrics.jsonl` in the run folder
regardless of console mode.

## Grouped Combat Metrics

Combat training metrics are aggregated by encounter category:

- **weak**: encounters ending with `_WEAK`
- **normal**: encounters ending with `_NORMAL`
- **elite**: encounters ending with `_ELITE`
- **boss**: encounters ending with `_BOSS`

Grouped metrics include per-category win rate and average HP lost. These appear
in the compact console, verbose logs, and `metrics.jsonl`. Per-encounter detail
goes to `metrics.jsonl` and verbose mode.

## Run Folders

Each training run creates a timestamped directory under `runs/`:

```text
runs/20260702_203800_combat_c1_ironclad_starter_act1_mixed/
  config.resolved.yaml   # full config snapshot
  train.log              # complete training log
  metrics.jsonl          # one JSON line per iteration
  crashes/               # crash debug bundles (on-demand)
```

## Crash Debug Bundles

The environment wrapper maintains a bounded ring buffer (last 50 actions).
On crash, a bundle is saved to the run folder containing:

- `last_actions.jsonl`: ring buffer with action, reward, encounter, step
- `last_state.json`: current observation snapshot
- `stderr_tail.txt`: last lines of STS2 process stderr
- `config.resolved.yaml`: exact config used

## Integration Options

### Option A: Hard-Control Non-Combat

Combat remains controlled by PPO. Non-combat STS2 decisions are selected by a
deterministic heuristic.

Current implementation:

```text
--heuristic-mode hard
```

The implementation narrows the non-combat action mask to a single heuristic
action. That means RLlib records the same action that is actually executed,
instead of silently replacing a policy action after sampling.

### Option B: Top-K Heuristic Mask

PPO still chooses, but only among the top heuristic-ranked non-combat actions.

Current hook:

```text
--heuristic-mode mask --heuristic-top-k 2
```

This mode is intentionally built on the same ranker as Option A.

### Option C: Behavior Cloning

The heuristic ranker can later emit labels:

```text
observation + legal_action_mask -> heuristic_action
```

Those labels can seed supervised pretraining before PPO fine-tuning. The RLlib
wrapper already exposes the selected heuristic action in `info["heuristic_action"]`
when a heuristic mode is active.

## Current STS2 Heuristic Scope

The first heuristic is deliberately conservative and non-combat only:

- `combat_play`: PPO controls the action.
- combat `card_select` decisions: PPO/control flow remains untouched.
- `map_select`: prefer safer progress, avoid low-HP elites, value rest/treasure.
- `card_reward`: pick obvious value cards, skip curses/status cards when allowed.
- `event_choice`: choose the first unlocked option by default.
- `rest_site`: rest at low HP, otherwise prefer smith/upgrade.
- `shop`: prefer relics, useful cards, card removal, then potions, otherwise leave.
- `card_select`: choose best or worst card based on prompt hints.
- `bundle_select`: choose first legal bundle.

These rules are not intended to be final STS2 strategy. They are a diagnostic
baseline that should make it easy to compare:

```text
no heuristic
vs hard non-combat heuristic
vs top-k non-combat mask
```

## Suggested First Experiments

Start with single-character Ironclad before multi-character:

```powershell
# Using presets (recommended):
python rllib\train_rllib.py --preset combat_train_act1_mixed --sts2-cli-path <path>

# Using manual flags:
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
  --heuristic-mode hard `
  --timesteps 10000000
```

Compare against the same run with `--heuristic-mode none`.

Primary metrics:

- `floor_mean`
- `max_floor`
- `boss_reached%`
- `boss_killed%`
- `act2%`
- `steps/sec`
- watchdog and process-recycle frequency
- `weak_win_rate`, `normal_win_rate`, `elite_win_rate`, `boss_win_rate`

## Later Curriculum Direction

The likely sequence is:

```text
memory/process stability
-> stage-scoped checkpoint metadata
-> combat-only curriculum checkpoint
-> full-run fine-tuning from combat checkpoint
-> hard non-combat heuristic
-> top-k non-combat mask
-> behavior cloning from heuristic labels
-> multi-character expansion
```

