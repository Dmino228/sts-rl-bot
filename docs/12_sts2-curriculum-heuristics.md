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

Two checkpoint entry modes are intentionally separate:

- `--resume-from <checkpoint>` restores the full RLlib Algorithm. Use this only
  to continue the same stage, including optimizer state and counters.
- `--init-from-rllib <checkpoint>` builds a fresh Algorithm for the current
  stage, transfers compatible policy/value weights, and leaves optimizer state
  and training iteration fresh. Use this for curriculum transfer into a new
  checkpoint directory.

Example stage transfer:

```powershell
python rllib\train_rllib.py `
  --preset boss_train_overfit_the_kin_fixed_deck `
  --init-from-rllib models\rllib\sts2\previous_stage\checkpoint_000123 `
  --training-stage overfit_the_kin_fixed_deck_v2 `
  --sts2-cli-path <path>
```

Safety rules:

- explicit `--resume-from` wins over auto-resume in the target stage and is
  logged clearly.
- encoder/model/weight shape mismatches fail fast.
- `checkpoint_metadata.json` records the source checkpoint and transfer mode.
- use a new `--training-stage` for stage transfer so the source checkpoint is
  preserved.

## Minimal Combat Curriculum Hook

The first STS2 curriculum profile is intentionally small and disabled by
default:

Starter-deck combat uses the official `Sts2Headless` JSON protocol:

```text
start_run -> enter_room(type="combat", encounter="<id>")
```

Non-starter combat deck modes insert a player-state setup step before combat:

```text
start_run -> set_player(deck/hp/relics/potions) -> enter_room(type="combat", encounter="<id>")
```

The generated deck must be applied to the actual `Sts2Headless` run state. It
is not valid to only alter Python observations.

Available combat deck modes:

- `starter`: exact current Ironclad starter baseline from `start_run`.
- `random_synthetic`: starter deck plus 3-8 legal Ironclad Act 1 reward cards,
  limited Strike/Defend removals, limited upgrades, and duplicate caps.
- `random_act1_floor_bucket`: synthetic early/mid/late Act 1 buckets with
  floor-conditioned card additions, HP, removals, and upgrades. This is not
  called realistic because it is not learned from real full-run snapshots.
- `random_boss_synthetic_safe`: synthetic boss-feasibility deck with extra
  block/scaling cards, 1-2 simple relics, one potion, and fixed logging.
- `fixed_the_kin_overfit`: exact deterministic THE_KIN_BOSS overfit deck,
  relics, potion, and HP. Use only to prove one controlled boss fight is
  learnable before widening the distribution.

### Quick Start with Presets

The most common experiments can now be launched with a single `--preset` flag:

```powershell
# Smoke test (2 workers, 50k steps, fixed encounter)
python rllib\train_rllib.py --preset combat_smoke_fixed --sts2-cli-path <path>

# Real training (8 workers, 1M steps, act1 mixed pool)
python rllib\train_rllib.py --preset combat_train_act1_mixed --sts2-cli-path <path>

# Real training with synthetic random combat decks
python rllib\train_rllib.py --preset combat_train_act1_mixed_random_synthetic --sts2-cli-path <path>

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

Debug episodes log the selected encounter, enemy pool, deck mode/source/size,
full deck, hand, added/removed/upgraded cards, relics, potions, HP, and pile
sizes to console and `debug_episodes.jsonl`.

Debugging:

```powershell
--sts2-debug-episodes 3
```

logs a reset-time snapshot and detailed per-step combat state for the first N
episodes per worker. The reset snapshot includes the resolved deck mode, deck
source, full deck card IDs/names/upgrades, pile sizes, relics, potions, current
HP, max HP, and selected encounter. These events are written to
`debug_episodes.jsonl` in the run folder.

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

Regression eval presets:

```powershell
python rllib\train_rllib.py `
  --preset eval_c0_the_kin_exact `
  --resume-from <checkpoint> `
  --sts2-cli-path <path>

python rllib\train_rllib.py `
  --preset eval_c1_the_kin_5_decks `
  --resume-from <checkpoint> `
  --sts2-cli-path <path>

python rllib\train_rllib.py `
  --preset eval_c2_the_kin_random_safe `
  --resume-from <checkpoint> `
  --sts2-cli-path <path>
```

These are `eval_only` presets: they run deterministic PPO combat eval and do
not save a new checkpoint.

Mixed curriculum pools can protect old skills during stage transfer:

```powershell
--curriculum-mix c0_the_kin_exact:0.8,c1_the_kin_5_decks:0.2
```

Known named profiles:

- `c0_the_kin_exact`: exact overfit THE_KIN_BOSS deck/relic/potion/HP.
- `c1_the_kin_5_decks`: fixed THE_KIN_BOSS with a small deterministic safe-deck pool.
- `c2_the_kin_random_safe`: fixed THE_KIN_BOSS with random safe boss decks.

Deck randomization and per-character schedules beyond these small profiles are
still later stages.
Do not call a deck mode realistic until it is generated from real full-run
combat snapshots or from a floor-conditioned Act 1 deck model.

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

Grouped metrics include per-category win rate and average HP lost. Category win
rate uses the category denominator, for example `weak_win_rate = weak_wins /
weak_fights`, not `weak_wins / all_fights`. These appear in the compact
console, verbose logs, and `metrics.jsonl`. Per-encounter detail goes to
`metrics.jsonl` and verbose mode.

## Run Folders

Each training run creates a timestamped directory under `runs/`:

```text
runs/20260702_203800_combat_c1_ironclad_starter_act1_mixed/
  config.resolved.yaml   # full config snapshot
  train.log              # complete training log
  metrics.jsonl          # one JSON line per iteration
  debug_episodes.jsonl   # reset deck snapshots and step debug events
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

## Boss Curriculum Diagnostics

The first long boss-only run showed a clear plateau pattern: dense combat reward
improved early, but `boss_win_rate` stayed around 0-3%. This means survival and
partial damage are learnable, but the task is not yet proven feasible for PPO
with the current random deck distribution and compact encoder.

Before another long boss-only run, use short feasibility checks:

```powershell
python rllib\train_rllib.py --preset boss_debug_fixed_the_kin_starter --sts2-cli-path <path>
python rllib\train_rllib.py --preset boss_debug_fixed_the_kin_safe_deck --sts2-cli-path <path>
python rllib\train_rllib.py --preset boss_train_fixed_the_kin_random_synthetic --sts2-cli-path <path>
```

Available fixed boss debug/train presets are generated for:

- `ceremonial_beast`
- `the_kin`
- `vantom`

For wider boss training:

```powershell
python rllib\train_rllib.py --preset boss_train_act1_boss_floor_bucket --sts2-cli-path <path>
```

New boss-specific metrics:

- `avg_boss_hp_remaining_on_loss`
- `avg_boss_hp_fraction_removed`
- `avg_min_boss_hp_reached`
- `avg_damage_dealt_total`
- `avg_turns_survived`
- `win_rate_by_boss`
- `hp_lost_by_boss`
- `boss_hp_remaining_by_boss`

New action-quality metrics:

- `end_turn_with_energy_rate`
- `end_turn_with_playable_attack_rate`
- `end_turn_with_playable_block_when_incoming_damage_rate`
- `block_when_incoming_damage_rate`
- `power_play_rate`
- `cards_played_by_id`

Use baselines on the same encounter and deck distribution:

```powershell
python rllib\train_rllib.py `
  --preset boss_debug_fixed_the_kin_safe_deck `
  --sts2-cli-path <path> `
  --eval-random-baseline 100 `
  --eval-greedy-baseline 100 `
  --dry-run
```

The `random_boss_synthetic_safe` deck mode is a synthetic feasibility tool, not
a realistic Act 1 model. It starts from the Ironclad starter deck, adds more
block/scaling cards, adds 1-2 simple relics and one potion, and logs the exact
deck/relic/potion setup.

The `combat_boss_potential` reward mode is intended for boss feasibility tests:

- `+10` terminal win
- `-1` terminal loss
- `-3` timeout
- per-step normalized boss HP reduction
- smaller normalized player HP loss penalty
- one-shot boss HP milestones at 75/50/25%

Encoder experiments can be run with:

```powershell
--sts2-encoder-mode compact
--sts2-encoder-mode flat
```

The flat encoder includes card/relic/potion/monster identity one-hots and changes
the observation shape, so keep its checkpoints in a separate training stage.

### THE_KIN_BOSS Overfit Test

When boss-only random synthetic training plateaus, the next diagnostic is not
another wide boss run. First overfit one exact fight:

- encounter: `THE_KIN_BOSS`
- deck mode: `fixed_the_kin_overfit`
- relics: Burning Blood, Anchor, Vajra, Oddly Smooth Stone, Bag of Preparation
- potion: Strength Potion
- HP: 80/80
- seed: `20260705`
- encoder: compact
- reward: `combat_boss_potential`

Debug:

```powershell
python rllib\train_rllib.py `
  --preset boss_debug_overfit_the_kin_fixed_deck `
  --sts2-cli-path <path>
```

Train:

```powershell
python rllib\train_rllib.py `
  --preset boss_train_overfit_the_kin_fixed_deck `
  --sts2-cli-path <path>
```

Success criterion: deterministic PPO eval should show a rising win rate on the
fixed encounter. If this does not learn, focus on action encoding/reward/model
capacity before adding boss/deck/seed diversity.
