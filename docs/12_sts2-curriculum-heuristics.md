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

Example:

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
  --sts2-combat-encounter SHRINKER_BEETLE_WEAK
```

It does not yet implement the full Act 1 randomized enemy pool, deck
randomization, or per-character C1/C2 schedules. It is the first integration
point for validating combat-only PPO on the real headless engine before adding
more curriculum breadth.

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
