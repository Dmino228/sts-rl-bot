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

The current priority is to test whether removing random non-combat decisions
lets the policy improve combat and Act 1 progression faster.

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
full-run baseline
-> hard non-combat heuristic
-> top-k non-combat mask
-> combat-only curriculum checkpoint
-> full-run fine-tuning from combat checkpoint
-> behavior cloning from heuristic labels
-> multi-character expansion
```
