# 07_REWARD_SHAPING_V3_2: SCALE-PRESERVING ROBUST REWARD SYSTEM

## Role and Verdict

V3.1 is a good step over V3.0. Keep the character-agnostic HP bootstrap,
robust HP extraction, one-shot act completion guard, and the decision to defer
curriculum learning and overblock penalties.

V3.2 fixes the remaining reward-shaping risks:

1. `tanh(delta / 50)` almost doubles normal combat reward scale compared with
   V3.0. A Strike goes from `+0.06` to about `+0.119`.
2. Positive-only monster HP deltas can be farmed when enemies heal, revive, or
   summon, because HP increases are ignored instead of becoming negative
   progress.
3. Death can be missed when `in_game == False` but the state no longer exposes a
   `GAME_OVER` or `DEATH` screen type.
4. Relic, upgrade, and removal rewards should use real deltas instead of a
   single fixed reward when several things change at once.
5. The anti-stall counter is currently action-step based, not turn based. A
   threshold of 40 action steps can hit normal boss fights and skill-heavy play.

---

## Design Principles

1. Preserve V3.0's normal combat scale while keeping hard bounds.
2. Treat monster HP and player HP as state potentials: reward improvements,
   penalize regressions.
3. Keep macro rewards sparse and below terminal/progress objectives.
4. Keep card-choice shaping conservative. Do not reward generic card pickup yet.
5. Make every event reward proportional to the number of actual items changed.

---

## Constants

```python
HP_DELTA_SCALE = 100.0
HP_DELTA_CAP = 100

FLOOR_REWARD = 3.0
COMBAT_VICTORY_REWARD = 2.0
RELIC_REWARD = 1.0
CARD_UPGRADE_REWARD = 0.5
CARD_REMOVE_REWARD = 0.5

DEATH_PENALTY = -25.0
ACT_COMPLETION_REWARD = 15.0

COMBAT_STEP_GRACE = 80
ANTI_STALL_PENALTY = -0.02
```

Helper:

```python
def _bounded_hp_reward(delta: int) -> float:
    clipped = max(-HP_DELTA_CAP, min(HP_DELTA_CAP, delta))
    return clipped / HP_DELTA_SCALE
```

Rationale:

- For normal deltas, this preserves V3.0 exactly: 6 damage is `+0.06`, 10 HP
  lost is `-0.10`.
- For extreme edge cases, the reward is bounded to `[-1.0, +1.0]`.
- Unlike `tanh`, clipped linear scoring stays additive for normal play and does
  not quietly reward many small hits more than one equivalent large hit.

---

## State Tracking

Initialize in `__init__()` and `reset()`:

```python
self.last_player_hp: Optional[int] = None
self.last_max_hp: Optional[int] = None
self.last_monster_total_hp: Optional[int] = None
self.last_floor: int = 0
self.last_screen_type: str = "NONE"
self.last_relic_ids: Optional[set[str]] = None
self.last_deck_size: Optional[int] = None
self.last_upgraded_cards: Optional[int] = None
self.last_act: int = 1
self.last_in_combat: bool = False
self.step_count: int = 0
self.combat_step_count: int = 0
```

Notes:

- Use `None` for first-step bootstrapping. This avoids starter relic/deck/HP
  assumptions leaking into the first reward calculation.
- Track relic IDs, not only relic count. This catches boss relic swaps and other
  replace-one-relic-with-one-relic events.

---

## Reward Formula V3.2

Before scoring macro rewards, determine whether the current state still belongs
to the active run:

```python
state_in_game = self.current_state.get("in_game", True)
terminal_screen = screen_type in ["GAME_OVER", "DEATH"]
score_macro = state_in_game and not terminal_screen
```

Macro deltas such as deck size, relic IDs, and floor should not be scored on
post-run/menu states. Otherwise a post-death empty state can look like massive
card removal.

### A. Monster HP Progress

Only score monster HP deltas after combat has already been bootstrapped. Also
score the transition into `COMBAT_REWARD`, because the state after a killing
blow may no longer be an active combat state:

```python
if (
    self.last_in_combat
    and self.last_monster_total_hp is not None
    and (in_combat or screen_type == "COMBAT_REWARD")
):
    monster_delta = self.last_monster_total_hp - current_monster_hp
    reward += _bounded_hp_reward(monster_delta)
```

Why this changes V3.1:

- V3.1 uses `max(0, last_monster_hp - current_monster_hp)`.
- If an enemy heals from 20 to 35 HP, V3.1 gives `0.0`; V3.2 gives `-0.15`.
- This removes damage farming against healers, revivers, and summoners.

### B. Player HP Progress

Apply player HP delta on all screens, not only as combat damage or
out-of-combat healing:

```python
if self.last_player_hp is not None:
    hp_delta = current_hp - self.last_player_hp
    reward += _bounded_hp_reward(hp_delta)
```

Why this changes V3.1:

- Damage is negative.
- Healing is positive.
- Combat healing is no longer silently undervalued.
- Event HP costs are penalized immediately, then balanced by whatever reward the
  event actually gives later.

This replaces both A2 damage taken and B6 healing. Do not also add the old B6
healing reward, or healing will be double-counted.

### C. Anti-Stall

Prefer turn-based tracking if CommunicationMod exposes a combat turn field. If
not, keep the existing action-step fallback but raise the grace window:

```python
if in_combat:
    self.combat_step_count += 1
    if self.combat_step_count > COMBAT_STEP_GRACE:
        reward += ANTI_STALL_PENALTY
```

Rationale:

- The current `40` threshold is not really "about 10 turns"; it is about 40
  commands/actions.
- Boss fights and defensive decks can legitimately exceed 40 action steps.
- `80` keeps pressure against loops without punishing normal Act 1 bosses as
  aggressively.

### D. Floor Progress

```python
if current_floor > self.last_floor:
    reward += FLOOR_REWARD * (current_floor - self.last_floor)
```

Keep this as the primary macro signal.

### E. Combat Victory

```python
if screen_type == "COMBAT_REWARD" and self.last_screen_type != "COMBAT_REWARD":
    reward += COMBAT_VICTORY_REWARD
    self.combat_step_count = 0
```

Keep this unchanged.

### F. Relics

```python
current_relic_ids = {r.get("id", r.get("name", "")) for r in game_state.get("relics", [])}
current_relic_ids.discard("")

if self.last_relic_ids is not None:
    new_relics = current_relic_ids - self.last_relic_ids
    reward += RELIC_REWARD * len(new_relics)
```

Why this changes V3.1:

- Count-based tracking misses relic swaps.
- ID-based tracking rewards actual newly acquired relics.

### G. Card Upgrades

```python
if self.last_upgraded_cards is not None:
    upgrade_delta = max(0, current_upgraded_cards - self.last_upgraded_cards)
    reward += CARD_UPGRADE_REWARD * upgrade_delta
```

### H. Card Removal

```python
if self.last_deck_size is not None and not in_combat:
    removed = max(0, self.last_deck_size - current_deck_size)
    reward += CARD_REMOVE_REWARD * removed
```

Keep generic card pickup unrewarded for now. Card additions are too contextual:
adding a high-impact attack early can be excellent, while adding a weak card
late can be bad. Let downstream survival/progress rewards decide this until the
state encoder and policy are strong enough for card-specific evaluation.

### I. Death

```python
act_completed = current_act > self.last_act
not_in_game = not self.current_state.get("in_game", True)
dead_screen = screen_type in ["GAME_OVER", "DEATH"]

terminal_failure = current_hp <= 0 or dead_screen or (not_in_game and not act_completed)

if terminal_failure and not self.terminal_reward_given:
    reward += DEATH_PENALTY
    self.terminal_reward_given = True
```

Why this changes V3.1:

- `step()` already terminates on `in_game == False`.
- `_calculate_reward()` should also know that this is terminal failure when the
  act did not advance.
- Otherwise some deaths can return `0.0` terminal reward if the final JSON no
  longer has a death screen.
- The terminal reward must be one-shot. Repeated post-death cleanup/menu states
  should not apply death again.

### J. Act Completion

```python
if current_act > self.last_act and screen_type not in ["GAME_OVER", "DEATH"]:
    reward += ACT_COMPLETION_REWARD
```

This is still one-shot, but stronger than V3.1. For Act-1-only training, beating
the Act 1 boss is the current task victory condition, so it should be visibly
stronger than any single macro event.

Reset handling:

- `step()` should terminate on an act transition, not on every state where
  `act > 1`.
- `reset()` should not try to navigate from the Act 2 map back to the main menu.
  CommunicationMod has no direct `ABANDON` command, and `START` is only valid
  from the main menu.
- When `reset()` is called while already inside a non-terminal run after an act
  transition, perform a soft reset: keep the current game state, bootstrap
  reward tracking from that state, and return it as the first observation of the
  next episode.

---

## Updated Reward Budget

| Component | Range | Notes |
|---|---:|---|
| Monster HP delta | `[-1.0, +1.0]` | Usually `[-0.3, +0.5]`; preserves `/100` scale |
| Player HP delta | `[-1.0, +1.0]` | Damage and healing share one channel |
| Anti-stall | `[-0.02, 0.0]` | Only after 80 combat action steps unless turn tracking exists |
| Floor progress | `+3.0` per floor | Primary macro signal |
| Combat victory | `+2.0` | Fight milestone |
| New relic | `+1.0` each | ID-based, catches swaps |
| Card upgrade | `+0.5` each | Delta-based |
| Card removal | `+0.5` each | Outside combat only |
| Death | `-25.0` | Terminal failure |
| Act completion | `+15.0` | Act-1 task success |

Important correction to V3/V3.1 wording:

Death does not need to make every late failed run net-negative. A floor-12 death
can reasonably have a higher return than a floor-3 death because progress is
useful training signal. The real invariant should be:

> At the same strategic position, survival must dominate death; completing the
> act must dominate dying after accumulating dense progress rewards.

Do not monitor `ep_rew_mean` as "negative means bad, positive means good".
Monitor it together with max floor, act completion rate, average HP at floor
entry, and death floor distribution.

---

## Implementation Checklist

1. Replace `COMBAT_SCALE` and `tanh` combat rewards with `_bounded_hp_reward`.
2. Add `last_in_combat`, optional `last_monster_total_hp`, and `last_relic_ids`.
3. Score monster HP deltas with signed progress, not `max(0, delta)`.
4. Replace separate damage-taken and healing components with one player HP delta
   channel.
5. Add `in_game == False` to death detection, guarded by `act_completed`.
6. Multiply floor, upgrade, removal, and relic rewards by real deltas.
7. Raise action-step anti-stall grace to 80, or switch to turn-based tracking if
   the JSON exposes a reliable turn counter.
8. Add a one-shot terminal guard such as `terminal_reward_given`.
9. Score macro deltas only while `in_game == True` and not on terminal screens.
10. Update reward logs from `V3.1` to `V3.2` and include component breakdowns for
   the first validation run.
