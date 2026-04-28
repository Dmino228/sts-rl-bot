# 06_REWARD_SHAPING_V3_1: SYMMETRICAL & ROBUST REWARD SYSTEM

## ROLE & CONTEXT
Following architectural review by GPT and Claude, we are implementing version 3.1 of the reward system.
We are discarding Curriculum Learning (complex, hinders diagnosis) and Overblock penalties (premature optimization).
V3.1 focuses on fixing 4 bugs/technical debt items identified in v3 while adding tanh-bounded combat signals.

---

## CHANGES FROM V3.0

### 1. tanh-Bounded Combat Signals (Points 1)
**Problem:** Linear `/100.0` scaling is unbounded for edge-case damage spikes.
**Fix:** Replace `dmg / 100.0` with `math.tanh(dmg / COMBAT_SCALE)` where `COMBAT_SCALE = 50`.
- Applied symmetrically to both `dmg_dealt` (A1) and `hp_lost` (A2)
- Output range: `[0.0, 1.0)` — mathematically bounded, no outlier gradients
- `tanh(6/50) = 0.119` for a Strike, `tanh(50/50) = 0.762` for 50 damage
- Preserves defense parity: blocking 10 damage saves the same reward as dealing 10 damage

### 2. One-Shot Act Completion — C2 Fix (Point 3)
**Problem:** `if act > 1` is True on EVERY step once the agent enters Act 2. With the
`terminated = True` check in `step()`, the episode ends immediately — but if there's even
one extra step, the agent gets `+10.0` repeatedly, completely corrupting the value function.
**Fix:** Added `self.last_act` tracking. C2 reward fires exactly once per act transition
when `current_act > self.last_act`.

### 3. Character-Agnostic HP Initialization (Point 4)
**Problem:** `last_player_hp = 80` and `last_max_hp = 80` are hardcoded for Ironclad.
Silent has 70 HP, Defect has 75, Watcher has 72. This would cause incorrect delta
calculations on the first step for any non-Ironclad character.
**Fix:** Initialize both to `None`. First-step bootstrap reads actual HP from game state.
All delta calculations guard against `None` with `if self.last_player_hp is not None`.

### 4. Robust HP Fallback Extraction (Point 5)
**Problem:** Old code: `if current_hp == 0: current_hp = player.get(...)`. This only
triggers when HP is exactly 0, missing cases where `game_state.current_hp` is simply
absent from the JSON (CommunicationMod drops it outside combat).
**Fix:** Priority-based extraction:
1. In combat → `combat_state.player` is authoritative
2. Outside combat → `game_state` top-level fields (if present)
3. Always → fall back to `self.last_player_hp` (last known good value)

---

## DEFERRED

### Curriculum Learning / Annealing (Point 2)
Good idea but introduces complexity (requires knowing total training steps, phase boundaries).
Makes debugging harder. Implement after v3.1 is validated.

### Overblock Penalty (Point 6)
Agent may learn this naturally. Premature to add a hand-crafted penalty before observing
whether the behavior persists past initial training noise.

---

## CONSTANTS

```python
COMBAT_SCALE = 50  # tanh normalization divisor for combat HP deltas
```

## STATE TRACKING (V3.1)

```python
self.last_player_hp: Optional[int] = None     # No hardcoded HP
self.last_max_hp: Optional[int] = None         # No hardcoded HP
self.last_monster_total_hp: int = 0
self.last_floor: int = 0
self.last_screen_type: str = "NONE"
self.last_relic_count: int = 1
self.last_deck_size: int = 10
self.last_upgraded_cards: int = 0
self.step_count: int = 0
self.combat_step_count: int = 0
self.last_act: int = 1      # NEW — one-shot C2 guard per act transition
```

## REWARD FORMULA (unchanged components not listed)

### A1. Damage Dealt — tanh-bounded
```python
dmg_dealt = max(0, self.last_monster_total_hp - current_monster_hp)
reward += math.tanh(dmg_dealt / COMBAT_SCALE)
```

### A2. Damage Taken — tanh-bounded (symmetric)
```python
if self.last_player_hp is not None:
    hp_lost = max(0, self.last_player_hp - current_hp)
    reward -= math.tanh(hp_lost / COMBAT_SCALE)
```

### C2. Act Completion — one-shot
```python
if current_act > self.last_act and screen_type not in ["GAME_OVER", "DEATH"]:
    reward += 10.0
```

## REWARD BUDGET (V3.1 adjusted)

| Component | Per-Step Range | Notes |
|---|---|---|
| Damage Dealt (tanh) | `[0.0, ~0.99]` | tanh(999/50)=0.999 for big hits |
| Damage Taken (tanh) | `[-0.99, 0.0]` | Symmetric with dealt |
| Anti-Stall | `[-0.02, 0.0]` | Only after 40 steps |
| Floor Progress | `+3.0` | Unchanged |
| Combat Victory | `+2.0` | Unchanged |
| Death | `-20.0` | Unchanged |
| Act Completion | `+10.0` | Now fires exactly once per act transition |