# 05_REWARD_SHAPING_V3: MATHEMATICALLY VERIFIED DENSE REWARD SYSTEM

## ROLE & CONTEXT
V2 was a significant improvement over V1, but training logs reveal that the agent struggles to learn coherent macro-strategy. This document audits V2 for weight imbalances, proposes corrections, and specifies V3 — a reward system with mathematically bounded components and verified gradient properties.

---

## PART 1: AUDIT OF V2 — CRITICAL ISSUES

### Issue #1: Combat Time Penalty vs. Damage Reward Imbalance
- **V2:** Time penalty = `-0.01` per combat step; Damage dealt reward = `dmg / 100.0`
- **Problem:** A typical Strike deals 6 damage → reward = `+0.06`. The agent nets `+0.05` per strike step. But a Defend that blocks 5 damage yields `0.0` damage reward minus `-0.01` time penalty = `-0.01`. **The agent is penalized for defensive play.** This teaches the model to never block — catastrophic against hard-hitting enemies like Gremlin Nob or Lagavulin.

### Issue #2: Damage Taken Penalty Scales Inversely with Max HP (Unstable)
- **V2:** `-(max(0, dmg_taken) / max(1, current_max_hp)) * 2.0`
- **Problem:** At max_hp=80, losing 10 HP = `-0.25`. At max_hp=40 (after boss relic), losing 10 HP = `-0.50`. The agent learns that taking a boss relic that lowers max HP makes ALL future damage "feel worse," discouraging relic-heavy strategies. The scaling is also non-linear in a way that makes the gradient landscape unstable — the same combat action produces different penalty magnitudes depending on an unrelated macro-economic decision.

### Issue #3: Floor Progress Reward is Too Low
- **V2:** Floor progress = `+1.0`
- **Problem:** The agent can earn `+1.0` from a single relic pickup or card upgrade. Clearing an entire floor of combat (which is far harder) yields the same `+1.0`. **Floor progress is the single most important signal** (reaching floor 51 = victory), but it's weighted equally with card upgrades.

### Issue #4: Relic Reward Encourages Relic Gambling, Not Strategic Play
- **V2:** Relic gained = `+2.0`
- **Problem:** This is the HIGHEST single-event reward in V2. The agent may learn to take unnecessary Elite fights for the relic reward, dying to Elites it can't beat. Relics should be a positive signal but not the dominant one.

### Issue #5: Card Removal Reward Has No Context
- **V2:** Deck size decreased = `+1.0`
- **Problem:** Card removal is good, but `+1.0` is given whether you removed a Curse (excellent) or a Strike (situational). The reward also triggers when cards are *exhausted* during combat (temporary deck size changes), creating noise.

### Issue #6: Combat Victory Reward is Too Small
- **V2:** COMBAT_REWARD screen = `+1.0`
- **Problem:** Winning a fight is the fundamental loop of StS. A `+1.0` reward for winning is equal to upgrading a card. The agent has no strong incentive to win fights efficiently.

### Issue #7: Death Penalty May Be Too Weak Relative to Cumulative Rewards
- **V2:** Death = `-10.0`
- **Problem:** In a run reaching floor 15, the agent accumulates ~15 floor rewards + several relic/upgrade rewards ≈ `+25.0` total. A `-10.0` death penalty only claws back 40% of accumulated reward. **The agent can learn that dying on floor 15 is "profitable" compared to dying on floor 5.** Death must be the strongest signal.

---

## PART 2: V3 DESIGN PRINCIPLES

1. **Bounded Components:** Every reward component has a known min/max per step. No component can dominate training through magnitude alone.
2. **Consistent Scaling:** All combat rewards use the same normalization divisor (`100.0`) to keep gradients uniform.
3. **Defense Parity:** Blocking damage is rewarded as strongly as dealing damage — both are valid combat strategies.
4. **Terminal Dominance:** Death penalty and victory reward are the strongest signals, ensuring the agent prioritizes survival and progress above all else.
5. **No Implicit Punishment of Macro Decisions:** Combat penalties do NOT scale with max_hp or other out-of-context variables.

---

## PART 3: STATE TRACKING

In `SlayTheSpireEnv.__init__()` and `reset()`, initialize:

```python
# Reward tracking — V3
self.last_player_hp: int = 80
self.last_max_hp: int = 80
self.last_monster_total_hp: int = 0
self.last_floor: int = 0
self.last_screen_type: str = "NONE"
self.last_relic_count: int = 1
self.last_deck_size: int = 10
self.last_upgraded_cards: int = 0
self.last_potion_count: int = 0
self.step_count: int = 0
self.combat_step_count: int = 0
```

---

## PART 4: THE V3 REWARD FORMULA

Initialize `reward = 0.0` at the start of `_calculate_reward()`.

### A. Combat Rewards (Only when actively in combat: `screen_type == "NONE"` and `"end" in available_commands`)

#### A1. Damage Dealt — Normalized
```python
dmg_dealt = max(0, self.last_monster_total_hp - current_monster_total_hp)
reward += dmg_dealt / 100.0
```
- **Range per step:** `[0.0, ~0.5]` (max single-turn damage ≈ 50)
- **Rationale:** Linear, bounded, no magic multipliers.

#### A2. Damage Prevented (Block + Avoidance)
```python
hp_lost = max(0, self.last_player_hp - current_player_hp)
reward -= hp_lost / 100.0
```
- **Range per step:** `[-0.8, 0.0]`
- **Rationale:** Same scale as damage dealt (`/100.0`). The agent gets equal "utility" from dealing 10 damage and preventing 10 damage. No asymmetric penalty that biases toward aggression or defense.

#### A3. Anti-Stall Time Pressure
```python
self.combat_step_count += 1
if self.combat_step_count > 40:
    reward -= 0.02  # escalating pressure after turn 40
```
- **Rationale:** V2's flat `-0.01` per step punished normal play. V3 only penalizes after 40 steps (≈10 combat turns), which indicates genuine stalling. The penalty is mild enough to not dominate but creates gradient pressure toward finishing fights.
- Reset `self.combat_step_count = 0` when transitioning OUT of combat.

### B. Macro-Economy Rewards (Calculated on ALL screens)

#### B1. Floor Progress — Primary Signal
```python
if current_floor > self.last_floor:
    reward += 3.0
```
- **Rationale:** This is the MOST important macro signal. Reaching higher floors is directly correlated with winning. At `3.0`, it dominates single-step combat rewards but is achievable only once per floor transition.

#### B2. Combat Victory
```python
if screen_type == "COMBAT_REWARD" and self.last_screen_type != "COMBAT_REWARD":
    reward += 2.0
    self.combat_step_count = 0  # reset stall counter
```
- **Rationale:** Winning a fight is a milestone. `2.0` makes it the second-strongest per-event signal. The `last_screen_type` guard prevents the V1 bug of looping on the reward screen.

#### B3. Relic Acquired
```python
if current_relic_count > self.last_relic_count:
    reward += 1.0
```
- **Rationale:** Relics are valuable but the reward is now BELOW floor progress and combat victory. The agent won't seek relics at the cost of dying.

#### B4. Card Upgrade
```python
if current_upgraded_cards > self.last_upgraded_cards:
    reward += 0.5
```
- **Rationale:** Upgrading is good but it's a passive action (rest site). Lower weight ensures the agent doesn't over-prioritize upgrading over progressing floors.

#### B5. Card Removal
```python
if current_deck_size < self.last_deck_size and screen_type != "NONE":
    reward += 0.5
```
- **Rationale:** The `screen_type != "NONE"` guard prevents exhaust-during-combat from triggering this. Only shop/event removals count.

#### B6. Healing Recognition
```python
hp_gained = max(0, current_player_hp - self.last_player_hp)
if hp_gained > 0 and screen_type != "NONE":
    reward += hp_gained / 100.0
```
- **Rationale:** Healing outside combat (rest, potions at event) is mildly rewarded. Same `/100.0` scale. The `screen_type != "NONE"` guard prevents combat healing (Reaper, Feed) from double-counting with the damage system.

### C. Terminal States

#### C1. Death — Dominant Penalty
```python
if current_player_hp <= 0 or screen_type in ["GAME_OVER", "DEATH"]:
    reward -= 20.0
```
- **Rationale:** At `-20.0`, death wipes out approximately a full run's worth of accumulated rewards. This ensures death is ALWAYS net-negative in the value function.

#### C2. Act Completion Bonus (Future-Proofing)
```python
current_act = game_state.get("act", 1)
if current_act > 1 and screen_type not in ["GAME_OVER", "DEATH"]:
    reward += 10.0  # Completing Act 1
```
- **Rationale:** Since training currently terminates at Act 2 boundary, this gives a massive positive signal for actually beating the Act 1 boss. This is the "victory" signal for now.

---

## PART 5: REWARD BUDGET ANALYSIS

| Component | Per-Step Range | Per-Event Range | Expected per Episode |
|---|---|---|---|
| Damage Dealt | `[0.0, +0.5]` | — | `+5.0` to `+15.0` |
| Damage Taken | `[-0.8, 0.0]` | — | `-3.0` to `-10.0` |
| Time Penalty | `[-0.02, 0.0]` | — | `0.0` to `-1.0` |
| Floor Progress | — | `+3.0` | `+15.0` to `+45.0` |
| Combat Victory | — | `+2.0` | `+6.0` to `+14.0` |
| Relic | — | `+1.0` | `+1.0` to `+5.0` |
| Card Upgrade | — | `+0.5` | `+0.5` to `+3.0` |
| Card Removal | — | `+0.5` | `+0.0` to `+2.0` |
| Healing | `[0.0, +0.3]` | — | `+0.0` to `+3.0` |
| Death | — | `-20.0` | `-20.0` (if died) |
| Act Completion | — | `+10.0` | `+10.0` (if won) |

**Successful run estimate:** `+40` to `+80` net reward.
**Failed run (floor ~8):** `-5` to `+5` net reward (death penalty wipes gains).

This gradient ensures the agent ALWAYS prefers longer runs over shorter ones, and winning over dying.

---

## PART 6: HELPER FUNCTIONS

```python
def _get_total_monster_hp(self, game_state: dict) -> int:
    """Sum current_hp of all alive monsters."""
    combat_state = game_state.get("combat_state", {})
    if not combat_state:
        return 0
    monsters = combat_state.get("monsters", [])
    return sum(
        m.get("current_hp", 0)
        for m in monsters
        if not m.get("is_gone", False)
    )

def _count_upgraded_cards(self, game_state: dict) -> int:
    """Count total upgrade level across all cards in deck."""
    deck = game_state.get("deck", [])
    return sum(card.get("upgrades", 0) for card in deck)

def _is_in_combat(self, state: dict) -> bool:
    """Check if we're actively in combat (playing cards)."""
    screen_type = state.get("game_state", {}).get("screen_type", "NONE")
    available_cmds = state.get("available_commands", [])
    return screen_type == "NONE" and "end" in available_cmds
```

---

## PART 7: FULL `_calculate_reward` IMPLEMENTATION

```python
def _calculate_reward(self) -> float:
    """V3 reward: bounded components, defense parity, terminal dominance."""
    if not self.current_state:
        return 0.0

    game_state = self.current_state.get("game_state", {})
    if not isinstance(game_state, dict):
        return 0.0

    available_cmds = self.current_state.get("available_commands", [])
    reward = 0.0

    # ── Extract current values safely ──
    screen_type = game_state.get("screen_type", "NONE")
    current_hp = game_state.get("current_hp", 0)
    current_max_hp = game_state.get("max_hp", 80)
    current_floor = game_state.get("floor", 0)
    current_monster_hp = self._get_total_monster_hp(game_state)
    current_relic_count = len(game_state.get("relics", []))
    current_deck_size = len(game_state.get("deck", []))
    current_upgraded_cards = self._count_upgraded_cards(game_state)

    # Also try combat_state for HP if game_state doesn't have it
    combat_state = game_state.get("combat_state", {})
    if combat_state:
        player = combat_state.get("player", {})
        if current_hp == 0:
            current_hp = player.get("current_hp", 0)
        if current_max_hp == 80:
            current_max_hp = player.get("max_hp", current_max_hp)

    in_combat = self._is_in_combat(self.current_state)

    # ══════════════════════════════════════════
    # A. COMBAT REWARDS
    # ══════════════════════════════════════════
    if in_combat:
        # A1. Damage dealt
        dmg_dealt = max(0, self.last_monster_total_hp - current_monster_hp)
        reward += dmg_dealt / 100.0

        # A2. Damage taken
        hp_lost = max(0, self.last_player_hp - current_hp)
        reward -= hp_lost / 100.0

        # A3. Anti-stall (only after 40 combat steps)
        self.combat_step_count += 1
        if self.combat_step_count > 40:
            reward -= 0.02

    # ══════════════════════════════════════════
    # B. MACRO-ECONOMY
    # ══════════════════════════════════════════

    # B1. Floor progress
    if current_floor > self.last_floor:
        reward += 3.0

    # B2. Combat victory
    if screen_type == "COMBAT_REWARD" and self.last_screen_type != "COMBAT_REWARD":
        reward += 2.0
        self.combat_step_count = 0

    # B3. Relic acquired
    if current_relic_count > self.last_relic_count:
        reward += 1.0

    # B4. Card upgraded
    if current_upgraded_cards > self.last_upgraded_cards:
        reward += 0.5

    # B5. Card removed (outside combat only)
    if current_deck_size < self.last_deck_size and screen_type != "NONE":
        reward += 0.5

    # B6. Healing (outside combat only)
    hp_gained = max(0, current_hp - self.last_player_hp)
    if hp_gained > 0 and not in_combat:
        reward += hp_gained / 100.0

    # ══════════════════════════════════════════
    # C. TERMINAL STATES
    # ══════════════════════════════════════════

    # C1. Death
    if current_hp <= 0 or screen_type in ["GAME_OVER", "DEATH"]:
        reward -= 20.0

    # C2. Act completion (Act 1 victory)
    current_act = game_state.get("act", 1)
    if current_act > 1 and screen_type not in ["GAME_OVER", "DEATH"]:
        reward += 10.0

    # Reset combat counter when leaving combat
    if not in_combat and self.last_screen_type == "NONE":
        self.combat_step_count = 0

    # ══════════════════════════════════════════
    # UPDATE TRACKING STATE (must be LAST)
    # ══════════════════════════════════════════
    self.last_player_hp = current_hp
    self.last_max_hp = current_max_hp
    self.last_monster_total_hp = current_monster_hp
    self.last_floor = current_floor
    self.last_screen_type = screen_type
    self.last_relic_count = current_relic_count
    self.last_deck_size = current_deck_size
    self.last_upgraded_cards = current_upgraded_cards

    # Periodic logging
    self.step_count += 1
    if self.step_count % 50 == 0:
        import sys
        print(
            f"[REWARD V3 #{self.step_count}] r={reward:.3f} | "
            f"hp={current_hp}/{current_max_hp} floor={current_floor} "
            f"screen={screen_type} combat_steps={self.combat_step_count}",
            file=sys.stderr
        )

    return reward
```

---

## PART 8: MIGRATION CHECKLIST

1. Add new tracking variables to `__init__()` and `reset()`
2. Add helper methods: `_get_total_monster_hp()`, `_count_upgraded_cards()`, `_is_in_combat()`
3. Replace `_calculate_reward()` with V3 implementation
4. Remove old `self.previous_hp` and `self.previous_floor` tracking
5. Verify via `sys.stderr` logs that rewards are in expected ranges during first training session
6. Monitor TensorBoard `ep_rew_mean` — should see upward trend within 10k steps
