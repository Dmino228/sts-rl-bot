# 05_REWARD_SHAPING_V3: MATHEMATICALLY VERIFIED DENSE REWARD SYSTEM

## ROLE & CONTEXT
V2 had weight imbalances that biased the agent toward aggression, punished defensive play, and undervalued survival. V3 uses bounded components, defense parity, and terminal dominance to produce stable gradients for PPO training.

## 1. STATE TRACKING (THE DELTAS)
In `SlayTheSpireEnv.__init__()` and `reset()`, initialize:
- `self.last_player_hp = 80`
- `self.last_max_hp = 80`
- `self.last_monster_total_hp = 0`
- `self.last_floor = 0`
- `self.last_screen_type = "NONE"`
- `self.last_relic_count = 1`
- `self.last_deck_size = 10`
- `self.last_upgraded_cards = 0`
- `self.step_count = 0`
- `self.combat_step_count = 0`

## 2. HELPER FUNCTIONS
- `_get_total_monster_hp(game_state)`: Sum `current_hp` of all monsters where `is_gone == False`.
- `_count_upgraded_cards(game_state)`: Sum `upgrades` field across all cards in `deck` array.
- `_is_in_combat(state)`: Returns True when `screen_type == "NONE"` AND `"end" in available_commands`.

## 3. THE REWARD FORMULA V3.0

Initialize `reward = 0.0`.

**A. Combat (ONLY when `_is_in_combat()` is True):**
- **A1. Damage Dealt:** `reward += max(0, dmg_dealt) / 100.0`
- **A2. Damage Taken:** `reward -= max(0, hp_lost) / 100.0` — Same scale as damage dealt. No asymmetric bias.
- **A3. Anti-Stall:** Only after 40+ combat steps: `reward -= 0.02`. Normal play is never penalized.

**B. Macro-Economy (ALL screens):**
- **B1. Floor Progress:** `reward += 3.0` — PRIMARY signal, highest per-event macro reward.
- **B2. Combat Victory:** `if screen_type == "COMBAT_REWARD" and last != "COMBAT_REWARD": reward += 2.0`
- **B3. Relic Acquired:** `reward += 1.0` — Below floor/victory to prevent relic gambling.
- **B4. Card Upgrade:** `reward += 0.5` — Mild incentive for smithing.
- **B5. Card Removal:** `reward += 0.5` — Only outside combat (`screen_type != "NONE"`) to filter exhaust noise.
- **B6. Healing:** `reward += hp_gained / 100.0` — Only outside combat. Same `/100.0` scale.

**C. Terminal States:**
- **C1. Death:** `reward -= 20.0` — Dominant penalty. Wipes a full run's gains.
- **C2. Act Completion:** `reward += 10.0` — Victory signal when `act > 1` and not dead.

## 4. KEY CHANGES FROM V2

| Aspect | V2 | V3 | Rationale |
|---|---|---|---|
| Time penalty | `-0.01` every step | `-0.02` only after 40 steps | V2 punished blocking |
| Damage taken | `-(dmg/max_hp)*2.0` | `-dmg/100.0` | V2 coupled combat with max_hp macro decisions |
| Floor progress | `+1.0` | `+3.0` | V2 undervalued the most important signal |
| Relic | `+2.0` | `+1.0` | V2 made relics the dominant reward |
| Combat victory | `+1.0` | `+2.0` | V2 equated winning fights with upgrading cards |
| Death | `-10.0` | `-20.0` | V2 allowed net-positive death on high floors |
| Card removal guard | none | `screen_type != "NONE"` | V2 triggered on exhaust during combat |

## 5. REWARD BUDGET

- **Successful Act 1 run (~15 floors):** `+40` to `+80` net reward
- **Failed run (~floor 8):** `-5` to `+5` net (death penalty erases gains)
- **Per-step combat range:** `[-0.82, +0.5]`
- **Death is ALWAYS net-negative** in the value function

## 6. IMPLEMENTATION
- Update ALL `self.last_*` at the VERY END of `_calculate_reward()`.
- Reset `self.combat_step_count = 0` on combat victory AND when leaving combat.
- Log reward to `sys.stderr` every 50 steps for monitoring.
