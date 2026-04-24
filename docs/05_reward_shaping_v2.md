# 05_REWARD_SHAPING_V2: ADVANCED ECONOMY AND SURVIVAL

## ROLE & CONTEXT
The previous reward logic was too naive and prone to reward hacking (stalling, COMBAT_REWARD infinite loops, and ignoring macro-economy). We are implementing V2, a mathematically bounded, robust reward system.

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

## 2. THE REWARD FORMULA (`step` method)
Initialize `reward = 0.0`.
Extract current values safely using `.get()`. For upgraded cards, sum the `upgrades` field across all cards in the `deck` array.

**A. Combat (Anti-Stall & Normalized Damage):**
- If `screen_type == "NONE"` (in combat playing cards): `reward -= 0.01` (Time penalty to prevent stalling).
- `dmg_dealt = self.last_monster_total_hp - current_monster_total_hp`
- `reward += max(0, dmg_dealt) / 100.0`
- `dmg_taken = self.last_player_hp - current_player_hp`
- `reward -= (max(0, dmg_taken) / max(1, current_max_hp)) * 2.0` (Survival scaling: hurts more at low max HP).

**B. Macro-Economy (The Real StS Gameplay):**
- If `current_floor > self.last_floor`: `reward += 1.0`
- If `current_relic_count > self.last_relic_count`: `reward += 2.0` (Encourages Elites/Events).
- If `current_deck_size < self.last_deck_size`: `reward += 1.0` (Encourages Card Removal at Shops/Events).
- If `current_upgraded_cards > self.last_upgraded_cards`: `reward += 1.0` (Encourages Smithing at Campfires).

**C. Safety & Terminal States:**
- COMBAT_REWARD FIX: 
  `if screen_type == "COMBAT_REWARD" and self.last_screen_type != "COMBAT_REWARD": reward += 1.0`
- DEATH:
  `if current_player_hp <= 0 or screen_type in ["GAME_OVER", "DEATH"]: reward -= 10.0`

## 3. STATE UPDATE
At the VERY END of the `step()` function, update all `self.last_*` tracking variables with the current ones.
Log the calculated reward to `sys.stderr` every ~50 steps for Architect monitoring.