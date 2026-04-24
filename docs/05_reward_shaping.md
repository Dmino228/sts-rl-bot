# 05_REWARD_SHAPING: DOPAMINE FOR THE AGENT (PHASE 4 - VERSION 2.0)

## ROLE & CONTEXT
The PPO agent currently has 20/20 vision (Phase 3.5) but lacks a dense reward signal. We are implementing a highly tuned Reward Function inside the `SlayTheSpireEnv.step()` method. The agent must learn macro-economy (Gold, Max HP, Floors) and micro-combat (Damage dealt vs. HP lost).

## 1. STATE TRACKING (THE DELTAS)
To calculate rewards, the environment must remember the state from the *previous* step.
- In `__init__()` and `reset()`, initialize tracking variables:
  - `self.last_player_hp = 80`
  - `self.last_max_hp = 80`
  - `self.last_gold = 99`
  - `self.last_floor = 0`
  - `self.last_monster_total_hp = 0`
  - `self.last_screen_type = "NONE"`

## 2. THE REWARD FORMULA V2.0
Initialize `reward = 0.0` at the start of `step()`. Parse the current state safely.

**A. Combat Metrics (Calculated ONLY if `current_screen_type == "COMBAT"` AND `self.last_screen_type == "COMBAT"`):**
- **Damage Dealt:** Calculate `monster_hp_delta = self.last_monster_total_hp - current_monster_total_hp`. 
  - If `monster_hp_delta > 0`: `reward += monster_hp_delta * 0.1`
- **Damage Taken:** Calculate `player_hp_delta = current_player_hp - self.last_player_hp`.
  - If `player_hp_delta < 0` (Took damage): `reward += player_hp_delta * 0.2` (Penalty is 2x stronger than attack reward to force blocking).
  - If `player_hp_delta > 0` (Healed, e.g., via Burning Blood or Potions): `reward += player_hp_delta * 0.1`

**B. Macro-Economy Metrics (Calculated on ALL screens):**
- **Floor Progress:** If `current_floor > self.last_floor`: `reward += 5.0`
- **Gold Gained:** If `current_gold > self.last_gold`: `reward += (current_gold - self.last_gold) * 0.01` (e.g., gaining 100 gold = +1.0)
- **Max HP Gained:** If `current_max_hp > self.last_max_hp`: `reward += 2.0`

**C. Milestones & Penalties:**
- **Victory:** If `self.last_screen_type == "COMBAT"` and `current_screen_type == "COMBAT_REWARD"`: `reward += 2.0`
- **Death:** If `current_player_hp <= 0` or `current_screen_type in ["GAME_OVER", "DEATH"]`: `reward -= 20.0`

## 3. IMPLEMENTATION DETAILS
- Create a safe helper `_get_total_monster_hp(game_state)` that sums `current_hp` of all monsters where `is_gone == False`.
- Update ALL tracking variables (`self.last_player_hp`, etc.) at the very END of the `step()` method to prepare for the next step.
- Add a periodic log to `sys.stderr` every 100 steps to print the calculated reward, so the Architect can verify the dense signal.