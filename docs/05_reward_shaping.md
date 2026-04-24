# 05_REWARD_SHAPING: DOPAMINE FOR THE AGENT (PHASE 4)

## ROLE & CONTEXT
The PPO agent is currently running but receiving a flat reward of `0.0`. You must implement a dense Reward Function inside the `SlayTheSpireEnv.step()` method. The bot must learn that dealing damage is good, taking damage is bad, and progressing through the game is the ultimate goal.

## 1. STATE TRACKING (THE DELTAS)
To calculate rewards, the environment must remember the previous state.
- In `__init__()` and `reset()`, initialize tracking variables:
  - `self.last_player_hp = 80`
  - `self.last_monster_total_hp = 0`
  - `self.last_floor = 0`

- Inside `step(action)`, BEFORE overwriting the current state, you must calculate the deltas based on the newly parsed JSON state from the game.

## 2. THE REWARD FORMULA
Implement a cumulative `reward` variable initialized to `0.0` at the start of each `step()`. Add/subtract based on these rules:

**A. Combat Metrics (The Micro Loop):**
- **Damage Dealt:** If the total HP of all living monsters decreases, reward the agent. 
  - *Calculation:* You need a helper function `_get_total_monster_hp(game_state)` that sums `current_hp` of all monsters where `is_gone` is False.
  - *Formula:* `reward += (self.last_monster_total_hp - current_monster_total_hp) * 0.1`
  - *Safety:* Use `max(0, ...)` to ensure the bot isn't penalized if a monster heals or a new monster spawns.
- **Damage Taken:** If the player's HP decreases, penalize the agent.
  - *Formula:* `reward -= (self.last_player_hp - current_player_hp) * 0.2`
  - *Note:* The penalty weight (0.2) is strictly higher than the attack weight (0.1) to strongly encourage the AI to use Block cards.

**B. Progression Metrics (The Macro Loop):**
- **Floor Reached:** Climbing the spire is the main objective.
  - *Formula:* If `current_floor > self.last_floor`: `reward += 5.0`
- **Winning Combat:** Bonus for successfully finishing an encounter.
  - *Detection:* If `screen_type` transitions from `"COMBAT"` to `"COMBAT_REWARD"`.
  - *Formula:* `reward += 2.0`
- **Death:** The ultimate failure.
  - *Formula:* If `current_hp <= 0` or `screen_type == "GAME_OVER"` or `screen_type == "DEATH"`: `reward -= 20.0`

## 3. IMPLEMENTATION DETAILS
- Ensure the tracking variables (`self.last_player_hp`, etc.) are updated at the very END of the `step()` function so they are ready for the next iteration.
- Log the calculated reward to `sys.stderr` occasionally (e.g., every 50 steps) so the Architect can monitor the dense signal.