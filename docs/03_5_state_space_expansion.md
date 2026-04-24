# 03_5_STATE_SPACE_EXPANSION: OPENING THE AGENT'S EYES

## ROLE & CONTEXT
The current Observation Space (82 dimensions) is blind to crucial mechanics: Block, Buffs/Debuffs, Deck sizes, and accurate Monster damage/multi-hits. Before implementing complex rewards, we must expand the State Encoder to feed the PPO algorithm a complete picture of the battlefield.

## CRITICAL RULE FROM GPT-5.3 REVIEW
Neural networks struggle with large, unscaled numbers. You MUST normalize all inputs.
- HP / Max_HP (Output: 0.0 to 1.0)
- Gold / 1000.0
- Block / 100.0
- Damage / 100.0

## THE NEW OBSERVATION SPACE VECTOR (~150 Dimensions)
Refactor `StateEncoder` to output a fixed-size 1D float32 numpy array. Use `.get()` safely to avoid KeyErrors outside of combat. Fill missing combat data with 0.0 when not in combat.

### 1. Global Context & Deck (Length: ~15)
- `screen_type` (One-Hot Encoded, e.g., COMBAT, MAP, SHOP, EVENT, REWARD, REST)
- `floor / 50.0`, `gold / 1000.0`, `ascension_level / 20.0`
- `current_hp / max_hp`, `energy / max_energy`
- **NEW:** `player_block / 100.0`
- **NEW:** `len(draw_pile) / 40.0`, `len(discard_pile) / 40.0`, `len(exhaust_pile) / 40.0`
- `potions_count / 5.0`

### 2. Player Powers/Status Effects (Length: 5)
Parse the `powers` array inside the player JSON object. Extract the `amount` for these specific IDs (default to 0.0 if not present):
- `Strength` (amount / 10.0)
- `Dexterity` (amount / 10.0)
- `Vulnerable` (amount / 5.0)
- `Weak` (amount / 5.0)
- `Frail` (amount / 5.0)

### 3. Hand Cards - Max 10 (Length: 10 x 8 = 80)
For each card:
- `[is_present, cost/4.0, damage/100.0, block/100.0, is_attack, is_skill, is_power, exhausts]`

### 4. Monsters - Max 5 (Length: 5 x 10 = 50)
For each monster in the `monsters` array:
- `is_present`
- `current_hp / max_hp`
- `block / 100.0`
- `intent_id` (Map intent to a normalized float, e.g., ATTACK = 0.1, DEFEND = 0.2, etc.)
- **NEW:** `move_adjusted_damage / 100.0` (Safely default to 0.0 if not attacking)
- **NEW:** `move_hits / 5.0` (Safely default to 0.0 if not attacking)
- **NEW (Monster Powers):** Extract `amount` for `Strength`, `Vulnerable`, `Weak`, and `Ritual` (common Cultist buff). Scale them down (e.g., amount / 10.0).

## TASK
Recalculate the exact exact flat size of this new array. Update the `gym.spaces.Box` in `env.py` to match this new dimension. Rewrite `encode()` in `state_encoder.py`.