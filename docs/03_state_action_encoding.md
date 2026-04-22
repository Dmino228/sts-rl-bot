# 03_STATE_ACTION_ENCODING: TENSORS AND ACTION MASKS (PHASE 2 - REVISED)

## ROLE & CONTEXT
The RL algorithm requires fixed-size vectors. The game has both Combat and Out-of-Combat phases. The State Encoder and Action Mapper MUST handle menus, map navigation, rewards, and combat seamlessly.

## 1. THE ACTION SPACE (DISCRETE)
Define a fixed `gymnasium.spaces.Discrete(100)` space. Map indices to string commands:
- `0-9`: `PLAY 1` to `PLAY 10` (Untargeted cards)
- `10-59`: `PLAY 1 0` to `PLAY 10 4` (Targeted cards, Target 0-4 covers up to 5 enemies)
- `60-64`: `POTION USE 0` to `POTION USE 4`
- `65`: `END` (End turn)
- `66`: `PROCEED` (Or CONFIRM)
- `67`: `RETURN` (Or SKIP/LEAVE/CANCEL)
- `68-97`: `CHOOSE 0` to `CHOOSE 29` (Used for dialogue options, selecting cards, pathing on the map, buying in shops)
- `98-99`: Reserved for future use (e.g., specific target potions).

## 2. ACTION MASKING (CRITICAL)
- Generate a binary `numpy.ndarray` of shape `(100,)`.
- You MUST inspect `game_state.get('screen_type')`.
- If `screen_type` is NOT "COMBAT", mask out ALL `PLAY` and `END` commands. Only allow `CHOOSE`, `PROCEED`, or `RETURN` based on the exact verbs found in the `available_commands` JSON array.
- If in combat, check energy costs and target availability to mask `PLAY` actions correctly.

## 3. THE OBSERVATION SPACE (BOX)
The Observation Space must be a single 1D `numpy.ndarray` of fixed size. It must include global context so the bot knows what screen it is on.
- **Global Context:** - `screen_type` encoded as a one-hot vector (e.g., EVENT, MAP, REWARD, COMBAT, REST, SHOP, NONE).
  - `floor / 50.0`, `gold / 1000.0`.
- **Player Stats:** `current_hp / max_hp`, `energy / max_energy`.
- **Potions (Max 5):** `[is_present, potion_type_id (categorical float)]`
- **Cards (Max 10):** `[is_present, cost, base_damage, base_block, is_attack, is_skill]`. IF NOT IN COMBAT, fill these 60 slots with zeros.
- **Monsters (Max 5):** `[is_present, current_hp / max_hp, intent_id, attack_damage]`. IF NOT IN COMBAT, fill these 20 slots with zeros.