import numpy as np
from typing import Dict, Any
from gymnasium import spaces
import math

class StateEncoder:
    """
    Parses the game_state JSON and outputs a fixed-size 1D NumPy array (205,).
    """
    def __init__(self):
        self.shape = (205,)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.shape, dtype=np.float32)
        
    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        obs = np.zeros(self.shape, dtype=np.float32)
        
        game_state = state.get("game_state", {})
        if not game_state:
            return obs
            
        screen_type = game_state.get("screen_type", "NONE")
        
        # 1. Global Context & Deck (0-16)
        screen_map = {"EVENT": 0, "MAP": 1, "REWARD": 2, "COMBAT": 3, "REST": 4, "SHOP": 5, "NONE": 6}
        screen_idx = screen_map.get(screen_type, 6)
        obs[screen_idx] = 1.0
        
        obs[7] = min(game_state.get("floor", 0) / 55.0, 1.0)
        obs[8] = math.tanh(game_state.get("gold", 0) / 500.0)
        obs[9] = game_state.get("ascension_level", 0) / 20.0
        
        combat_state = game_state.get("combat_state", {})
        player = combat_state.get("player", {}) if combat_state else {}
        
        current_hp = game_state.get("current_hp", player.get("current_hp", 0))
        max_hp = game_state.get("max_hp", player.get("max_hp", 1))
        energy = player.get("energy", 0)
        
        obs[10] = current_hp / max_hp if max_hp > 0 else 0.0
        obs[11] = min(energy / 3.0, 1.0)
        
        # NEW: Player Block
        obs[12] = math.tanh(player.get("block", 0) / 50.0)
        
        # NEW: Deck sizes
        obs[13] = len(combat_state.get("draw_pile", [])) / 40.0
        obs[14] = len(combat_state.get("discard_pile", [])) / 40.0
        obs[15] = len(combat_state.get("exhaust_pile", [])) / 40.0
        
        potions = game_state.get("potions", [])
        valid_potions = [p for p in potions if p.get("id", "Potion Slot") != "Potion Slot"]
        obs[16] = len(valid_potions) / 5.0
        
        # 2. Player Powers/Status Effects (17-21)
        player_powers = player.get("powers", [])
        powers_dict = {p.get("id"): p.get("amount", 0) for p in player_powers}
        
        obs[17] = powers_dict.get("Strength", 0) / 10.0
        obs[18] = powers_dict.get("Dexterity", 0) / 10.0
        obs[19] = powers_dict.get("Vulnerable", 0) / 5.0
        obs[20] = powers_dict.get("Weak", 0) / 5.0
        obs[21] = powers_dict.get("Frail", 0) / 5.0
        
        # 3. Hand Cards - Max 10 (22-101)
        hand = combat_state.get("hand", [])
        for i, card in enumerate(hand):
            if i >= 10:
                break
            base_idx = 22 + (i * 8)
            obs[base_idx] = 1.0 # is_present
            
            cost = card.get("cost", 0)
            if cost == -1:
                cost = player.get("energy", 0)
            obs[base_idx + 1] = max(0.0, min(cost / 5.0, 1.0))
            obs[base_idx + 2] = math.tanh(card.get("damage", 0) / 50.0)
            obs[base_idx + 3] = math.tanh(card.get("block", 0) / 50.0)
            
            card_type = card.get("type", "").upper()
            obs[base_idx + 4] = 1.0 if card_type == "ATTACK" else 0.0
            obs[base_idx + 5] = 1.0 if card_type == "SKILL" else 0.0
            obs[base_idx + 6] = 1.0 if card_type == "POWER" else 0.0
            obs[base_idx + 7] = 1.0 if card.get("exhausts", False) else 0.0
            
        # 4. Monsters - Max 5 (102-151)
        monsters = combat_state.get("monsters", [])
        for i, m in enumerate(monsters):
            if i >= 5:
                break
                
            base_idx = 102 + (i * 10)
            is_gone = m.get("is_gone", False)
            if is_gone or m.get("current_hp", 0) <= 0:
                continue
                
            obs[base_idx] = 1.0 # is_present
            m_hp = m.get("current_hp", 0)
            m_max_hp = m.get("max_hp", 1)
            obs[base_idx + 1] = m_hp / m_max_hp if m_max_hp > 0 else 0.0
            obs[base_idx + 2] = math.tanh(m.get("block", 0) / 50.0)
            
            intent_str = m.get("intent", "NONE")
            intent_map = {
                "NONE": 0.0, "ATTACK": 0.1, "ATTACK_BUFF": 0.2, "ATTACK_DEBUFF": 0.3,
                "ATTACK_DEFEND": 0.4, "BUFF": 0.5, "DEBUFF": 0.6, "STRONG_DEBUFF": 0.7,
                "DEFEND": 0.8, "DEFEND_DEBUFF": 0.9, "DEFEND_BUFF": 1.0, 
                "MAGIC": 0.6, "SLEEP": 0.0, "STUN": 0.0, "UNKNOWN": 0.0,
                "ESCAPE": 0.0
            }
            obs[base_idx + 3] = intent_map.get(intent_str, 0.0)
            obs[base_idx + 4] = math.tanh(m.get("move_adjusted_damage", 0) / 50.0)
            obs[base_idx + 5] = min(m.get("move_hits", 0) / 5.0, 1.0)
            
            m_powers = m.get("powers", [])
            m_powers_dict = {p.get("id"): p.get("amount", 0) for p in m_powers}
            
            obs[base_idx + 6] = m_powers_dict.get("Strength", 0) / 10.0
            obs[base_idx + 7] = m_powers_dict.get("Vulnerable", 0) / 10.0
            obs[base_idx + 8] = m_powers_dict.get("Weak", 0) / 10.0
            obs[base_idx + 9] = m_powers_dict.get("Ritual", 0) / 10.0
            
        # 5. Watcher Stances - 3 bits (152-154)
        stance_str = str(player.get("stance", "")).upper()
        if 152 < self.shape[0]:
            obs[152] = 1.0 if stance_str == "WRATH" else 0.0
        if 153 < self.shape[0]:
            obs[153] = 1.0 if stance_str == "CALM" else 0.0
        if 154 < self.shape[0]:
            obs[154] = 1.0 if stance_str == "DIVINITY" else 0.0

        # 6. Defect Orbs - Max 10 slots (155-204)
        orbs = player.get("orbs", [])
        for i in range(10):
            base_idx = 155 + (i * 5)
            if base_idx + 4 >= self.shape[0]:
                break
            if i < len(orbs):
                orb = orbs[i]
                orb_id = str(orb.get("id", "Empty")).upper()
            else:
                orb_id = "EMPTY"
            
            obs[base_idx] = 1.0 if orb_id == "LIGHTNING" else 0.0
            obs[base_idx + 1] = 1.0 if orb_id == "FROST" else 0.0
            obs[base_idx + 2] = 1.0 if orb_id == "DARK" else 0.0
            obs[base_idx + 3] = 1.0 if orb_id == "PLASMA" else 0.0
            obs[base_idx + 4] = 1.0 if orb_id == "EMPTY" else 0.0

        return obs
