import numpy as np
from typing import Dict, Any
from gymnasium import spaces

class StateEncoder:
    """
    Parses the game_state JSON and outputs a fixed-size 1D NumPy array (101,).
    """
    def __init__(self):
        self.shape = (101,)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.shape, dtype=np.float32)
        
    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        obs = np.zeros(self.shape, dtype=np.float32)
        
        game_state = state.get("game_state", {})
        if not game_state:
            return obs
            
        screen_type = game_state.get("screen_type", "NONE")
        
        # 1. Global Context (0-8)
        screen_map = {"EVENT": 0, "MAP": 1, "REWARD": 2, "COMBAT": 3, "REST": 4, "SHOP": 5, "NONE": 6}
        screen_idx = screen_map.get(screen_type, 6)
        obs[screen_idx] = 1.0
        
        obs[7] = game_state.get("floor", 0) / 50.0
        obs[8] = game_state.get("gold", 0) / 1000.0
        
        # 2. Player Stats (9-10)
        combat_state = game_state.get("combat_state", {})
        player = combat_state.get("player", {}) if combat_state else {}
        
        current_hp = game_state.get("current_hp", player.get("current_hp", 0))
        max_hp = game_state.get("max_hp", player.get("max_hp", 1))
        energy = player.get("energy", 0)
        
        obs[9] = current_hp / max_hp if max_hp > 0 else 0.0
        obs[10] = min(energy / 3.0, 1.0)
        
        # 3. Potions (11-20)
        potions = game_state.get("potions", [])
        for i, p in enumerate(potions):
            if i >= 5:
                break
            base_idx = 11 + (i * 2)
            if p.get("id", "Potion Slot") != "Potion Slot":
                obs[base_idx] = 1.0
                obs[base_idx + 1] = 0.5 # Dummy value for now
                
        # 4. Cards and Monsters (only populated in COMBAT)
        if combat_state:
            # Cards (21-80)
            hand = combat_state.get("hand", [])
            for i, card in enumerate(hand):
                if i >= 10:
                    break
                    
                base_idx = 21 + (i * 6)
                obs[base_idx] = 1.0 # is_present
                cost = card.get("cost", 0)
                obs[base_idx + 1] = max(0.0, min(cost / 3.0, 1.0))
                obs[base_idx + 2] = min(card.get("damage", 0) / 50.0, 1.0)
                obs[base_idx + 3] = min(card.get("block", 0) / 50.0, 1.0)
                
                card_type = card.get("type", "").upper()
                obs[base_idx + 4] = 1.0 if card_type == "ATTACK" else 0.0
                obs[base_idx + 5] = 1.0 if card_type == "SKILL" else 0.0
                
            # Monsters (81-100)
            monsters = combat_state.get("monsters", [])
            for i, m in enumerate(monsters):
                if i >= 5:
                    break
                    
                base_idx = 81 + (i * 4)
                is_gone = m.get("is_gone", False)
                if is_gone or m.get("current_hp", 0) <= 0:
                    continue
                    
                obs[base_idx] = 1.0 # is_present
                m_hp = m.get("current_hp", 0)
                m_max_hp = m.get("max_hp", 1)
                obs[base_idx + 1] = m_hp / m_max_hp if m_max_hp > 0 else 0.0
                
                intent_str = m.get("intent", "NONE")
                intent_map = {
                    "NONE": 0.0, "ATTACK": 0.1, "ATTACK_BUFF": 0.2, "ATTACK_DEBUFF": 0.3,
                    "ATTACK_DEFEND": 0.4, "BUFF": 0.5, "DEBUFF": 0.6, "STRONG_DEBUFF": 0.7,
                    "DEFEND": 0.8, "DEFEND_DEBUFF": 0.9, "DEFEND_BUFF": 1.0, 
                    "MAGIC": 0.6, "SLEEP": 0.0, "STUN": 0.0, "UNKNOWN": 0.0,
                    "ESCAPE": 0.0
                }
                obs[base_idx + 2] = intent_map.get(intent_str, 0.0)
                obs[base_idx + 3] = min(m.get("move_adjusted_damage", 0) / 50.0, 1.0)
                
        return obs
