import numpy as np
import sys
import os

# Add the project root to the sys.path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from action_space import ActionMapper, ActionMasker
from state_encoder import StateEncoder

def test_combat_encoding():
    mock_state = {
        "available_commands": ["play", "end", "potion"],
        "game_state": {
            "screen_type": "COMBAT",
            "floor": 1,
            "gold": 99,
            "current_hp": 40,
            "max_hp": 80,
            "potions": [
                {"id": "Fire Potion", "can_use": True, "requires_target": False},
                {"id": "Potion Slot", "can_use": False, "requires_target": False}
            ],
            "combat_state": {
                "player": {
                    "current_hp": 40,
                    "max_hp": 80,
                    "energy": 2
                },
                "monsters": [
                    {"is_gone": False, "current_hp": 15, "max_hp": 30, "intent": "ATTACK", "move_adjusted_damage": 10},
                    {"is_gone": True, "current_hp": 0, "max_hp": 20, "intent": "NONE", "move_adjusted_damage": 0}
                ],
                "hand": [
                    {"cost": 1, "is_playable": True, "has_target": True, "damage": 6, "block": 0, "type": "ATTACK"},
                    {"cost": 3, "is_playable": False, "has_target": False, "damage": 0, "block": 0, "type": "SKILL"}, # Unplayable
                    {"cost": 0, "is_playable": True, "has_target": False, "damage": 0, "block": 5, "type": "SKILL"}
                ]
            }
        }
    }
    
    encoder = StateEncoder()
    obs = encoder.encode(mock_state)
    assert obs.shape == (101,), f"Expected shape (101,), got {obs.shape}"
    
    # Global context
    assert obs[3] == 1.0, "Screen type COMBAT should be at index 3"
    assert obs[7] == 1/50.0, "Floor should be 1/50"
    assert obs[8] == 99/1000.0, "Gold should be 99/1000"
    
    # Player Stats
    assert obs[9] == 0.5, "Player HP should be 40/80 = 0.5"
    assert abs(obs[10] - 0.666) < 0.01, "Player Energy should be 2/3 = 0.666"
    
    # Potions
    assert obs[11] == 1.0, "Potion 0 Present"
    assert obs[13] == 0.0, "Potion 1 Not Present"
    
    # Cards
    assert obs[21] == 1.0, "Card 0 Present should be 1.0"
    assert abs(obs[22] - 0.333) < 0.01, "Card 0 Cost should be 1/3 = 0.333"
    
    # Monsters
    assert obs[81] == 1.0, "Monster 0 present"
    assert obs[82] == 0.5, "Monster 0 HP should be 15/30 = 0.5"
    
    # Test ActionMapper and Masker
    masker = ActionMasker()
    mask = masker.get_mask(mock_state)
    assert mask.shape == (100,), f"Expected mask shape (100,), got {mask.shape}"
    
    valid_actions = np.where(mask == 1)[0]
    
    assert 65 in valid_actions
    assert 60 in valid_actions
    assert 10 in valid_actions
    assert 2 in valid_actions
    assert 61 not in valid_actions
    assert 1 not in valid_actions
    assert 11 not in valid_actions
    assert 68 not in valid_actions # CHOOSE should not be valid here


def test_event_encoding():
    mock_state = {
        "available_commands": ["choose", "leave"],
        "game_state": {
            "screen_type": "EVENT",
            "floor": 4,
            "gold": 150,
            "current_hp": 75,
            "max_hp": 80,
            "choice_list": ["Choice 1", "Choice 2", "Choice 3"]
        }
    }
    
    encoder = StateEncoder()
    obs = encoder.encode(mock_state)
    assert obs.shape == (101,)
    
    # Global context
    assert obs[0] == 1.0, "Screen type EVENT should be at index 0"
    assert obs[3] == 0.0, "COMBAT should be 0"
    assert obs[7] == 4/50.0
    assert obs[8] == 150/1000.0
    
    # Player Stats
    assert obs[9] == 75/80.0
    assert obs[10] == 0.0, "Energy should be 0 out of combat"
    
    # Combat features should be 0
    assert np.all(obs[21:] == 0.0), "All cards and monsters should be 0 out of combat"
    
    # Mask
    masker = ActionMasker()
    mask = masker.get_mask(mock_state)
    
    valid_actions = np.where(mask == 1)[0]
    
    # We expect LEAVE (maps to RETURN which is 67)
    # and CHOOSE 0, 1, 2 (68, 69, 70)
    assert 67 in valid_actions, "RETURN/LEAVE should be valid"
    assert 68 in valid_actions, "CHOOSE 0"
    assert 69 in valid_actions, "CHOOSE 1"
    assert 70 in valid_actions, "CHOOSE 2"
    assert 71 not in valid_actions, "CHOOSE 3 should be invalid"
    assert 65 not in valid_actions, "END should be invalid"
    assert 0 not in valid_actions, "PLAY should be invalid"
