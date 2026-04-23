import numpy as np
from typing import Dict, Any, List

class ActionMapper:
    """
    Maps an integer action [0, 99] to a CommunicationMod string command.
    """
    def __init__(self):
        self.action_space_size = 100

    def get_action_string(self, action_id: int, state: Dict[str, Any] = None) -> str:
        if 0 <= action_id <= 9:
            # Untargeted card
            card_idx = action_id + 1
            return f"PLAY {card_idx}"
        elif 10 <= action_id <= 59:
            # Targeted card
            offset = action_id - 10
            card_idx = (offset // 5) + 1
            target_idx = offset % 5
            return f"PLAY {card_idx} {target_idx}"
        elif 60 <= action_id <= 64:
            # Potion
            potion_idx = action_id - 60
            return f"POTION USE {potion_idx}"
        elif action_id == 65:
            return "END"
        elif action_id == 66:
            if state:
                cmds = state.get("available_commands", [])
                if "confirm" in cmds:
                    return "CONFIRM"
                elif "proceed" in cmds:
                    return "PROCEED"
                elif state.get("game_state", {}).get("screen_type") == "GRID":
                    # Fallback for GRID: confirm is the intended action
                    return "CONFIRM"
            return "PROCEED"
        elif action_id == 67:
            return "RETURN"
        elif 68 <= action_id <= 97:
            choice_idx = action_id - 68
            
            if state:
                game_state = state.get("game_state", {})
                screen_type = game_state.get("screen_type", "NONE") if game_state else "NONE"
                available_cmds = state.get("available_commands", [])
                
                if screen_type == "GRID" and "choose" not in available_cmds:
                    row = choice_idx // 5
                    col = choice_idx % 5
                    X = 500 + (col * 230)
                    Y = 400 + (row * 350)
                    return f"CLICK Left {X} {Y}"
                    
            return f"CHOOSE {choice_idx}"
        elif action_id == 98:
            return "STATE"
        elif action_id == 99:
            return "WAIT 10"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")


class ActionMasker:
    """
    Generates a boolean mask of shape (100,) where 1 is valid and 0 is invalid.
    """
    def __init__(self):
        self.action_space_size = 100

    def _get_num_choices(self, game_state: Dict[str, Any]) -> int:
        if not game_state:
            return 0
            
        if "choice_list" in game_state:
            return len(game_state["choice_list"])
            
        screen_state = game_state.get("screen_state", {})
        if screen_state:
            for key in ["cards", "rewards", "choices", "items", "next_nodes"]:
                if key in screen_state:
                    return len(screen_state[key])
                    
        if "next_nodes" in game_state:
            return len(game_state["next_nodes"])
            
        if "reward_list" in game_state:
            return len(game_state["reward_list"])
            
        return 0

    def get_mask(self, state: Dict[str, Any]) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)

        available_cmds = state.get("available_commands", [])
        
        # If available_commands is missing but we got an error, parse it from the error string
        if not available_cmds and "error" in state:
            err_str = state.get("error", "")
            if "Possible commands: [" in err_str:
                cmds_str = err_str.split("Possible commands: [")[1].split("]")[0]
                available_cmds = [c.strip() for c in cmds_str.split(",")]

        game_state = state.get("game_state", {})
        screen_type = game_state.get("screen_type", "NONE") if game_state else "NONE"

        # Global commands
        if "proceed" in available_cmds or "confirm" in available_cmds:
            mask[66] = 1
        if "return" in available_cmds or "skip" in available_cmds or "leave" in available_cmds or "cancel" in available_cmds:
            mask[67] = 1

        if "choose" in available_cmds:
            num_choices = self._get_num_choices(game_state)
            if num_choices == 0:
                import sys
                print("WARNING: Could not determine number of choices. Defaulting to CHOOSE 0 only.", file=sys.stderr)
                num_choices = 1
            
            for i in range(min(num_choices, 30)):
                mask[68 + i] = 1
                
        # Strict GRID Screen Logic Override
        # Logs prove: CommunicationMod DOES provide "choose" on GRID screens.
        # The real bug: after card selection, "cancel" appears in available_commands,
        # enabling RETURN (mask[67]). The untrained model picks RETURN over CONFIRM,
        # cancelling the selection and creating an infinite select→cancel loop.
        # Fix: on GRID, ALWAYS mask out RETURN to force the pipeline:
        #   select card → confirm. No backing out.
        # Also: selected_cards is always [] in CommunicationMod JSON, so we use
        # confirm_up as the sole authoritative signal from the game engine.
        if screen_type == "GRID":
            import sys
            screen_state = game_state.get("screen_state", {})
            confirm_up = screen_state.get("confirm_up", False)
            cards = screen_state.get("cards", [])
            num_choices = len(cards)
            
            print(
                f"[GRID DEBUG] confirm_up={confirm_up}, "
                f"total_cards={num_choices}, "
                f"available_cmds={available_cmds}",
                file=sys.stderr
            )
            
            # ALWAYS block RETURN/cancel on GRID — prevents the infinite loop
            mask[67] = 0
            
            if confirm_up:
                # Selection complete — force the bot to confirm
                # Mask out ALL CHOOSE to prevent card toggling
                for i in range(30):
                    mask[68 + i] = 0
                # Force CONFIRM as the ONLY available action
                mask[66] = 1
            else:
                # Still selecting — first clear any CHOOSE set by standard logic
                # (standard logic uses _get_num_choices which may read choice_list
                # instead of screen_state.cards, enabling wrong indices)
                for i in range(30):
                    mask[68 + i] = 0
                # Then enable only CHOOSE for actual cards in the grid
                for i in range(min(num_choices, 30)):
                    mask[68 + i] = 1
                # Block CONFIRM until a card is selected
                mask[66] = 0

        if not game_state:
            return mask

        # Combat specific logic
        # In Slay the Spire, normal combat has screen_type == "NONE". 
        # We should rely directly on available_cmds.
        if "end" in available_cmds:
            mask[65] = 1

            if "potion" in available_cmds:
                potions = game_state.get("potions", [])
                for i, p in enumerate(potions):
                    if i >= 5:
                        break
                    if p.get("can_use", False) and not p.get("requires_target", False):
                        mask[60 + i] = 1

            if "play" in available_cmds:
                combat_state = game_state.get("combat_state", {})
                player = combat_state.get("player", {})
                energy = player.get("energy", 0)
                hand = combat_state.get("hand", [])
                monsters = combat_state.get("monsters", [])
                
                valid_targets = []
                for i, m in enumerate(monsters):
                    if i >= 5:
                        break
                    if not m.get("is_gone", False) and not m.get("half_dead", False) and m.get("current_hp", 0) > 0:
                        valid_targets.append(i)

                # Strictly loop only through actual cards in hand
                num_cards = min(len(hand), 10)
                for i in range(num_cards):
                    card = hand[i]
                    
                    is_playable = card.get("is_playable")
                    cost = card.get("cost", 0)
                    
                    # Strict playability check
                    if is_playable is not None and not is_playable:
                        continue
                        
                    # If cost is -2 it's an unplayable status/curse
                    if cost == -2:
                        continue
                    
                    # Strict energy check
                    if cost >= 0 and cost > energy:
                        continue
                    
                    has_target = card.get("has_target", False)
                    
                    if has_target:
                        for t_idx in valid_targets:
                            mask[10 + (i * 5) + t_idx] = 1
                    else:
                        mask[i] = 1

        return mask
