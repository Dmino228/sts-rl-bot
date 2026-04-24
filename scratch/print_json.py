import json

with open('SlayTheSpire.log', 'r', encoding='utf-8') as f:
    for line in f:
        if '"game_state"' in line or '"available_commands"' in line:
            start = line.find('{')
            if start == -1: continue
            try:
                data = json.loads(line[start:])
                gs = data.get('game_state', {})
                if 'combat_state' in gs:
                    print(json.dumps(data, indent=2))
                    break
            except Exception as e:
                pass
