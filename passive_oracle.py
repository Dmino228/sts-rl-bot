"""
Passive Oracle — Pure observer for V3.1 Reward System.

The human plays via the game UI normally. This script passively reads
every state CommunicationMod pushes, calculates the V3.1 reward delta,
and displays it in a dedicated console window + logs/passive_oracle.log.

No commands are typed — the script auto-responds with STATE (no-op)
after each observation to keep the pipe alive without interfering.
"""

import sys

# ── IMMEDIATE HANDSHAKE — must be first output ──
sys.__stdout__.write("ready\n")
sys.__stdout__.flush()

import ctypes
import json
import math
import os
import time

COMBAT_SCALE = 50
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# CONSOLE WINDOW (display-only, no human input needed)
# ──────────────────────────────────────────────────────────────

def _setup_console():
    """Spawn a dedicated console window for oracle display."""
    kernel32 = ctypes.windll.kernel32
    kernel32.FreeConsole()
    if not kernel32.AllocConsole():
        return None
    kernel32.SetConsoleTitleW("STS Oracle  —  V3.1 Reward Monitor")
    try:
        return open("CONOUT$", "w", encoding="utf-8")
    except OSError:
        return None


def _cprint(con, msg):
    if con:
        try:
            con.write(msg + "\n")
            con.flush()
        except OSError:
            pass  # Console closed (Alt+F4)


# ──────────────────────────────────────────────────────────────
# COMMUNICATIONMOD PIPE I/O
# ──────────────────────────────────────────────────────────────

def _send_command(cmd):
    try:
        sys.__stdout__.write(cmd + "\n")
        sys.__stdout__.flush()
    except (OSError, BrokenPipeError):
        pass  # Game closed


def _read_state():
    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            return {}
        if not line:
            return {}
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
            if isinstance(state, dict):
                return state
        except json.JSONDecodeError:
            continue


# ──────────────────────────────────────────────────────────────
# V3.1 HELPERS
# ──────────────────────────────────────────────────────────────

def _get_total_monster_hp(gs):
    cs = gs.get("combat_state", {})
    if not cs:
        return 0
    return sum(m.get("current_hp", 0) for m in cs.get("monsters", [])
               if not m.get("is_gone", False))


def _count_upgraded_cards(gs):
    return sum(c.get("upgrades", 0) for c in gs.get("deck", []))


def _is_in_combat(state):
    st = state.get("game_state", {}).get("screen_type", "NONE")
    return st == "NONE" and "end" in state.get("available_commands", [])


# ──────────────────────────────────────────────────────────────
# STATE FINGERPRINT (for deduplication)
# ──────────────────────────────────────────────────────────────

def _fingerprint(state, gs, hp, monster_hp):
    """Quick hash to detect real state changes vs STATE echo."""
    return (
        gs.get("floor", 0),
        gs.get("screen_type", ""),
        hp,
        monster_hp,
        gs.get("act", 1),
        len(gs.get("deck", [])),
        len(gs.get("relics", [])),
    )


# ──────────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────────

def _show(con, gs, state, in_combat, hp, max_hp, floor, act):
    screen = gs.get("screen_type", "NONE")
    cmds = state.get("available_commands", [])

    _cprint(con, "")
    _cprint(con, "=" * 55)
    _cprint(con, f"  Floor {floor} | Act {act} | HP {hp}/{max_hp}"
                 f" | {screen}")
    _cprint(con, "=" * 55)

    if in_combat:
        cs = gs.get("combat_state", {})
        p = cs.get("player", {})
        _cprint(con, f"  Energy: {p.get('energy',0)}  Block: {p.get('block',0)}")
        for i, c in enumerate(cs.get("hand", [])):
            tgt = " [T]" if c.get("has_target") else ""
            _cprint(con, f"    [{i}] {c.get('name','?')} "
                         f"({c.get('cost','?')}){tgt}")
        for i, m in enumerate(cs.get("monsters", [])):
            if m.get("is_gone"):
                continue
            _cprint(con, f"    M[{i}] {m.get('name','?')} "
                         f"{m.get('current_hp',0)}/{m.get('max_hp',0)} "
                         f"-> {m.get('intent','?')}")
    else:
        ss = gs.get("screen_state", {})
        if screen == "COMBAT_REWARD":
            for i, r in enumerate(ss.get("rewards", [])):
                _cprint(con, f"    [{i}] {r.get('reward_type','?')}")
        elif screen == "CARD_REWARD":
            for i, c in enumerate(ss.get("cards", [])):
                _cprint(con, f"    [{i}] {c.get('name','?')}")
        elif screen == "REST":
            _cprint(con, f"  Options: {ss.get('rest_options', [])}")
        elif screen == "MAP":
            _cprint(con, f"  (Map screen)")
        elif screen == "EVENT":
            ev = ss.get("event_name", "?")
            opts = ss.get("options", [])
            _cprint(con, f"  Event: {ev}")
            for o in opts:
                _cprint(con, f"    - {o.get('text', '?')}")

    _cprint(con, f"  cmds: {cmds}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    con = _setup_console()

    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "passive_oracle.log")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(f"=== Oracle Session {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_f.flush()
    _cprint(con, f"  Log: {log_path}")

    _cprint(con, "=========================================")
    _cprint(con, "  STS Passive Oracle — V3.1 Rewards")
    _cprint(con, "  Play the game normally via UI.")
    _cprint(con, "  Rewards are calculated automatically.")
    _cprint(con, "=========================================")

    # V3.1 tracking
    last_hp = None
    last_max_hp = None
    last_monster_hp = 0
    last_floor = 0
    last_screen = "NONE"
    last_act = 1
    last_relics = 1
    last_deck = 10
    last_upgrades = 0
    combat_steps = 0
    step = 0
    total_r = 0.0
    last_fp = None

    try:
        while True:
            state = _read_state()
            if not state:
                _cprint(con, "[ORACLE] Pipe closed.")
                break

            gs = state.get("game_state", {})
            if not isinstance(gs, dict):
                gs = {}

            in_game = state.get("in_game", False)

            # Not in game — throttled wait (1s) to avoid STATE spam
            if not in_game:
                _cprint(con, f"[Pre-game] cmds={state.get('available_commands',[])}")
                time.sleep(1.0)
                _send_command("STATE")
                continue

            # ── Extract state ──
            screen = gs.get("screen_type", "NONE")
            floor = gs.get("floor", 0)
            monster_hp = _get_total_monster_hp(gs)
            relics = len(gs.get("relics", []))
            deck = len(gs.get("deck", []))
            upgrades = _count_upgraded_cards(gs)
            in_combat = _is_in_combat(state)

            # Robust HP
            hp = last_hp
            max_hp = last_max_hp
            cs = gs.get("combat_state", {})
            if in_combat and cs:
                pd = cs.get("player", {})
                hp = pd.get("current_hp", hp)
                max_hp = pd.get("max_hp", max_hp)
            else:
                gh = gs.get("current_hp")
                gm = gs.get("max_hp")
                if gh is not None:
                    hp = gh
                if gm is not None:
                    max_hp = gm
            if hp is None:
                hp = gs.get("current_hp", 0)
            if max_hp is None:
                max_hp = gs.get("max_hp", 80)

            # ── Dedup: skip if state hasn't actually changed ──
            fp = _fingerprint(state, gs, hp, monster_hp)
            if fp == last_fp:
                time.sleep(0.1)
                _send_command("STATE")
                continue
            last_fp = fp

            # ── V3.1 REWARD ──
            reward = 0.0
            parts = []
            act = gs.get("act", 1)

            # A. Combat
            if in_combat:
                dmg = max(0, last_monster_hp - monster_hp)
                a1 = math.tanh(dmg / COMBAT_SCALE)
                reward += a1
                if a1 > 0.001:
                    parts.append(f"dmg:{a1:+.3f}")

                if last_hp is not None:
                    lost = max(0, last_hp - hp)
                    a2 = -math.tanh(lost / COMBAT_SCALE)
                    reward += a2
                    if a2 < -0.001:
                        parts.append(f"hit:{a2:+.3f}")

                combat_steps += 1
                if combat_steps > 40:
                    reward -= 0.02
                    parts.append("stall")

            # B. Macro
            if floor > last_floor:
                reward += 3.0
                parts.append("floor:+3")

            if screen == "COMBAT_REWARD" and last_screen != "COMBAT_REWARD":
                reward += 2.0
                combat_steps = 0
                parts.append("victory:+2")

            if relics > last_relics:
                reward += 1.0
                parts.append("relic:+1")

            if upgrades > last_upgrades:
                reward += 0.5
                parts.append("upgrade:+0.5")

            if deck < last_deck and screen != "NONE":
                reward += 0.5
                parts.append("removed:+0.5")

            if last_hp is not None:
                hg = max(0, hp - last_hp)
                if hg > 0 and not in_combat:
                    h = hg / 100.0
                    reward += h
                    parts.append(f"heal:{h:+.3f}")

            # C. Terminal
            if hp <= 0 or screen in ["GAME_OVER", "DEATH"]:
                reward -= 20.0
                parts.append("DEATH:-20")

            if act > last_act and screen not in ["GAME_OVER", "DEATH"]:
                reward += 10.0
                parts.append(f"ACT{act}:+10")

            if not in_combat and last_screen == "NONE":
                combat_steps = 0

            total_r += reward
            step += 1

            # ── DISPLAY ──
            _show(con, gs, state, in_combat, hp, max_hp, floor, act)

            icon = "▲" if reward > 0 else "▼" if reward < 0 else "─"
            _cprint(con, f"  {icon} dr={reward:+.4f}  |  Total={total_r:+.3f}"
                         f"  |  step #{step}")
            if parts:
                _cprint(con, f"  [{' | '.join(parts)}]")

            # ── LOG ──
            try:
                msg = (f"[#{step}] dr={reward:+.4f} tot={total_r:+.3f} "
                       f"HP={hp}/{max_hp} F={floor} A={act} S={screen}")
                if parts:
                    msg += f" [{' | '.join(parts)}]"
                log_f.write(msg + "\n")
                log_f.flush()
            except OSError:
                pass

            # ── UPDATE TRACKING ──
            last_hp = hp
            last_max_hp = max_hp
            last_monster_hp = monster_hp
            last_floor = floor
            last_screen = screen
            last_act = act
            last_relics = relics
            last_deck = deck
            last_upgrades = upgrades

            # ── AUTO-RESPOND: keep pipe alive without interfering ──
            _send_command("STATE")

    except (KeyboardInterrupt, EOFError, OSError, BrokenPipeError):
        _cprint(con, "[ORACLE] Shutting down...")
    finally:
        try:
            log_f.close()
        except Exception:
            pass
        _cprint(con, f"\nDone. Total={total_r:+.3f} ({step} steps)")
        _cprint(con, f"Log: {log_path}")
        _cprint(con, "\nPress Enter to close...")
        if con:
            try:
                # Keep console window open so user can read final stats
                con_in = open("CONIN$", "r")
                con_in.readline()
                con_in.close()
            except OSError:
                pass
            try:
                con.close()
            except OSError:
                pass


if __name__ == "__main__":
    main()
