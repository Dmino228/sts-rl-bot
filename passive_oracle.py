"""
Passive Oracle — Pure observer for V3.2 Reward System.

The human plays via the game UI normally. This script passively reads
every state CommunicationMod pushes, calculates the V3.2 reward delta,
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
import os
import time

HP_DELTA_SCALE = 100.0
HP_DELTA_CAP = 100

FLOOR_REWARD = 3.0
COMBAT_VICTORY_REWARD = 2.0
RELIC_REWARD = 1.0
CARD_UPGRADE_REWARD = 0.5
CARD_REMOVE_REWARD = 0.5

DEATH_PENALTY = -25.0
ACT_COMPLETION_REWARD = 15.0

COMBAT_STEP_GRACE = 80
ANTI_STALL_PENALTY = -0.02

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
    kernel32.SetConsoleTitleW("STS Oracle  —  V3.2 Reward Monitor")
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
# V3.2 HELPERS
# ──────────────────────────────────────────────────────────────

def _get_total_monster_hp(gs):
    cs = gs.get("combat_state", {})
    if not cs:
        return 0
    return sum(m.get("current_hp", 0) for m in cs.get("monsters", [])
               if not m.get("is_gone", False))


def _count_upgraded_cards(gs):
    return sum(c.get("upgrades", 0) for c in gs.get("deck", []))


def _get_relic_ids(gs):
    relic_ids = set()
    for relic in gs.get("relics", []):
        relic_id = relic.get("id") or relic.get("name")
        if relic_id:
            relic_ids.add(str(relic_id))
    return relic_ids


def _bounded_hp_reward(delta):
    clipped = max(-HP_DELTA_CAP, min(HP_DELTA_CAP, delta))
    return clipped / HP_DELTA_SCALE


def _initial_reward_tracking():
    return (
        None,    # last_hp
        None,    # last_max_hp
        None,    # last_monster_hp
        0,       # last_floor
        "NONE",  # last_screen
        1,       # last_act
        None,    # last_relic_ids
        None,    # last_deck
        None,    # last_upgrades
        False,   # last_in_combat
        0,       # combat_steps
        None,    # last_fp
    )


def _is_in_combat(state):
    st = state.get("game_state", {}).get("screen_type", "NONE")
    return st == "NONE" and "end" in state.get("available_commands", [])


# ──────────────────────────────────────────────────────────────
# STATE FINGERPRINT (for deduplication)
# ──────────────────────────────────────────────────────────────

def _fingerprint(state, gs, hp, monster_hp):
    """Quick hash to detect real state changes vs STATE echo."""
    return (
        state.get("in_game", False),
        gs.get("floor", 0),
        gs.get("screen_type", ""),
        hp,
        monster_hp,
        gs.get("act", 1),
        len(gs.get("deck", [])),
        _count_upgraded_cards(gs),
        tuple(sorted(_get_relic_ids(gs))),
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
    _cprint(con, "  STS Passive Oracle — V3.2 Rewards")
    _cprint(con, "  Play the game normally via UI.")
    _cprint(con, "  Rewards are calculated automatically.")
    _cprint(con, "=========================================")

    (
        last_hp,
        last_max_hp,
        last_monster_hp,
        last_floor,
        last_screen,
        last_act,
        last_relic_ids,
        last_deck,
        last_upgrades,
        last_in_combat,
        combat_steps,
        last_fp,
    ) = _initial_reward_tracking()
    terminal_seen = False
    step = 0
    total_r = 0.0

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
            screen = gs.get("screen_type", "NONE")
            floor = gs.get("floor", 0)

            if terminal_seen:
                if not in_game or screen in ["GAME_OVER", "DEATH"]:
                    (
                        last_hp,
                        last_max_hp,
                        last_monster_hp,
                        last_floor,
                        last_screen,
                        last_act,
                        last_relic_ids,
                        last_deck,
                        last_upgrades,
                        last_in_combat,
                        combat_steps,
                        last_fp,
                    ) = _initial_reward_tracking()
                    _cprint(con, f"[Post-run] cmds={state.get('available_commands',[])}")
                    time.sleep(1.0)
                    _send_command("STATE")
                    continue

                # A new in-game, non-terminal run started. Bootstrap it cleanly.
                (
                    last_hp,
                    last_max_hp,
                    last_monster_hp,
                    last_floor,
                    last_screen,
                    last_act,
                    last_relic_ids,
                    last_deck,
                    last_upgrades,
                    last_in_combat,
                    combat_steps,
                    last_fp,
                ) = _initial_reward_tracking()
                terminal_seen = False

            # Pre-game: throttle wait (1s) to avoid STATE spam.
            if not in_game and last_fp is None:
                _cprint(con, f"[Pre-game] cmds={state.get('available_commands',[])}")
                time.sleep(1.0)
                _send_command("STATE")
                continue

            # ── Extract state ──
            monster_hp = _get_total_monster_hp(gs)
            relic_ids = _get_relic_ids(gs)
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
            hp_is_known = hp is not None
            if max_hp is None:
                max_hp = gs.get("max_hp", 80)

            # ── Dedup: skip if state hasn't actually changed ──
            fp = _fingerprint(state, gs, hp, monster_hp)
            if fp == last_fp:
                time.sleep(0.1)
                _send_command("STATE")
                continue
            last_fp = fp

            # ── V3.2 REWARD ──
            reward = 0.0
            parts = []
            act = gs.get("act", 1)

            # A. HP progress
            if (
                last_in_combat
                and last_monster_hp is not None
                and (in_combat or screen == "COMBAT_REWARD")
            ):
                monster_delta = last_monster_hp - monster_hp
                mr = _bounded_hp_reward(monster_delta)
                reward += mr
                if abs(mr) > 0.001:
                    label = "mhp" if mr > 0 else "mheal"
                    parts.append(f"{label}:{mr:+.3f}")

            if last_hp is not None and hp_is_known:
                hp_delta = hp - last_hp
                hr = _bounded_hp_reward(hp_delta)
                reward += hr
                if abs(hr) > 0.001:
                    label = "heal" if hr > 0 else "hit"
                    parts.append(f"{label}:{hr:+.3f}")

            if in_combat:
                combat_steps += 1
                if combat_steps > COMBAT_STEP_GRACE:
                    reward += ANTI_STALL_PENALTY
                    parts.append("stall")

            # B. Macro
            if in_game and screen not in ["GAME_OVER", "DEATH"]:
                if floor > last_floor:
                    fr = FLOOR_REWARD * (floor - last_floor)
                    reward += fr
                    parts.append(f"floor:{fr:+.1f}")

                if screen == "COMBAT_REWARD" and last_screen != "COMBAT_REWARD":
                    reward += COMBAT_VICTORY_REWARD
                    combat_steps = 0
                    parts.append(f"victory:{COMBAT_VICTORY_REWARD:+.1f}")

                if last_relic_ids is not None:
                    relic_delta = len(relic_ids - last_relic_ids)
                    if relic_delta:
                        rr = RELIC_REWARD * relic_delta
                        reward += rr
                        parts.append(f"relic:{rr:+.1f}")

                if last_upgrades is not None:
                    upgrade_delta = max(0, upgrades - last_upgrades)
                    if upgrade_delta:
                        ur = CARD_UPGRADE_REWARD * upgrade_delta
                        reward += ur
                        parts.append(f"upgrade:{ur:+.1f}")

                if last_deck is not None and not in_combat:
                    removed = max(0, last_deck - deck)
                    if removed:
                        cr = CARD_REMOVE_REWARD * removed
                        reward += cr
                        parts.append(f"removed:{cr:+.1f}")

            # C. Terminal
            dead_screen = screen in ["GAME_OVER", "DEATH"]
            dead_by_hp = hp_is_known and hp <= 0
            terminal_failure = dead_by_hp or dead_screen
            if terminal_failure:
                reward += DEATH_PENALTY
                parts.append(f"DEATH:{DEATH_PENALTY:+.0f}")
                terminal_seen = True

            if act > last_act and screen not in ["GAME_OVER", "DEATH"]:
                reward += ACT_COMPLETION_REWARD
                parts.append(f"ACT{act}:{ACT_COMPLETION_REWARD:+.0f}")

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
            last_relic_ids = relic_ids
            last_deck = deck
            last_upgrades = upgrades
            last_in_combat = in_combat

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
