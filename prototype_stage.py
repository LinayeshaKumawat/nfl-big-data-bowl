#!/usr/bin/env python3
"""
Stage-1 prototype (NO external libraries — uses only the stdlib `csv`).

Goal: confirm we can, for ONE game from Week 1, identify a valid passing play
that contains a WR running a route and a likely covering defender.

This DOES NOT plot trajectories or compute per-frame distance, because those
require the tracking file (weekN.csv), which is NOT present in this repo.
Everything here is the join/selection logic that feeds the plot the moment a
tracking file is available.
"""
import csv
import pathlib
from collections import defaultdict

BASE = str(pathlib.Path(__file__).resolve().parent)  # repo root = this script's folder
DATA = f"{BASE}/data"   # all dataset CSVs live in data/

# ---- 1. Pick one Week-1 game -------------------------------------------------
week1_games = []
with open(f"{DATA}/games.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["week"] == "1":
            week1_games.append(row)

game = week1_games[0]
GAME_ID = game["gameId"]
print("=" * 70)
print(f"SELECTED WEEK-1 GAME: {GAME_ID}  "
      f"({game['visitorTeamAbbr']} @ {game['homeTeamAbbr']}, {game['gameDate']})")
print("=" * 70)

# ---- 2. Player lookup --------------------------------------------------------
players = {}
with open(f"{DATA}/players.csv", newline="") as f:
    for row in csv.DictReader(f):
        players[row["nflId"]] = row  # keys: officialPosition, displayName, ...

# ---- 3. Scouting rows for this game, indexed by playId -----------------------
scouting_by_play = defaultdict(list)
with open(f"{DATA}/pffScoutingData.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["gameId"] == GAME_ID:
            scouting_by_play[row["playId"]].append(row)

# ---- 4. Plays for this game that are completed/attempted passes --------------
#  passResult: C=complete, I=incomplete, S=sack, IN=interception, R=scramble
PASS_RESULTS = {"C", "I", "IN"}  # the ball was actually thrown toward a target
plays = []
with open(f"{DATA}/plays.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["gameId"] == GAME_ID and row["passResult"] in PASS_RESULTS:
            plays.append(row)

print(f"\nPassing plays (C/I/IN) in this game: {len(plays)}")

# ---- 5. Find plays with a WR on a route AND a coverage defender --------------
def is_wr(nfl_id):
    p = players.get(nfl_id)
    return p is not None and p["officialPosition"] == "WR"

candidates = []
for play in plays:
    pid = play["playId"]
    rows = scouting_by_play.get(pid, [])
    wrs_on_route = [r for r in rows
                    if r["pff_role"] == "Pass Route" and is_wr(r["nflId"])]
    coverage_defenders = [r for r in rows if r["pff_role"] == "Coverage"]
    if wrs_on_route and coverage_defenders:
        candidates.append((play, wrs_on_route, coverage_defenders))

print(f"Passing plays with >=1 WR-on-route AND >=1 coverage defender: "
      f"{len(candidates)}")

# ---- 6. Show the first valid candidate in detail -----------------------------
play, wrs, defs = candidates[0]
print("\n" + "-" * 70)
print(f"SELECTED PLAY  gameId={play['gameId']}  playId={play['playId']}")
print("-" * 70)
print(f"  Q{play['quarter']}  {play['down']} & {play['yardsToGo']}  "
      f"({play['possessionTeam']} vs {play['defensiveTeam']} defense)")
print(f"  Coverage: {play['pff_passCoverage']} ({play['pff_passCoverageType']})")
print(f"  Desc: {play['playDescription']}")

print(f"\n  WR(s) running a route ({len(wrs)}):")
for r in wrs:
    pl = players[r["nflId"]]
    print(f"    - nflId={r['nflId']:>6}  {pl['displayName']:<22} "
          f"linedUp={r['pff_positionLinedUp']}")

print(f"\n  Coverage defender(s) ({len(defs)}):")
for r in defs:
    pl = players.get(r["nflId"], {"displayName": "?", "officialPosition": "?"})
    print(f"    - nflId={r['nflId']:>6}  {pl['displayName']:<22} "
          f"pos={pl['officialPosition']:<3} linedUp={r['pff_positionLinedUp']}")

# ---- 7. Suggest the WR/defender pair to track (nearest alignment) ------------
# Lined-up codes carry a side (L/R) + hash; we pair the first WR with the
# coverage defender whose lined-up label shares the same side, as a first guess.
def side(code):
    for s in ("L", "R"):
        if code.endswith(s) or f"-{s}" in code or code.startswith(s):
            return s
    return "?"

wr = wrs[0]
wr_side = side(wr["pff_positionLinedUp"])
same_side = [d for d in defs if side(d["pff_positionLinedUp"]) == wr_side]
chosen_def = (same_side or defs)[0]

print("\n" + "=" * 70)
print("PROPOSED PAIR TO TRACK ONCE TRACKING DATA IS AVAILABLE")
print("=" * 70)
wr_pl = players[wr["nflId"]]
def_pl = players.get(chosen_def["nflId"], {"displayName": "?"})
print(f"  WR       : nflId={wr['nflId']}  {wr_pl['displayName']}  "
      f"(lined up {wr['pff_positionLinedUp']})")
print(f"  Defender : nflId={chosen_def['nflId']}  {def_pl['displayName']}  "
      f"(lined up {chosen_def['pff_positionLinedUp']})")

print("\nNEXT STEP (blocked): need weekN.csv tracking rows filtered to")
print(f"  gameId={GAME_ID} AND playId={play['playId']} AND")
print(f"  nflId in ({wr['nflId']}, {chosen_def['nflId']})")
print("  columns required: frameId, x, y, s, dir, event")
print("  -> then: plot x/y from 'ball_snap' to 'pass_forward', per-frame distance.")
