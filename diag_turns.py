#!/usr/bin/env python3
"""Diagnostic: what do WR heading-change magnitudes actually look like?
Helps calibrate the cut threshold/window instead of guessing."""
import csv, math
from collections import defaultdict

BASE = "/projects/sandbox/nfl-big-data-bowl"
DATA = f"{BASE}/data"   # all dataset CSVs live in data/
GAME_ID = "2021090900"
role = {}
with open(f"{DATA}/pffScoutingData.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["gameId"] == GAME_ID:
            role[(r["playId"], r["nflId"])] = r["pff_role"]
players = {}
with open(f"{DATA}/players.csv", newline="") as f:
    for r in csv.DictReader(f):
        players[r["nflId"]] = r

plays = defaultdict(lambda: defaultdict(dict))
events = defaultdict(dict)
with open(f"{DATA}/tracking_{GAME_ID}.csv", newline="") as f:
    for r in csv.DictReader(f):
        pid, fid, nid = r["playId"], int(r["frameId"]), r["nflId"]
        if r["event"] not in ("", "None"):
            events[pid][fid] = r["event"]
        if nid in ("", "NA"):
            continue
        plays[pid][nid][fid] = (float(r["x"]), float(r["y"]), float(r["s"]))

def ang_diff(a, b):
    return (a - b + 180) % 360 - 180

def smooth(d):
    fids = sorted(d)
    out = {}
    for i, fid in enumerate(fids):
        lo, hi = max(0, i-1), min(len(fids), i+2)
        xs = sum(d[fids[j]][0] for j in range(lo, hi))/(hi-lo)
        ys = sum(d[fids[j]][1] for j in range(lo, hi))/(hi-lo)
        out[fid] = (xs, ys, d[fid][2])
    return out

# collect max heading change per WR-route using VEL_WIN=2, over windows 2..5
buckets = {w: 0 for w in (20,30,40,50,60,90,120)}
examples = []
for pid, pdata in plays.items():
    for nid, d in pdata.items():
        if role.get((pid,nid)) != "Pass Route":
            continue
        if players.get(nid,{}).get("officialPosition") != "WR":
            continue
        sm = smooth(d)
        fids = sorted(sm)
        # heading via 2-frame velocity
        head = {}
        for i in range(2, len(fids)):
            x0,y0,_ = sm[fids[i-2]]; x1,y1,s1 = sm[fids[i]]
            dx,dy = x1-x0, y1-y0
            if dx*dx+dy*dy > 1e-6:
                head[fids[i]] = (math.degrees(math.atan2(dy,dx)), s1)
        hk = sorted(head)
        best = 0; bestf=None
        for W in (2,3,4,5):
            for i in range(W, len(hk)):
                fa, fb = hk[i], hk[i-W]
                if fa-fb > W+2:   # frames must be contiguous-ish
                    continue
                turn = abs(ang_diff(head[fa][0], head[fb][0]))
                spd = head[fa][1]
                if spd >= 2.5 and turn > best:
                    best, bestf = turn, fa
        for thr in buckets:
            if best >= thr:
                buckets[thr]+=1
        if best>=40:
            examples.append((best, pid, players[nid]["displayName"], bestf))

print("WR-routes with a max heading change >= threshold (speed>=2.5yd/s):")
for thr in sorted(buckets):
    print(f"  >= {thr:>3} deg : {buckets[thr]}")
print(f"\nTotal WR-routes examined: "
      f"{sum(1 for pid in plays for nid in plays[pid] if role.get((pid,nid))=='Pass Route' and players.get(nid,{}).get('officialPosition')=='WR')}")
examples.sort(reverse=True)
print("\nSharpest cuts found (deg, play, WR, frame):")
for e in examples[:15]:
    print(f"  {e[0]:5.0f}  play {e[1]:>5}  {e[2]:<20} frame {e[3]}")
