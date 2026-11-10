#!/usr/bin/env python3
"""
Route Lie Detector — Stage 1 verification.

Confirms we can track ONE wide receiver and the most-likely covering defender
through ONE passing play, from ball_snap to pass_forward, and measure the
separation (distance) between them at every frame.

Pure standard library (no pandas/matplotlib — the offline sandbox has neither).
Outputs two SVG charts that can be opened in the file explorer.
"""
import csv
import math
from collections import defaultdict

BASE = "/projects/sandbox/nfl-big-data-bowl"
GAME_ID = "2021090900"
TRACK = f"{BASE}/tracking_{GAME_ID}.csv"

# ----------------------------------------------------------------------------
# 1. Reference data: players + scouting roles for this game
# ----------------------------------------------------------------------------
players = {}
with open(f"{BASE}/players.csv", newline="") as f:
    for row in csv.DictReader(f):
        players[row["nflId"]] = row

# scouting role per (playId, nflId) for this game
role = {}           # (playId, nflId) -> pff_role
linedup = {}        # (playId, nflId) -> pff_positionLinedUp
with open(f"{BASE}/pffScoutingData.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["gameId"] == GAME_ID:
            role[(row["playId"], row["nflId"])] = row["pff_role"]
            linedup[(row["playId"], row["nflId"])] = row["pff_positionLinedUp"]

# play metadata
play_meta = {}
with open(f"{BASE}/plays.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["gameId"] == GAME_ID:
            play_meta[row["playId"]] = row


def pos_of(nfl_id):
    p = players.get(nfl_id)
    return p["officialPosition"] if p else "?"


def name_of(nfl_id):
    p = players.get(nfl_id)
    return p["displayName"] if p else f"id{nfl_id}"


# ----------------------------------------------------------------------------
# 2. Choose the play. We use play 97 (confirmed clean snap + pass events).
# ----------------------------------------------------------------------------
PLAY_ID = "97"

# ----------------------------------------------------------------------------
# 3. Load tracking rows for this play only.
#    Structure: tracks[nflId] = list of frame dicts (sorted by frameId)
# ----------------------------------------------------------------------------
tracks = defaultdict(list)
frame_event = {}  # frameId -> event label (from any player row; ball marker uses nflId empty)
with open(TRACK, newline="") as f:
    for row in csv.DictReader(f):
        if row["playId"] != PLAY_ID:
            continue
        fid = int(row["frameId"])
        if row["event"] and row["event"] != "None":
            frame_event[fid] = row["event"]
        nid = row["nflId"]
        if nid == "" or nid == "NA":
            continue  # football row, skip for player tracking
        tracks[nid].append({
            "frameId": fid,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "s": float(row["s"]),
            "dir": float(row["dir"]) if row["dir"] not in ("", "NA") else None,
            "team": row["team"],
        })

for nid in tracks:
    tracks[nid].sort(key=lambda r: r["frameId"])

# find snap + pass frames
snap_frame = min((f for f, e in frame_event.items() if e == "ball_snap"), default=None)
pass_frame = min((f for f, e in frame_event.items()
                  if e in ("pass_forward", "autoevent_passforward")), default=None)
print(f"Play {PLAY_ID}: snap frame={snap_frame}, pass frame={pass_frame}")
print(f"Events on play: {sorted(frame_event.items())}")

# ----------------------------------------------------------------------------
# 4. Identify WRs on routes and coverage defenders that actually have tracks.
# ----------------------------------------------------------------------------
wrs = [nid for nid in tracks
       if role.get((PLAY_ID, nid)) == "Pass Route" and pos_of(nid) == "WR"]
defenders = [nid for nid in tracks if role.get((PLAY_ID, nid)) == "Coverage"]

print("\nWRs on routes:")
for nid in wrs:
    print(f"  {name_of(nid):<20} nflId={nid} linedUp={linedup.get((PLAY_ID,nid))}")
print("Coverage defenders:")
for nid in defenders:
    print(f"  {name_of(nid):<20} nflId={nid} pos={pos_of(nid):<3} "
          f"linedUp={linedup.get((PLAY_ID,nid))}")


def frame_at(track, fid):
    for r in track:
        if r["frameId"] == fid:
            return r
    return None


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ----------------------------------------------------------------------------
# 5. Pick the WR + the TRUE cover man = nearest defender to that WR at snap.
#    Pick the WR whose nearest defender is closest (cleanest man matchup).
# ----------------------------------------------------------------------------
best = None  # (wr, defender, snap_distance)
for wr in wrs:
    wr_snap = frame_at(tracks[wr], snap_frame)
    if not wr_snap:
        continue
    nearest = None
    for d in defenders:
        d_snap = frame_at(tracks[d], snap_frame)
        if not d_snap:
            continue
        dd = dist(wr_snap, d_snap)
        if nearest is None or dd < nearest[1]:
            nearest = (d, dd)
    if nearest and (best is None or nearest[1] < best[2]):
        best = (wr, nearest[0], nearest[1])

WR, DEF, snap_sep = best
print(f"\nCHOSEN MATCHUP (nearest at snap):")
print(f"  WR       : {name_of(WR)} (nflId={WR}, {linedup.get((PLAY_ID,WR))})")
print(f"  Defender : {name_of(DEF)} (nflId={DEF}, {pos_of(DEF)}, "
      f"{linedup.get((PLAY_ID,DEF))})")
print(f"  Separation at snap: {snap_sep:.2f} yards")

# ----------------------------------------------------------------------------
# 6. Build the per-frame series from snap -> pass.
# ----------------------------------------------------------------------------
wr_tr = {r["frameId"]: r for r in tracks[WR]}
df_tr = {r["frameId"]: r for r in tracks[DEF]}
frames = [f for f in range(snap_frame, pass_frame + 1)
          if f in wr_tr and f in df_tr]

series = []
for f in frames:
    w, d = wr_tr[f], df_tr[f]
    series.append({"frameId": f, "sep": dist(w, d),
                   "wx": w["x"], "wy": w["y"], "dx": d["x"], "dy": d["y"],
                   "wdir": w["dir"], "ddir": d["dir"]})

print(f"\nFrames snap->pass: {len(series)} "
      f"({(len(series)-1)/10:.1f}s at 10 Hz)")
print(f"{'frame':>5} {'WR(x,y)':>16} {'DEF(x,y)':>16} {'sep(yd)':>8}")
for s in series:
    print(f"{s['frameId']:>5}  ({s['wx']:5.1f},{s['wy']:5.1f})  "
          f"  ({s['dx']:5.1f},{s['dy']:5.1f})   {s['sep']:7.2f}")

seps = [s["sep"] for s in series]
print(f"\nSeparation: start={seps[0]:.2f}  min={min(seps):.2f}  "
      f"max={max(seps):.2f}  end(at pass)={seps[-1]:.2f} yards")

# ============================================================================
# 7. Render SVG #1: field trajectories, and SVG #2: separation-over-time.
# ============================================================================
def svg_trajectories(series, path):
    # field coords: x is 0..120 (length incl endzones), y is 0..53.3 (width)
    xs = [s["wx"] for s in series] + [s["dx"] for s in series]
    ys = [s["wy"] for s in series] + [s["dy"] for s in series]
    pad = 3
    minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    W, H = 900, 420
    def X(x): return (x - minx) / (maxx - minx) * (W - 80) + 60
    def Y(y): return H - 50 - (y - miny) / (maxy - miny) * (H - 90)

    def polyline(pts, color):
        d = " ".join(f"{X(px):.1f},{Y(py):.1f}" for px, py in pts)
        return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    def arrow(x, y, deg, color):
        # dir: degrees clockwise from north (0=+y). Convert to field dx,dy.
        rad = math.radians(deg)
        dx, dy = math.sin(rad), math.cos(rad)
        x2, y2 = x + dx * 2.0, y + dy * 2.0
        return (f'<line x1="{X(x):.1f}" y1="{Y(y):.1f}" x2="{X(x2):.1f}" '
                f'y2="{Y(y2):.1f}" stroke="{color}" stroke-width="2" '
                f'marker-end="url(#ah{color[1:]})"/>')

    wr_pts = [(s["wx"], s["wy"]) for s in series]
    df_pts = [(s["dx"], s["dy"]) for s in series]
    e = []
    e.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="sans-serif">')
    e.append(f'<rect width="{W}" height="{H}" fill="#f6fbf3"/>')
    e.append('<defs>'
             '<marker id="ahC0392b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#C0392b"/></marker>'
             '<marker id="ah2471a3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#2471a3"/></marker>'
             '</defs>')
    # yard gridlines (every 5 yd in field x)
    gx = int(minx // 5 * 5)
    while gx <= maxx:
        if minx <= gx <= maxx:
            e.append(f'<line x1="{X(gx):.1f}" y1="30" x2="{X(gx):.1f}" y2="{H-50}" '
                     f'stroke="#cfe3c8" stroke-width="1"/>')
            e.append(f'<text x="{X(gx):.1f}" y="{H-34}" font-size="10" '
                     f'fill="#7a9" text-anchor="middle">{gx}</text>')
        gx += 5
    # trajectories
    e.append(polyline(wr_pts, "#C0392b"))
    e.append(polyline(df_pts, "#2471a3"))
    # start markers (open circle) + direction arrow at snap
    s0 = series[0]
    e.append(f'<circle cx="{X(s0["wx"]):.1f}" cy="{Y(s0["wy"]):.1f}" r="6" '
             f'fill="white" stroke="#C0392b" stroke-width="2.5"/>')
    e.append(f'<circle cx="{X(s0["dx"]):.1f}" cy="{Y(s0["dy"]):.1f}" r="6" '
             f'fill="white" stroke="#2471a3" stroke-width="2.5"/>')
    if s0["wdir"] is not None:
        e.append(arrow(s0["wx"], s0["wy"], s0["wdir"], "#C0392b"))
    if s0["ddir"] is not None:
        e.append(arrow(s0["dx"], s0["dy"], s0["ddir"], "#2471a3"))
    # end markers (filled) at pass
    s1 = series[-1]
    e.append(f'<circle cx="{X(s1["wx"]):.1f}" cy="{Y(s1["wy"]):.1f}" r="5" fill="#C0392b"/>')
    e.append(f'<circle cx="{X(s1["dx"]):.1f}" cy="{Y(s1["dy"]):.1f}" r="5" fill="#2471a3"/>')
    # legend + title
    e.append(f'<text x="20" y="22" font-size="15" font-weight="bold" fill="#222">'
             f'Route Lie Detector — Play {PLAY_ID}: {name_of(WR)} (WR) vs '
             f'{name_of(DEF)} ({pos_of(DEF)}), snap→pass</text>')
    e.append(f'<circle cx="70" cy="{H-14}" r="5" fill="#C0392b"/>'
             f'<text x="82" y="{H-10}" font-size="12">{name_of(WR)} (WR)</text>')
    e.append(f'<circle cx="300" cy="{H-14}" r="5" fill="#2471a3"/>'
             f'<text x="312" y="{H-10}" font-size="12">{name_of(DEF)} ({pos_of(DEF)})</text>')
    e.append(f'<text x="560" y="{H-10}" font-size="11" fill="#666">'
             f'open circle=snap, filled=at pass, arrow=facing/dir at snap</text>')
    e.append('</svg>')
    with open(path, "w") as fp:
        fp.write("\n".join(e))


def svg_separation(series, path):
    W, H = 900, 360
    ml, mr, mt, mb = 60, 30, 40, 50
    seps = [s["sep"] for s in series]
    fids = [s["frameId"] for s in series]
    smin, smax = 0, max(seps) * 1.1
    f0, f1 = fids[0], fids[-1]
    def X(f): return ml + (f - f0) / (f1 - f0) * (W - ml - mr)
    def Y(v): return H - mb - (v - smin) / (smax - smin) * (H - mt - mb)
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    # axes
    e.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}" stroke="#333"/>')
    e.append(f'<line x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}" stroke="#333"/>')
    # y gridlines
    yv = 0
    while yv <= smax:
        e.append(f'<line x1="{ml}" y1="{Y(yv):.1f}" x2="{W-mr}" y2="{Y(yv):.1f}" '
                 f'stroke="#eee"/>')
        e.append(f'<text x="{ml-8}" y="{Y(yv)+4:.1f}" font-size="10" '
                 f'text-anchor="end" fill="#555">{yv}</text>')
        yv += 2
    # x ticks in seconds from snap
    for s in series:
        if (s["frameId"] - f0) % 5 == 0:
            secs = (s["frameId"] - f0) / 10
            e.append(f'<text x="{X(s["frameId"]):.1f}" y="{H-mb+16:.1f}" '
                     f'font-size="10" text-anchor="middle" fill="#555">{secs:.1f}s</text>')
    # separation line
    pts = " ".join(f"{X(s['frameId']):.1f},{Y(s['sep']):.1f}" for s in series)
    e.append(f'<polyline points="{pts}" fill="none" stroke="#8e44ad" stroke-width="2.5"/>')
    # dots
    for s in series:
        e.append(f'<circle cx="{X(s["frameId"]):.1f}" cy="{Y(s["sep"]):.1f}" '
                 f'r="2.5" fill="#8e44ad"/>')
    # title + labels
    e.append(f'<text x="20" y="24" font-size="15" font-weight="bold">'
             f'WR–Defender separation over time (snap → pass)</text>')
    e.append(f'<text x="{(W)/2}" y="{H-8}" font-size="12" text-anchor="middle" '
             f'fill="#333">time since snap</text>')
    e.append(f'<text x="16" y="{(H)/2}" font-size="12" text-anchor="middle" '
             f'transform="rotate(-90 16 {(H)/2})" fill="#333">separation (yards)</text>')
    e.append('</svg>')
    with open(path, "w") as fp:
        fp.write("\n".join(e))


svg_trajectories(series, f"{BASE}/play{PLAY_ID}_trajectories.svg")
svg_separation(series, f"{BASE}/play{PLAY_ID}_separation.svg")
print(f"\nWrote:\n  {BASE}/play{PLAY_ID}_trajectories.svg"
      f"\n  {BASE}/play{PLAY_ID}_separation.svg")
