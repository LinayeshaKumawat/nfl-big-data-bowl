#!/usr/bin/env python3
"""
Route Lie Detector — Stage 2: candidate deception-moment detection.

Scans ALL passing plays in tracking_2021090900.csv, detects significant WR
direction changes (cuts), matches each WR to the most plausible covering
defender over a multi-frame window, estimates defender reaction delay, and
measures separation before vs. 0.5s after the cut. Ranks the top-5 candidate
deception moments and renders an SVG of the strongest.

Design choices that honor the stated constraints:
  * Smoothing: 3-frame moving average on x/y before deriving heading/velocity,
    to reduce per-frame tracking jitter.
  * Heading from VELOCITY (atan2 of dx,dy), not the raw `dir` column, so a cut
    is a real change of travel direction. Angle differences are wrapped to
    [-180,180] so 359->1 degrees reads as +2, not -358.
  * A cut requires the WR to actually be MOVING (speed gate) so we don't call
    noise at a near-standstill a "cut".
  * Defender match: scored over a window BEFORE the cut using mean proximity
    AND mirroring (correlation of movement), not snap distance alone. If the
    top-2 candidates are close in score, the matchup is flagged UNCERTAIN.
  * Reaction delay: frames after the cut until the defender's own heading
    change exceeds a threshold; capped, and labeled "no clear reaction" if none.
  * Guards: cut must sit between ball_snap and the pass, with >=PRE frames
    before and >=POST frames after inside the route window.
  * Separation increase is reported as an OBSERVATION, never asserted as proof
    of "beating" coverage. Deception score explicitly combines cut sharpness +
    defender reaction lag + separation gained, so pure speed separation with an
    instant defender reaction scores low.

Pure stdlib (offline sandbox has no pandas/matplotlib).
"""
import csv
import math
from collections import defaultdict

BASE = "/projects/sandbox/nfl-big-data-bowl"
GAME_ID = "2021090900"
TRACK = f"{BASE}/tracking_{GAME_ID}.csv"

# ---- tunables ---------------------------------------------------------------
HZ = 10                      # frames per second
SMOOTH = 3                   # moving-average window (odd) on x/y
VEL_WIN = 2                  # frames between positions used to compute heading
CUT_MIN_ANGLE = 35.0         # deg: minimum heading change to call a cut
CUT_WINS = (2, 3, 4, 5)      # frame spans over which a cut may develop (take max)
WR_MIN_SPEED = 2.5           # yd/s: WR must be moving this fast at the cut
PRE = 3                      # frames required before cut (and sep "before" ref)
POST = 5                     # frames after cut = 0.5s separation measurement
DEF_REACT_ANGLE = 25.0       # deg: defender heading change counts as a reaction
DEF_REACT_MAXLAG = 12        # frames (1.2s) cap on how long we wait for reaction
MATCH_WIN = 5                # frames before cut used to score defender match
UNCERTAIN_RATIO = 0.80       # if 2nd-best match score >= 0.8*best -> uncertain


# ---- reference tables -------------------------------------------------------
players = {}
with open(f"{BASE}/players.csv", newline="") as f:
    for row in csv.DictReader(f):
        players[row["nflId"]] = row

role, linedup = {}, {}
with open(f"{BASE}/pffScoutingData.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["gameId"] == GAME_ID:
            role[(row["playId"], row["nflId"])] = row["pff_role"]
            linedup[(row["playId"], row["nflId"])] = row["pff_positionLinedUp"]

play_meta = {}
with open(f"{BASE}/plays.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["gameId"] == GAME_ID:
            play_meta[row["playId"]] = row


def pos_of(n):
    p = players.get(n)
    return p["officialPosition"] if p else "?"


def name_of(n):
    p = players.get(n)
    return p["displayName"] if p else f"id{n}"


# ---- angle helpers (wraparound-safe) ---------------------------------------
def ang_diff(a, b):
    """Smallest signed difference a-b in degrees, wrapped to [-180,180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def heading_from(p0, p1):
    """Heading in degrees of travel from p0 to p1 (atan2 on field dx,dy).
    Returns None if the two points are effectively identical (no motion)."""
    dx, dy = p1["x"] - p0["x"], p1["y"] - p0["y"]
    if dx * dx + dy * dy < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ---- load tracking, grouped by play, then by player ------------------------
# plays[playId] = { nflId: {frameId: {x,y,s,dir,team}}, '_events': {frameId:ev} }
plays = defaultdict(lambda: {"_events": {}})
with open(TRACK, newline="") as f:
    for row in csv.DictReader(f):
        pid = row["playId"]
        fid = int(row["frameId"])
        if row["event"] and row["event"] != "None":
            plays[pid]["_events"][fid] = row["event"]
        nid = row["nflId"]
        if nid in ("", "NA"):
            continue
        plays[pid].setdefault(nid, {})[fid] = {
            "x": float(row["x"]), "y": float(row["y"]),
            "s": float(row["s"]),
            "dir": float(row["dir"]) if row["dir"] not in ("", "NA") else None,
            "team": row["team"],
        }


def smooth_track(frames_dict):
    """Return ordered list of frames with moving-averaged x,y."""
    fids = sorted(frames_dict)
    sm = []
    half = SMOOTH // 2
    for i, fid in enumerate(fids):
        lo, hi = max(0, i - half), min(len(fids), i + half + 1)
        xs = [frames_dict[fids[j]]["x"] for j in range(lo, hi)]
        ys = [frames_dict[fids[j]]["y"] for j in range(lo, hi)]
        r = dict(frames_dict[fid])
        r["frameId"] = fid
        r["xs"] = sum(xs) / len(xs)
        r["ys"] = sum(ys) / len(ys)
        sm.append(r)
    return sm


def heading_series(sm):
    """heading[i] = travel heading using points i-VEL_WIN .. i (smoothed)."""
    h = [None] * len(sm)
    for i in range(VEL_WIN, len(sm)):
        p0 = {"x": sm[i - VEL_WIN]["xs"], "y": sm[i - VEL_WIN]["ys"]}
        p1 = {"x": sm[i]["xs"], "y": sm[i]["ys"]}
        h[i] = heading_from(p0, p1)
    return h


# ---- main detection ---------------------------------------------------------
candidates = []

for pid, pdata in plays.items():
    meta = play_meta.get(pid, {})
    if meta.get("passResult") not in ("C", "I", "IN"):
        continue  # only plays where a pass was actually thrown
    events = pdata["_events"]
    snap = min((f for f, e in events.items() if e == "ball_snap"), default=None)
    thrown = min((f for f, e in events.items()
                  if e in ("pass_forward", "autoevent_passforward")), default=None)
    if snap is None or thrown is None or thrown - snap < PRE + POST + 2:
        continue

    # WRs on routes + coverage defenders that have tracks on this play
    wrs = [n for n in pdata if n != "_events"
           and role.get((pid, n)) == "Pass Route" and pos_of(n) == "WR"]
    defs = [n for n in pdata if n != "_events"
            and role.get((pid, n)) == "Coverage"]
    if not wrs or not defs:
        continue

    # pre-smooth defender tracks once
    def_sm = {d: smooth_track(pdata[d]) for d in defs}
    def_sm_by_f = {d: {r["frameId"]: r for r in def_sm[d]} for d in defs}
    def_head = {d: heading_series(def_sm[d]) for d in defs}
    def_head_by_f = {d: {def_sm[d][i]["frameId"]: def_head[d][i]
                         for i in range(len(def_sm[d]))} for d in defs}

    for wr in wrs:
        sm = smooth_track(pdata[wr])
        idx_by_f = {sm[i]["frameId"]: i for i in range(len(sm))}
        head = heading_series(sm)

        def best_turn_at(i):
            """Largest wrapped heading change ending at index i, measured over
            any span in CUT_WINS. Returns (signed_turn, span) or (0.0, 0)."""
            if head[i] is None:
                return 0.0, 0
            best_t, best_w = 0.0, 0
            for w in CUT_WINS:
                j = i - w
                if j < 0 or head[j] is None:
                    continue
                # require the frames to be roughly contiguous (no big gaps)
                if sm[i]["frameId"] - sm[j]["frameId"] > w + 2:
                    continue
                t = ang_diff(head[i], head[j])
                if abs(t) > abs(best_t):
                    best_t, best_w = t, w
            return best_t, best_w

        # find candidate cut frames for this WR, then keep the single strongest
        # (one cut per route) so we don't double-count the same move.
        wr_cuts = []
        for i in range(len(sm)):
            fid = sm[i]["frameId"]
            if fid < snap + PRE or fid > thrown - POST:
                continue
            if i + 1 >= len(sm):
                continue
            if sm[i]["s"] < WR_MIN_SPEED:
                continue  # not really moving -> ignore noise
            turn, span = best_turn_at(i)
            if abs(turn) < CUT_MIN_ANGLE:
                continue
            # local peak: strongest within +/-2 frames
            neigh = [abs(best_turn_at(k)[0]) for k in range(max(0, i - 2),
                     min(len(sm), i + 3))]
            if abs(turn) < max(neigh) - 1e-9:
                continue
            wr_cuts.append((abs(turn), i, fid, turn))

        if not wr_cuts:
            continue
        # strongest cut on this route
        wr_cuts.sort(reverse=True)
        _, i, fid, turn = wr_cuts[0]

        if True:
            # ---- defender matching over MATCH_WIN frames before the cut ----
            wr_window = [sm[j] for j in range(max(0, i - MATCH_WIN), i + 1)]
            scored = []
            for d in defs:
                dfb = def_sm_by_f[d]
                pair = [(w, dfb[w["frameId"]]) for w in wr_window
                        if w["frameId"] in dfb]
                if len(pair) < 3:
                    continue
                mean_prox = sum(dist({"x": w["xs"], "y": w["ys"]},
                                     {"x": d2["xs"], "y": d2["ys"]})
                                for w, d2 in pair) / len(pair)
                # mirroring: how similar are their frame-to-frame moves
                mir = 0.0
                for k in range(1, len(pair)):
                    wdx = pair[k][0]["xs"] - pair[k - 1][0]["xs"]
                    wdy = pair[k][0]["ys"] - pair[k - 1][0]["ys"]
                    ddx = pair[k][1]["xs"] - pair[k - 1][1]["xs"]
                    ddy = pair[k][1]["ys"] - pair[k - 1][1]["ys"]
                    mir += wdx * ddx + wdy * ddy  # dot product (aligned motion)
                # lower proximity is better; higher mirroring is better
                score = mean_prox - 0.5 * mir
                scored.append((score, d, mean_prox))
            if not scored:
                continue
            scored.sort(key=lambda t: t[0])
            best_score, DEF, best_prox = scored[0]
            uncertain = False
            note = []
            if len(scored) > 1:
                second = scored[1][0]
                # scores can be negative; compare magnitude robustly
                denom = abs(best_score) if abs(best_score) > 1e-6 else 1.0
                if abs(second - best_score) / denom < (1 - UNCERTAIN_RATIO):
                    uncertain = True
                    note.append(f"ambiguous match ({name_of(DEF)} vs "
                                f"{name_of(scored[1][1])})")
            if best_prox > 6.0:
                uncertain = True
                note.append(f"defender far at cut ({best_prox:.1f}yd)")

            # ---- separation before vs 0.5s after -------------------------
            dfb = def_sm_by_f[DEF]
            f_before = sm[i - PRE]["frameId"]
            f_after = fid + POST
            if (f_before not in dfb or fid not in dfb
                    or f_after not in idx_by_f or f_after not in dfb):
                continue
            wr_b = {"x": sm[idx_by_f[f_before]]["xs"], "y": sm[idx_by_f[f_before]]["ys"]}
            wr_c = {"x": sm[i]["xs"], "y": sm[i]["ys"]}
            wr_a = {"x": sm[idx_by_f[f_after]]["xs"], "y": sm[idx_by_f[f_after]]["ys"]}
            d_b = {"x": dfb[f_before]["xs"], "y": dfb[f_before]["ys"]}
            d_c = {"x": dfb[fid]["xs"], "y": dfb[fid]["ys"]}
            d_a = {"x": dfb[f_after]["xs"], "y": dfb[f_after]["ys"]}
            sep_before = dist(wr_b, d_b)
            sep_at = dist(wr_c, d_c)
            sep_after = dist(wr_a, d_a)
            sep_gain = sep_after - sep_before

            # ---- defender reaction delay ----------------------------------
            dh = def_head_by_f[DEF]
            react_lag = None
            # defender heading just before the cut (reference)
            ref_fids = [fid - k for k in range(1, 4) if (fid - k) in dh and dh[fid - k] is not None]
            if ref_fids:
                ref_h = dh[ref_fids[0]]
                for lag in range(1, DEF_REACT_MAXLAG + 1):
                    ff = fid + lag
                    if ff in dh and dh[ff] is not None:
                        if abs(ang_diff(dh[ff], ref_h)) >= DEF_REACT_ANGLE:
                            react_lag = lag
                            break
            react_desc = (f"{react_lag/HZ:.1f}s" if react_lag is not None
                          else "no clear reaction <=1.2s")
            if react_lag is None:
                note.append("defender reaction unclear")

            # ---- deception score ------------------------------------------
            # Reward: sharp cut, LARGE defender lag (slow to react), and
            # separation gained AFTER the cut. A play where the defender reacts
            # instantly (small lag) scores low even if separation grows, which
            # guards against calling pure speed-separation "deception".
            lag_term = (react_lag / HZ) if react_lag is not None else 1.2
            score = (abs(turn) / 45.0) * (0.3 + lag_term) * max(0.0, sep_gain)

            candidates.append({
                "pid": pid, "wr": wr, "def": DEF, "cut_frame": fid,
                "turn": turn, "wr_speed": sm[i]["s"],
                "sep_before": sep_before, "sep_at": sep_at,
                "sep_after": sep_after, "sep_gain": sep_gain,
                "react_lag": react_lag, "react_desc": react_desc,
                "match_prox": best_prox, "uncertain": uncertain,
                "note": "; ".join(note) if note else "",
                "snap": snap, "thrown": thrown, "score": score,
                "coverage": meta.get("pff_passCoverage", "?"),
                "cov_type": meta.get("pff_passCoverageType", "?"),
            })

# ---- rank & report ----------------------------------------------------------
candidates.sort(key=lambda c: c["score"], reverse=True)
top5 = candidates[:5]

print(f"Scanned plays with a thrown pass; found {len(candidates)} candidate cuts.")
print("\n================ TOP 5 CANDIDATE DECEPTION MOMENTS ================")
for rank, c in enumerate(top5, 1):
    print(f"\n#{rank}  play {c['pid']}  score={c['score']:.2f}"
          f"{'   [UNCERTAIN]' if c['uncertain'] else ''}")
    print(f"    WR {name_of(c['wr'])} ({linedup.get((c['pid'],c['wr']))})"
          f"  vs  DEF {name_of(c['def'])} "
          f"({pos_of(c['def'])}, {linedup.get((c['pid'],c['def']))})")
    print(f"    coverage: {c['coverage']} ({c['cov_type']})")
    print(f"    cut frame {c['cut_frame']} (t+{(c['cut_frame']-c['snap'])/HZ:.1f}s "
          f"after snap), WR turn {c['turn']:+.0f} deg at {c['wr_speed']:.1f} yd/s")
    print(f"    separation: before={c['sep_before']:.2f}  at cut={c['sep_at']:.2f}"
          f"  +0.5s={c['sep_after']:.2f}  (gain {c['sep_gain']:+.2f} yd)")
    print(f"    defender reaction lag: {c['react_desc']}"
          f"   | nearest-at-cut {c['match_prox']:.1f} yd")
    if c["note"]:
        print(f"    NOTE: {c['note']}")

# ---- SVG of the strongest example ------------------------------------------
if not top5:
    print("\nNo candidates to visualize.")
    raise SystemExit

best = top5[0]
pid = best["pid"]
wr, DEF = best["wr"], best["def"]
sm_wr = smooth_track(plays[pid][wr])
sm_df = smooth_track(plays[pid][DEF])
wr_by = {r["frameId"]: r for r in sm_wr}
df_by = {r["frameId"]: r for r in sm_df}
f0, f1 = best["snap"], best["thrown"]
frames = [f for f in range(f0, f1 + 1) if f in wr_by and f in df_by]


def svg_best(path):
    xs = [wr_by[f]["xs"] for f in frames] + [df_by[f]["xs"] for f in frames]
    ys = [wr_by[f]["ys"] for f in frames] + [df_by[f]["ys"] for f in frames]
    pad = 3
    minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    W, Hh = 960, 480
    def X(x): return (x - minx) / (maxx - minx) * (W - 90) + 60
    def Y(y): return Hh - 60 - (y - miny) / (maxy - miny) * (Hh - 110)

    def poly(pts, color):
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    wr_pts = [(wr_by[f]["xs"], wr_by[f]["ys"]) for f in frames]
    df_pts = [(df_by[f]["xs"], df_by[f]["ys"]) for f in frames]
    cf = best["cut_frame"]
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="sans-serif">',
         f'<rect width="{W}" height="{Hh}" fill="#f6fbf3"/>']
    gx = int(minx // 5 * 5)
    while gx <= maxx:
        if minx <= gx <= maxx:
            e.append(f'<line x1="{X(gx):.1f}" y1="34" x2="{X(gx):.1f}" y2="{Hh-60}" stroke="#d6e8cf"/>')
            e.append(f'<text x="{X(gx):.1f}" y="{Hh-44}" font-size="10" fill="#7a9" text-anchor="middle">{gx}</text>')
        gx += 5
    e.append(poly(wr_pts, "#C0392b"))
    e.append(poly(df_pts, "#2471a3"))
    # snap markers
    e.append(f'<circle cx="{X(wr_pts[0][0]):.1f}" cy="{Y(wr_pts[0][1]):.1f}" r="6" fill="white" stroke="#C0392b" stroke-width="2.5"/>')
    e.append(f'<circle cx="{X(df_pts[0][0]):.1f}" cy="{Y(df_pts[0][1]):.1f}" r="6" fill="white" stroke="#2471a3" stroke-width="2.5"/>')
    # cut marker on WR path
    if cf in wr_by:
        cx, cy = wr_by[cf]["xs"], wr_by[cf]["ys"]
        e.append(f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="9" fill="none" stroke="#e67e22" stroke-width="3"/>')
        e.append(f'<text x="{X(cx)+12:.1f}" y="{Y(cy)-8:.1f}" font-size="12" font-weight="bold" fill="#e67e22">CUT ({best["turn"]:+.0f} deg)</text>')
        # defender position at the cut
        if cf in df_by:
            e.append(f'<circle cx="{X(df_by[cf]["xs"]):.1f}" cy="{Y(df_by[cf]["ys"]):.1f}" r="5" fill="#2471a3" opacity="0.6"/>')
            e.append(f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{X(df_by[cf]["xs"]):.1f}" y2="{Y(df_by[cf]["ys"]):.1f}" stroke="#999" stroke-dasharray="3,3"/>')
    # end markers (at pass)
    e.append(f'<circle cx="{X(wr_pts[-1][0]):.1f}" cy="{Y(wr_pts[-1][1]):.1f}" r="5" fill="#C0392b"/>')
    e.append(f'<circle cx="{X(df_pts[-1][0]):.1f}" cy="{Y(df_pts[-1][1]):.1f}" r="5" fill="#2471a3"/>')
    # title + caption
    e.append(f'<text x="20" y="24" font-size="15" font-weight="bold" fill="#222">Route Lie Detector — strongest candidate: play {pid}</text>')
    tag = "  [UNCERTAIN MATCH]" if best["uncertain"] else ""
    e.append(f'<text x="20" y="44" font-size="12" fill="#444">{name_of(wr)} (WR) vs {name_of(DEF)} ({pos_of(DEF)}) — {best["coverage"]} {best["cov_type"]}{tag}</text>')
    e.append(f'<text x="20" y="{Hh-20}" font-size="12" fill="#555">sep before cut {best["sep_before"]:.2f}yd -> +0.5s {best["sep_after"]:.2f}yd (gain {best["sep_gain"]:+.2f}); defender reaction {best["react_desc"]}</text>')
    e.append(f'<circle cx="640" cy="20" r="5" fill="#C0392b"/><text x="652" y="24" font-size="12">{name_of(wr)} (WR)</text>')
    e.append(f'<circle cx="800" cy="20" r="5" fill="#2471a3"/><text x="812" y="24" font-size="12">{name_of(DEF)}</text>')
    e.append('</svg>')
    with open(path, "w") as fp:
        fp.write("\n".join(e))


out = f"{BASE}/stage2_top_candidate_play{pid}.svg"
svg_best(out)
print(f"\nWrote strongest-example SVG: {out}")
