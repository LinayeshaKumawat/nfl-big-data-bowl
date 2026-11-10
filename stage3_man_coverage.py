#!/usr/bin/env python3
"""
Route Lie Detector — Stage 3: man-coverage-only refinement.

Changes vs Stage 2:
  * FILTER to man coverage using the verified column pff_passCoverageType=="Man"
    (values confirmed: Man / Zone / Other). In man coverage a defender is
    ASSIGNED to a receiver, so a late reaction to a cut is football-meaningful;
    in zone it is not, which is why we restrict to man here.
  * IMPROVED receiver-defender matching: over a multi-frame window before the
    cut, score each candidate defender by BOTH mean proximity and movement
    mirroring (normalized), combine into one score, and derive an explicit
    CONFIDENCE (high/medium/low) from the margin between the top two and the
    absolute closeness. Low-confidence matchups are clearly flagged.
  * KEEP Stage-2 cut detection (velocity-heading, wraparound-safe, smoothing,
    multi-span turn, speed gate, local peak, one cut/route), reaction-delay
    estimate, and before/+0.5s separation.
  * REPORT counts: man plays, WR routes, valid cuts after filtering.

Caveat kept front-of-mind in output: a defender heading change near a WR cut is
movement CONSISTENT WITH reacting to the cut -- not proof the defender was
tricked. We never claim causation.

Pure stdlib (offline sandbox: no pandas/matplotlib).
"""
import csv
import math
from collections import defaultdict

BASE = "/projects/sandbox/nfl-big-data-bowl"
GAME_ID = "2021090900"
TRACK = f"{BASE}/tracking_{GAME_ID}.csv"

# ---- tunables (unchanged from Stage 2 unless noted) -------------------------
HZ = 10
SMOOTH = 3
VEL_WIN = 2
CUT_MIN_ANGLE = 35.0
CUT_WINS = (2, 3, 4, 5)
WR_MIN_SPEED = 2.5
PRE = 3
POST = 5
DEF_REACT_ANGLE = 25.0
DEF_REACT_MAXLAG = 12
MATCH_WIN = 5
# confidence thresholds for the improved matcher
MATCH_CLOSE_YD = 2.5          # best defender within this at cut -> supports high conf
MATCH_MARGIN = 0.25           # relative gap to 2nd-best needed for high confidence

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
    p = players.get(n); return p["officialPosition"] if p else "?"


def name_of(n):
    p = players.get(n); return p["displayName"] if p else f"id{n}"


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def heading_from(p0, p1):
    dx, dy = p1["x"] - p0["x"], p1["y"] - p0["y"]
    if dx * dx + dy * dy < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ---- load tracking ----------------------------------------------------------
plays = defaultdict(lambda: {"_events": {}})
with open(TRACK, newline="") as f:
    for row in csv.DictReader(f):
        pid = row["playId"]; fid = int(row["frameId"])
        if row["event"] and row["event"] != "None":
            plays[pid]["_events"][fid] = row["event"]
        nid = row["nflId"]
        if nid in ("", "NA"):
            continue
        plays[pid].setdefault(nid, {})[fid] = {
            "x": float(row["x"]), "y": float(row["y"]), "s": float(row["s"]),
            "dir": float(row["dir"]) if row["dir"] not in ("", "NA") else None,
        }


def smooth_track(fd):
    fids = sorted(fd); half = SMOOTH // 2; out = []
    for i, fid in enumerate(fids):
        lo, hi = max(0, i - half), min(len(fids), i + half + 1)
        r = dict(fd[fid]); r["frameId"] = fid
        r["xs"] = sum(fd[fids[j]]["x"] for j in range(lo, hi)) / (hi - lo)
        r["ys"] = sum(fd[fids[j]]["y"] for j in range(lo, hi)) / (hi - lo)
        out.append(r)
    return out


def heading_series(sm):
    h = [None] * len(sm)
    for i in range(VEL_WIN, len(sm)):
        h[i] = heading_from({"x": sm[i - VEL_WIN]["xs"], "y": sm[i - VEL_WIN]["ys"]},
                            {"x": sm[i]["xs"], "y": sm[i]["ys"]})
    return h


# ---- improved man-coverage defender matcher --------------------------------
def match_defender(wr_window, def_sm_by_f, defs):
    """Return (best_def, confidence, detail) scoring candidates over the window
    before the cut using normalized proximity + mirroring. Confidence is
    high/medium/low based on margin to 2nd-best and absolute closeness."""
    scored = []
    for d in defs:
        dfb = def_sm_by_f[d]
        pair = [(w, dfb[w["frameId"]]) for w in wr_window if w["frameId"] in dfb]
        if len(pair) < 3:
            continue
        mean_prox = sum(dist({"x": w["xs"], "y": w["ys"]},
                             {"x": d2["xs"], "y": d2["ys"]})
                        for w, d2 in pair) / len(pair)
        # mirroring = mean cosine similarity of frame-to-frame motion vectors
        cs = []
        for k in range(1, len(pair)):
            wdx = pair[k][0]["xs"] - pair[k - 1][0]["xs"]
            wdy = pair[k][0]["ys"] - pair[k - 1][0]["ys"]
            ddx = pair[k][1]["xs"] - pair[k - 1][1]["xs"]
            ddy = pair[k][1]["ys"] - pair[k - 1][1]["ys"]
            wn = math.hypot(wdx, wdy); dn = math.hypot(ddx, ddy)
            if wn > 1e-6 and dn > 1e-6:
                cs.append((wdx * ddx + wdy * ddy) / (wn * dn))
        mirror = sum(cs) / len(cs) if cs else 0.0  # -1..1
        prox_at_cut = dist({"x": pair[-1][0]["xs"], "y": pair[-1][0]["ys"]},
                           {"x": pair[-1][1]["xs"], "y": pair[-1][1]["ys"]})
        # combined cost: closer + better mirroring -> lower cost
        cost = mean_prox - 2.0 * mirror
        scored.append({"d": d, "cost": cost, "prox": mean_prox,
                       "mirror": mirror, "prox_at_cut": prox_at_cut})
    if not scored:
        return None, "none", {}
    scored.sort(key=lambda s: s["cost"])
    best = scored[0]
    conf, reasons = "high", []
    if best["prox_at_cut"] > MATCH_CLOSE_YD:
        conf = "medium"; reasons.append(f"best {best['prox_at_cut']:.1f}yd at cut")
    if len(scored) > 1:
        a, b = scored[0]["cost"], scored[1]["cost"]
        spread = (abs(b - a) / (abs(a) + 1e-6))
        if spread < MATCH_MARGIN:
            conf = "low"
            reasons.append(f"ambiguous vs {name_of(scored[1]['d'])}")
    if best["mirror"] < 0.2:
        # weak co-movement means it may not be a true man assignment
        conf = "low" if conf != "low" else conf
        reasons.append(f"weak mirroring ({best['mirror']:.2f})")
    best["conf"] = conf
    best["reasons"] = "; ".join(reasons)
    return best["d"], conf, best


# ---- main loop (man coverage only) -----------------------------------------
n_man_plays = 0
n_wr_routes = 0
n_cuts = 0
candidates = []

for pid, pdata in plays.items():
    meta = play_meta.get(pid, {})
    if meta.get("passResult") not in ("C", "I", "IN"):
        continue
    if meta.get("pff_passCoverageType") != "Man":        # <-- the filter
        continue
    events = pdata["_events"]
    snap = min((f for f, e in events.items() if e == "ball_snap"), default=None)
    thrown = min((f for f, e in events.items()
                  if e in ("pass_forward", "autoevent_passforward")), default=None)
    if snap is None or thrown is None or thrown - snap < PRE + POST + 2:
        continue
    n_man_plays += 1

    wrs = [n for n in pdata if n != "_events"
           and role.get((pid, n)) == "Pass Route" and pos_of(n) == "WR"]
    defs = [n for n in pdata if n != "_events" and role.get((pid, n)) == "Coverage"]
    if not wrs or not defs:
        continue

    def_sm = {d: smooth_track(pdata[d]) for d in defs}
    def_sm_by_f = {d: {r["frameId"]: r for r in def_sm[d]} for d in defs}
    def_head = {d: heading_series(def_sm[d]) for d in defs}
    def_head_by_f = {d: {def_sm[d][i]["frameId"]: def_head[d][i]
                         for i in range(len(def_sm[d]))} for d in defs}

    for wr in wrs:
        n_wr_routes += 1
        sm = smooth_track(pdata[wr])
        idx_by_f = {sm[i]["frameId"]: i for i in range(len(sm))}
        head = heading_series(sm)

        def best_turn_at(i):
            if head[i] is None:
                return 0.0, 0
            bt, bw = 0.0, 0
            for w in CUT_WINS:
                j = i - w
                if j < 0 or head[j] is None:
                    continue
                if sm[i]["frameId"] - sm[j]["frameId"] > w + 2:
                    continue
                t = ang_diff(head[i], head[j])
                if abs(t) > abs(bt):
                    bt, bw = t, w
            return bt, bw

        wr_cuts = []
        for i in range(len(sm)):
            fid = sm[i]["frameId"]
            if fid < snap + PRE or fid > thrown - POST:
                continue
            if i + 1 >= len(sm) or sm[i]["s"] < WR_MIN_SPEED:
                continue
            turn, _ = best_turn_at(i)
            if abs(turn) < CUT_MIN_ANGLE:
                continue
            neigh = [abs(best_turn_at(k)[0]) for k in range(max(0, i - 2),
                     min(len(sm), i + 3))]
            if abs(turn) < max(neigh) - 1e-9:
                continue
            wr_cuts.append((abs(turn), i, fid, turn))
        if not wr_cuts:
            continue
        wr_cuts.sort(reverse=True)
        _, i, fid, turn = wr_cuts[0]
        n_cuts += 1

        wr_window = [sm[j] for j in range(max(0, i - MATCH_WIN), i + 1)]
        DEF, conf, mdetail = match_defender(wr_window, def_sm_by_f, defs)
        if DEF is None:
            continue

        dfb = def_sm_by_f[DEF]
        f_before = sm[i - PRE]["frameId"]; f_after = fid + POST
        if (f_before not in dfb or fid not in dfb
                or f_after not in idx_by_f or f_after not in dfb):
            continue
        wr_b = {"x": sm[idx_by_f[f_before]]["xs"], "y": sm[idx_by_f[f_before]]["ys"]}
        wr_a = {"x": sm[idx_by_f[f_after]]["xs"], "y": sm[idx_by_f[f_after]]["ys"]}
        wr_c = {"x": sm[i]["xs"], "y": sm[i]["ys"]}
        sep_before = dist(wr_b, {"x": dfb[f_before]["xs"], "y": dfb[f_before]["ys"]})
        sep_at = dist(wr_c, {"x": dfb[fid]["xs"], "y": dfb[fid]["ys"]})
        sep_after = dist(wr_a, {"x": dfb[f_after]["xs"], "y": dfb[f_after]["ys"]})
        sep_gain = sep_after - sep_before

        dh = def_head_by_f[DEF]
        react_lag = None
        ref = [dh[fid - k] for k in range(1, 4) if (fid - k) in dh and dh[fid - k] is not None]
        if ref:
            for lag in range(1, DEF_REACT_MAXLAG + 1):
                ff = fid + lag
                if ff in dh and dh[ff] is not None and abs(ang_diff(dh[ff], ref[0])) >= DEF_REACT_ANGLE:
                    react_lag = lag; break
        react_desc = f"{react_lag/HZ:.1f}s" if react_lag is not None else "no clear reaction <=1.2s"

        lag_term = (react_lag / HZ) if react_lag is not None else 1.2
        score = (abs(turn) / 45.0) * (0.3 + lag_term) * max(0.0, sep_gain)

        candidates.append({
            "pid": pid, "wr": wr, "def": DEF, "cut_frame": fid, "turn": turn,
            "wr_speed": sm[i]["s"], "sep_before": sep_before, "sep_at": sep_at,
            "sep_after": sep_after, "sep_gain": sep_gain, "react_lag": react_lag,
            "react_desc": react_desc, "conf": conf,
            "match_detail": mdetail, "snap": snap, "thrown": thrown,
            "score": score, "coverage": meta.get("pff_passCoverage", "?"),
        })

# ---- report counts ----------------------------------------------------------
print("================ MAN-COVERAGE FILTER RESULTS ================")
print(f"Man-coverage passing plays analyzed : {n_man_plays}")
print(f"WR routes within those plays        : {n_wr_routes}")
print(f"Valid cuts detected (one per route) : {n_cuts}")
print(f"Scored candidate moments            : {len(candidates)}")

candidates.sort(key=lambda c: c["score"], reverse=True)
top = candidates[:5]
print("\n============== TOP CANDIDATE DECEPTION MOMENTS (MAN) ==============")
for rank, c in enumerate(top, 1):
    m = c["match_detail"]
    print(f"\n#{rank}  play {c['pid']}  score={c['score']:.2f}  "
          f"match-confidence={c['conf'].upper()}")
    print(f"    WR {name_of(c['wr'])} ({linedup.get((c['pid'],c['wr']))}) "
          f"vs DEF {name_of(c['def'])} ({pos_of(c['def'])}, "
          f"{linedup.get((c['pid'],c['def']))})  | {c['coverage']}")
    print(f"    cut t+{(c['cut_frame']-c['snap'])/HZ:.1f}s, turn {c['turn']:+.0f} deg "
          f"@ {c['wr_speed']:.1f} yd/s")
    print(f"    separation before={c['sep_before']:.2f} -> +0.5s={c['sep_after']:.2f}"
          f" (gain {c['sep_gain']:+.2f} yd)")
    print(f"    defender movement after cut: lag {c['react_desc']} "
          f"(consistent-with-reaction, NOT proof of being tricked)")
    print(f"    match: mirroring={m.get('mirror',0):+.2f}, "
          f"mean-prox={m.get('prox',0):.1f}yd, prox-at-cut={m.get('prox_at_cut',0):.1f}yd"
          f"{'  [' + m.get('reasons','') + ']' if m.get('reasons') else ''}")

# ---- SVG of best VALID (high/medium confidence, positive gain) example ------
valid = [c for c in candidates if c["conf"] in ("high", "medium") and c["sep_gain"] > 0]
pick = valid[0] if valid else (top[0] if top else None)
if pick is None:
    print("\nNo candidates to visualize."); raise SystemExit
if not valid:
    print("\n(NOTE: no high/medium-confidence positive-gain example; "
          "visualizing best-scoring candidate with its confidence label.)")

pid = pick["pid"]; wr = pick["wr"]; DEF = pick["def"]
sm_wr = smooth_track(plays[pid][wr]); sm_df = smooth_track(plays[pid][DEF])
wr_by = {r["frameId"]: r for r in sm_wr}; df_by = {r["frameId"]: r for r in sm_df}
f0, f1 = pick["snap"], pick["thrown"]
frames = [f for f in range(f0, f1 + 1) if f in wr_by and f in df_by]


def svg(path):
    xs = [wr_by[f]["xs"] for f in frames] + [df_by[f]["xs"] for f in frames]
    ys = [wr_by[f]["ys"] for f in frames] + [df_by[f]["ys"] for f in frames]
    pad = 3; minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    W, H = 960, 480
    def X(x): return (x - minx) / (maxx - minx) * (W - 90) + 60
    def Y(y): return H - 60 - (y - miny) / (maxy - miny) * (H - 110)
    def poly(pts, c): return (f'<polyline points="'
        + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        + f'" fill="none" stroke="{c}" stroke-width="2.5"/>')
    wr_pts = [(wr_by[f]["xs"], wr_by[f]["ys"]) for f in frames]
    df_pts = [(df_by[f]["xs"], df_by[f]["ys"]) for f in frames]
    cf = pick["cut_frame"]
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#f6fbf3"/>']
    gx = int(minx // 5 * 5)
    while gx <= maxx:
        if minx <= gx <= maxx:
            e.append(f'<line x1="{X(gx):.1f}" y1="48" x2="{X(gx):.1f}" y2="{H-60}" stroke="#d6e8cf"/>')
            e.append(f'<text x="{X(gx):.1f}" y="{H-44}" font-size="10" fill="#7a9" text-anchor="middle">{gx}</text>')
        gx += 5
    e.append(poly(wr_pts, "#C0392b")); e.append(poly(df_pts, "#2471a3"))
    e.append(f'<circle cx="{X(wr_pts[0][0]):.1f}" cy="{Y(wr_pts[0][1]):.1f}" r="6" fill="white" stroke="#C0392b" stroke-width="2.5"/>')
    e.append(f'<circle cx="{X(df_pts[0][0]):.1f}" cy="{Y(df_pts[0][1]):.1f}" r="6" fill="white" stroke="#2471a3" stroke-width="2.5"/>')
    if cf in wr_by:
        cx, cy = wr_by[cf]["xs"], wr_by[cf]["ys"]
        e.append(f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="9" fill="none" stroke="#e67e22" stroke-width="3"/>')
        e.append(f'<text x="{X(cx)+12:.1f}" y="{Y(cy)-8:.1f}" font-size="12" font-weight="bold" fill="#e67e22">CUT ({pick["turn"]:+.0f} deg)</text>')
        if cf in df_by:
            e.append(f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{X(df_by[cf]["xs"]):.1f}" y2="{Y(df_by[cf]["ys"]):.1f}" stroke="#999" stroke-dasharray="3,3"/>')
    e.append(f'<circle cx="{X(wr_pts[-1][0]):.1f}" cy="{Y(wr_pts[-1][1]):.1f}" r="5" fill="#C0392b"/>')
    e.append(f'<circle cx="{X(df_pts[-1][0]):.1f}" cy="{Y(df_pts[-1][1]):.1f}" r="5" fill="#2471a3"/>')
    e.append(f'<text x="20" y="24" font-size="15" font-weight="bold" fill="#222">Route Lie Detector (MAN) — best example: play {pid}</text>')
    e.append(f'<text x="20" y="44" font-size="12" fill="#444">{name_of(wr)} (WR) vs {name_of(DEF)} ({pos_of(DEF)}) — {pick["coverage"]} | match confidence: {pick["conf"].upper()}</text>')
    e.append(f'<text x="20" y="{H-20}" font-size="12" fill="#555">sep {pick["sep_before"]:.2f}yd -> +0.5s {pick["sep_after"]:.2f}yd (gain {pick["sep_gain"]:+.2f}); defender movement lag {pick["react_desc"]} (consistent-with, not proof)</text>')
    e.append(f'<circle cx="650" cy="20" r="5" fill="#C0392b"/><text x="662" y="24" font-size="12">{name_of(wr)}</text>')
    e.append(f'<circle cx="810" cy="20" r="5" fill="#2471a3"/><text x="822" y="24" font-size="12">{name_of(DEF)}</text>')
    e.append('</svg>')
    open(path, "w").write("\n".join(e))


out = f"{BASE}/stage3_man_best_play{pid}.svg"
svg(out)
print(f"\nWrote best man-coverage example SVG: {out}")
