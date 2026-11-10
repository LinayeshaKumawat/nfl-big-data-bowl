#!/usr/bin/env python3
"""
Route Lie Detector — Stage 4: multi-game man-coverage runner.

Runs the SAME man-coverage detector as Stage 3, but across EVERY
tracking_<gameId>.csv present in the repo folder. Thresholds are IDENTICAL to
Stage 3 (no loosening) so results are comparable as the sample grows.

Pipeline per play (man coverage only):
  smoothing -> velocity-heading (wraparound-safe) -> multi-span cut detection
  (speed-gated, local peak, one cut/route) -> improved defender match over a
  pre-cut window (proximity + mirroring -> confidence) -> reaction-delay ->
  before/+0.5s separation -> deception score.

Outputs:
  * counts: man plays, WR routes, valid cuts, positive-separation candidates
  * ranked candidates (prioritizing reliable matchups + positive separation)
  * SVGs for the 3 strongest validated candidates

Caveat preserved: defender movement near a cut is CONSISTENT WITH reacting,
not proof of being tricked. Positive separation is NOT proof of deception.

Pure stdlib (offline sandbox: no pandas/matplotlib).
"""
import csv
import glob
import math
import os
from collections import defaultdict

BASE = "/projects/sandbox/nfl-big-data-bowl"
DATA = f"{BASE}/data"   # all dataset CSVs live in data/

# ---- thresholds (IDENTICAL to Stage 3) --------------------------------------
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
MATCH_CLOSE_YD = 2.5
MATCH_MARGIN = 0.25

# ---- reference tables (shared across games) ---------------------------------
players = {}
with open(f"{DATA}/players.csv", newline="") as f:
    for row in csv.DictReader(f):
        players[row["nflId"]] = row

# role/linedup keyed by (gameId, playId, nflId); meta keyed by (gameId, playId)
role, linedup = {}, {}
with open(f"{DATA}/pffScoutingData.csv", newline="") as f:
    for row in csv.DictReader(f):
        k = (row["gameId"], row["playId"], row["nflId"])
        role[k] = row["pff_role"]
        linedup[k] = row["pff_positionLinedUp"]

play_meta = {}
with open(f"{DATA}/plays.csv", newline="") as f:
    for row in csv.DictReader(f):
        play_meta[(row["gameId"], row["playId"])] = row

game_label = {}
with open(f"{DATA}/games.csv", newline="") as f:
    for row in csv.DictReader(f):
        game_label[row["gameId"]] = f"{row['visitorTeamAbbr']} @ {row['homeTeamAbbr']}"


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


def match_defender(wr_window, def_sm_by_f, defs):
    scored = []
    for d in defs:
        dfb = def_sm_by_f[d]
        pair = [(w, dfb[w["frameId"]]) for w in wr_window if w["frameId"] in dfb]
        if len(pair) < 3:
            continue
        mean_prox = sum(dist({"x": w["xs"], "y": w["ys"]},
                             {"x": d2["xs"], "y": d2["ys"]})
                        for w, d2 in pair) / len(pair)
        cs = []
        for k in range(1, len(pair)):
            wdx = pair[k][0]["xs"] - pair[k - 1][0]["xs"]
            wdy = pair[k][0]["ys"] - pair[k - 1][0]["ys"]
            ddx = pair[k][1]["xs"] - pair[k - 1][1]["xs"]
            ddy = pair[k][1]["ys"] - pair[k - 1][1]["ys"]
            wn = math.hypot(wdx, wdy); dn = math.hypot(ddx, ddy)
            if wn > 1e-6 and dn > 1e-6:
                cs.append((wdx * ddx + wdy * ddy) / (wn * dn))
        mirror = sum(cs) / len(cs) if cs else 0.0
        prox_at_cut = dist({"x": pair[-1][0]["xs"], "y": pair[-1][0]["ys"]},
                           {"x": pair[-1][1]["xs"], "y": pair[-1][1]["ys"]})
        cost = mean_prox - 2.0 * mirror
        scored.append({"d": d, "cost": cost, "prox": mean_prox,
                       "mirror": mirror, "prox_at_cut": prox_at_cut})
    if not scored:
        return None, "none", {}
    scored.sort(key=lambda s: s["cost"])
    best = scored[0]; conf, reasons = "high", []
    if best["prox_at_cut"] > MATCH_CLOSE_YD:
        conf = "medium"; reasons.append(f"best {best['prox_at_cut']:.1f}yd at cut")
    if len(scored) > 1:
        a, b = scored[0]["cost"], scored[1]["cost"]
        if (abs(b - a) / (abs(a) + 1e-6)) < MATCH_MARGIN:
            conf = "low"; reasons.append(f"ambiguous vs {name_of(scored[1]['d'])}")
    if best["mirror"] < 0.2:
        conf = "low"; reasons.append(f"weak mirroring ({best['mirror']:.2f})")
    best["conf"] = conf; best["reasons"] = "; ".join(reasons)
    return best["d"], conf, best


def analyze_game(track_path, gid):
    plays = defaultdict(lambda: {"_events": {}})
    with open(track_path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["playId"]; fid = int(row["frameId"])
            if row["event"] and row["event"] != "None":
                plays[pid]["_events"][fid] = row["event"]
            nid = row["nflId"]
            if nid in ("", "NA"):
                continue
            plays[pid].setdefault(nid, {})[fid] = {
                "x": float(row["x"]), "y": float(row["y"]), "s": float(row["s"]),
            }

    counts = {"man_plays": 0, "wr_routes": 0, "cuts": 0}
    cands = []
    for pid, pdata in plays.items():
        meta = play_meta.get((gid, pid), {})
        if meta.get("passResult") not in ("C", "I", "IN"):
            continue
        if meta.get("pff_passCoverageType") != "Man":
            continue
        events = pdata["_events"]
        snap = min((f for f, e in events.items() if e == "ball_snap"), default=None)
        thrown = min((f for f, e in events.items()
                      if e in ("pass_forward", "autoevent_passforward")), default=None)
        if snap is None or thrown is None or thrown - snap < PRE + POST + 2:
            continue
        counts["man_plays"] += 1

        wrs = [n for n in pdata if n != "_events"
               and role.get((gid, pid, n)) == "Pass Route" and pos_of(n) == "WR"]
        defs = [n for n in pdata if n != "_events"
                and role.get((gid, pid, n)) == "Coverage"]
        if not wrs or not defs:
            continue
        def_sm = {d: smooth_track(pdata[d]) for d in defs}
        def_sm_by_f = {d: {r["frameId"]: r for r in def_sm[d]} for d in defs}
        def_head = {d: heading_series(def_sm[d]) for d in defs}
        def_head_by_f = {d: {def_sm[d][i]["frameId"]: def_head[d][i]
                             for i in range(len(def_sm[d]))} for d in defs}

        for wr in wrs:
            counts["wr_routes"] += 1
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
            counts["cuts"] += 1

            wr_window = [sm[j] for j in range(max(0, i - MATCH_WIN), i + 1)]
            DEF, conf, md = match_defender(wr_window, def_sm_by_f, defs)
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

            dh = def_head_by_f[DEF]; react_lag = None
            ref = [dh[fid - k] for k in range(1, 4) if (fid - k) in dh and dh[fid - k] is not None]
            if ref:
                for lag in range(1, DEF_REACT_MAXLAG + 1):
                    ff = fid + lag
                    if ff in dh and dh[ff] is not None and abs(ang_diff(dh[ff], ref[0])) >= DEF_REACT_ANGLE:
                        react_lag = lag; break
            react_desc = f"{react_lag/HZ:.1f}s" if react_lag is not None else "no clear reaction <=1.2s"

            lag_term = (react_lag / HZ) if react_lag is not None else 1.2
            score = (abs(turn) / 45.0) * (0.3 + lag_term) * max(0.0, sep_gain)

            cands.append({
                "gid": gid, "pid": pid, "wr": wr, "def": DEF, "cut_frame": fid,
                "turn": turn, "wr_speed": sm[i]["s"], "sep_before": sep_before,
                "sep_at": sep_at, "sep_after": sep_after, "sep_gain": sep_gain,
                "react_lag": react_lag, "react_desc": react_desc, "conf": conf,
                "mirror": md.get("mirror", 0), "prox_at_cut": md.get("prox_at_cut", 0),
                "reasons": md.get("reasons", ""), "snap": snap, "thrown": thrown,
                "score": score, "coverage": meta.get("pff_passCoverage", "?"),
            })
    return counts, cands, plays


# ---- discover & run all available games ------------------------------------
track_files = sorted(glob.glob(f"{DATA}/tracking_*.csv"))
print(f"Tracking files found: {len(track_files)}")
for tf in track_files:
    gid = os.path.basename(tf)[len("tracking_"):-len(".csv")]
    print(f"  {os.path.basename(tf)}  ({game_label.get(gid,'?')})")
print()

total = {"man_plays": 0, "wr_routes": 0, "cuts": 0}
all_cands = []
plays_by_game = {}
for tf in track_files:
    gid = os.path.basename(tf)[len("tracking_"):-len(".csv")]
    counts, cands, plays = analyze_game(tf, gid)
    for k in total:
        total[k] += counts[k]
    all_cands.extend(cands)
    plays_by_game[gid] = plays
    print(f"[{gid} {game_label.get(gid,'?'):<10}] man plays={counts['man_plays']:>2} "
          f"WR routes={counts['wr_routes']:>3} valid cuts={counts['cuts']:>2} "
          f"candidates={len(cands):>2}")

pos = [c for c in all_cands if c["sep_gain"] > 0]
print("\n==================== COMBINED TOTALS ====================")
print(f"Games analyzed              : {len(track_files)}")
print(f"Man-coverage passing plays  : {total['man_plays']}")
print(f"WR routes                   : {total['wr_routes']}")
print(f"Valid cuts                  : {total['cuts']}")
print(f"Scored candidates           : {len(all_cands)}")
print(f"Positive-separation (>0)    : {len(pos)}")

# ---- rank: reliable matchup (high/medium) + positive separation first -------
conf_rank = {"high": 0, "medium": 1, "low": 2, "none": 3}
ranked = sorted(all_cands, key=lambda c: (
    not (c["sep_gain"] > 0),          # positive gain first
    conf_rank.get(c["conf"], 3),       # reliable matchups first
    -c["score"],                       # then by score
))

print("\n============== TOP CANDIDATES (reliable match + positive sep first) ==============")
for rank, c in enumerate(ranked[:8], 1):
    print(f"\n#{rank}  game {c['gid']} play {c['pid']}  score={c['score']:.2f}  "
          f"conf={c['conf'].upper()}  sep_gain={c['sep_gain']:+.2f}")
    print(f"    WR {name_of(c['wr'])} vs DEF {name_of(c['def'])} ({pos_of(c['def'])})"
          f"  | {c['coverage']}")
    print(f"    cut t+{(c['cut_frame']-c['snap'])/HZ:.1f}s turn {c['turn']:+.0f}deg"
          f" @ {c['wr_speed']:.1f}yd/s | sep {c['sep_before']:.2f}->+0.5s {c['sep_after']:.2f}")
    print(f"    defender lag {c['react_desc']} (consistent-with, not proof) | "
          f"mirror={c['mirror']:+.2f} prox@cut={c['prox_at_cut']:.1f}yd"
          f"{'  ['+c['reasons']+']' if c['reasons'] else ''}")

# ---- SVGs for the 3 strongest VALIDATED candidates --------------------------
validated = [c for c in ranked if c["sep_gain"] > 0 and c["conf"] in ("high", "medium")]
to_plot = validated[:3]
print("\n----------------------------------------------------------------")
if not validated:
    print("No VALIDATED candidates (positive separation + reliable matchup) "
          "in the current sample. Not generating 'strongest example' SVGs to "
          "avoid implying a deception that the data does not support.")
    if ranked:
        print("For transparency, plotting the single top-ranked candidate with "
              "its (unvalidated) confidence label.")
        to_plot = ranked[:1]
else:
    print(f"Validated candidates available: {len(validated)}. Plotting top "
          f"{len(to_plot)}.")


def svg_candidate(c, path):
    gid, pid = c["gid"], c["pid"]
    pdata = plays_by_game[gid][pid]
    sm_wr = smooth_track(pdata[c["wr"]]); sm_df = smooth_track(pdata[c["def"]])
    wr_by = {r["frameId"]: r for r in sm_wr}; df_by = {r["frameId"]: r for r in sm_df}
    frames = [f for f in range(c["snap"], c["thrown"] + 1) if f in wr_by and f in df_by]
    xs = [wr_by[f]["xs"] for f in frames] + [df_by[f]["xs"] for f in frames]
    ys = [wr_by[f]["ys"] for f in frames] + [df_by[f]["ys"] for f in frames]
    pad = 3; minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    W, H = 960, 480
    def X(x): return (x - minx) / (maxx - minx) * (W - 90) + 60
    def Y(y): return H - 60 - (y - miny) / (maxy - miny) * (H - 110)
    def poly(pts, col): return ('<polyline points="'
        + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        + f'" fill="none" stroke="{col}" stroke-width="2.5"/>')
    wr_pts = [(wr_by[f]["xs"], wr_by[f]["ys"]) for f in frames]
    df_pts = [(df_by[f]["xs"], df_by[f]["ys"]) for f in frames]
    cf = c["cut_frame"]
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#f6fbf3"/>']
    gx = int(minx // 5 * 5)
    while gx <= maxx:
        if minx <= gx <= maxx:
            e.append(f'<line x1="{X(gx):.1f}" y1="48" x2="{X(gx):.1f}" y2="{H-60}" stroke="#d6e8cf"/>')
        gx += 5
    e.append(poly(wr_pts, "#C0392b")); e.append(poly(df_pts, "#2471a3"))
    e.append(f'<circle cx="{X(wr_pts[0][0]):.1f}" cy="{Y(wr_pts[0][1]):.1f}" r="6" fill="white" stroke="#C0392b" stroke-width="2.5"/>')
    e.append(f'<circle cx="{X(df_pts[0][0]):.1f}" cy="{Y(df_pts[0][1]):.1f}" r="6" fill="white" stroke="#2471a3" stroke-width="2.5"/>')
    if cf in wr_by:
        cx, cy = wr_by[cf]["xs"], wr_by[cf]["ys"]
        e.append(f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="9" fill="none" stroke="#e67e22" stroke-width="3"/>')
        e.append(f'<text x="{X(cx)+12:.1f}" y="{Y(cy)-8:.1f}" font-size="12" font-weight="bold" fill="#e67e22">CUT ({c["turn"]:+.0f} deg)</text>')
        if cf in df_by:
            e.append(f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{X(df_by[cf]["xs"]):.1f}" y2="{Y(df_by[cf]["ys"]):.1f}" stroke="#999" stroke-dasharray="3,3"/>')
    e.append(f'<circle cx="{X(wr_pts[-1][0]):.1f}" cy="{Y(wr_pts[-1][1]):.1f}" r="5" fill="#C0392b"/>')
    e.append(f'<circle cx="{X(df_pts[-1][0]):.1f}" cy="{Y(df_pts[-1][1]):.1f}" r="5" fill="#2471a3"/>')
    e.append(f'<text x="20" y="24" font-size="15" font-weight="bold" fill="#222">Route Lie Detector — {game_label.get(gid,"?")} play {pid}</text>')
    e.append(f'<text x="20" y="44" font-size="12" fill="#444">{name_of(c["wr"])} (WR) vs {name_of(c["def"])} ({pos_of(c["def"])}) — {c["coverage"]} | match: {c["conf"].upper()}</text>')
    e.append(f'<text x="20" y="{H-20}" font-size="12" fill="#555">sep {c["sep_before"]:.2f}->+0.5s {c["sep_after"]:.2f}yd (gain {c["sep_gain"]:+.2f}); def lag {c["react_desc"]} (consistent-with, not proof)</text>')
    open(path, "w").write("\n".join(e))


written = []
for n, c in enumerate(to_plot, 1):
    out = f"{BASE}/stage4_candidate{n}_{c['gid']}_play{c['pid']}.svg"
    svg_candidate(c, out)
    written.append(out)
    print(f"  wrote {os.path.basename(out)}  "
          f"(conf={c['conf']}, sep_gain={c['sep_gain']:+.2f})")

print("\nDone.")
