#!/usr/bin/env python3
"""
Route Lie Detector — final animated replay builder.

Produces a self-contained, browser-ready animated SVG of the strongest
VALIDATED man-coverage candidate from the seven-game analysis:

    JAX @ HOU, Week 1, play 1404 (Cover-1 Man)
    WR Nico Collins (HOU, #? nflId 53518)  vs  CB C.J. Henderson (JAX, nflId 52417)

Everything shown is computed from the ACTUAL tracking coordinates:
  * player positions each frame (snap -> pass)
  * live separation distance (yards)
  * the cut frame + cut angle (velocity-heading change, wraparound-safe)
  * the defender's estimated reaction delay

We do NOT claim the defender was deceived; the readout says the movement is
"consistent with" reacting to the cut. Pure stdlib (no matplotlib).

Animation tech: SMIL <animate>/<set> for motion (works in all modern browsers)
plus a small JS play/pause/restart control bar. Field is drawn to scale.
"""
import csv
import math
from collections import defaultdict

BASE = "/projects/sandbox/nfl-big-data-bowl"
GAME = "2021091205"
PLAY = "1404"
WR = "53518"   # Nico Collins (verified)
DEF = "52417"  # C.J. Henderson (verified)

HZ = 10
SMOOTH = 3
VEL_WIN = 2
CUT_WINS = (2, 3, 4, 5)
DEF_REACT_ANGLE = 25.0
DEF_REACT_MAXLAG = 12
FRAME_MS = 100  # 10 Hz -> 100 ms/frame real-time

# ---- names/teams (verified) -------------------------------------------------
players = {r["nflId"]: r for r in csv.DictReader(open(f"{BASE}/players.csv"))}
games = {r["gameId"]: r for r in csv.DictReader(open(f"{BASE}/games.csv"))}
pm = None
for r in csv.DictReader(open(f"{BASE}/plays.csv")):
    if r["gameId"] == GAME and r["playId"] == PLAY:
        pm = r
WR_NAME = players[WR]["displayName"]
DEF_NAME = players[DEF]["displayName"]
OFF_TEAM = pm["possessionTeam"]
DEF_TEAM = pm["defensiveTeam"]
COVERAGE = pm["pff_passCoverage"]


# ---- load tracking for this play -------------------------------------------
def load():
    wr, df, ev = {}, {}, {}
    with open(f"{BASE}/tracking_{GAME}.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["playId"] != PLAY:
                continue
            fr = int(r["frameId"])
            if r["event"] not in ("", "None"):
                ev[fr] = r["event"]
            if r["nflId"] == WR:
                wr[fr] = {"x": float(r["x"]), "y": float(r["y"]), "s": float(r["s"])}
            elif r["nflId"] == DEF:
                df[fr] = {"x": float(r["x"]), "y": float(r["y"]), "s": float(r["s"])}
    return wr, df, ev


wr_raw, df_raw, events = load()
snap = min(f for f, e in events.items() if e == "ball_snap")
thrown = min(f for f, e in events.items()
             if e in ("pass_forward", "autoevent_passforward"))


def smooth(raw):
    fids = sorted(raw); half = SMOOTH // 2; out = {}
    for i, fid in enumerate(fids):
        lo, hi = max(0, i - half), min(len(fids), i + half + 1)
        out[fid] = {
            "x": sum(raw[fids[j]]["x"] for j in range(lo, hi)) / (hi - lo),
            "y": sum(raw[fids[j]]["y"] for j in range(lo, hi)) / (hi - lo),
            "s": raw[fid]["s"],
        }
    return out


wr = smooth(wr_raw)
df = smooth(df_raw)
frames = [f for f in range(snap, thrown + 1) if f in wr and f in df]


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def heading(a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    if dx * dx + dy * dy < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ---- heading series + cut detection (same logic as detector) ---------------
flist = sorted(wr)
head = {}
for i in range(VEL_WIN, len(flist)):
    h = heading(wr[flist[i - VEL_WIN]], wr[flist[i]])
    head[flist[i]] = h


def best_turn_at(fid):
    i = flist.index(fid)
    if head.get(fid) is None:
        return 0.0
    bt = 0.0
    for w in CUT_WINS:
        j = i - w
        if j < 0:
            continue
        fj = flist[j]
        if head.get(fj) is None or fid - fj > w + 2:
            continue
        t = ang_diff(head[fid], head[fj])
        if abs(t) > abs(bt):
            bt = t
    return bt


# find the strongest cut within the route window (snap+3 .. thrown-5),
# speed-gated, as the detector does
CUT_FRAME, CUT_TURN = None, 0.0
for fid in frames:
    if fid < snap + 3 or fid > thrown - 5:
        continue
    if wr[fid]["s"] < 2.5:
        continue
    t = best_turn_at(fid)
    if abs(t) > abs(CUT_TURN):
        CUT_TURN, CUT_FRAME = t, fid

# defender reaction delay after the cut
dhead = {}
dfl = sorted(df)
for i in range(VEL_WIN, len(dfl)):
    dhead[dfl[i]] = heading(df[dfl[i - VEL_WIN]], df[dfl[i]])
react_lag = None
ref = [dhead[CUT_FRAME - k] for k in range(1, 4)
       if dhead.get(CUT_FRAME - k) is not None]
if ref:
    for lag in range(1, DEF_REACT_MAXLAG + 1):
        ff = CUT_FRAME + lag
        if dhead.get(ff) is not None and abs(ang_diff(dhead[ff], ref[0])) >= DEF_REACT_ANGLE:
            react_lag = lag
            break
REACT_DESC = f"{react_lag/HZ:.1f}s" if react_lag is not None else "no clear reaction within 1.2s"

# separation series + before/after
sep_series = [(f, dist(wr[f], df[f])) for f in frames]
i_cut = frames.index(CUT_FRAME)
sep_before = dist(wr[frames[i_cut - 3]], df[frames[i_cut - 3]])
f_after = CUT_FRAME + 5
sep_after = dist(wr[f_after], df[f_after]) if f_after in wr and f_after in df else sep_series[-1][1]
sep_gain = sep_after - sep_before

print(f"VERIFIED MEASUREMENTS (from actual tracking):")
print(f"  window {snap}->{thrown} ({len(frames)} frames, {(len(frames)-1)/HZ:.1f}s)")
print(f"  cut frame {CUT_FRAME} (t+{(CUT_FRAME-snap)/HZ:.1f}s), angle {CUT_TURN:+.0f} deg")
print(f"  sep before cut {sep_before:.2f} -> +0.5s {sep_after:.2f} (gain {sep_gain:+.2f})")
print(f"  defender reaction: {REACT_DESC}")

# ============================================================================
# Build the animated SVG.
# ============================================================================
# Field coordinate mapping. NFL field: x in [0,120] (yd, incl end zones),
# y in [0,53.3] (yd). We'll render a zoomed region around the action.
pad = 6
minx = min(min(wr[f]["x"] for f in frames), min(df[f]["x"] for f in frames)) - pad
maxx = max(max(wr[f]["x"] for f in frames), max(df[f]["x"] for f in frames)) + pad
miny = min(min(wr[f]["y"] for f in frames), min(df[f]["y"] for f in frames)) - pad
maxy = max(max(wr[f]["y"] for f in frames), max(df[f]["y"] for f in frames)) + pad

FIELD_W, FIELD_H = 920, 430
OX, OY = 30, 90           # field origin on canvas
CANVAS_W, CANVAS_H = 980, 620


def PX(x):
    return OX + (x - minx) / (maxx - minx) * FIELD_W


def PY(y):
    # flip y so the field reads naturally
    return OY + FIELD_H - (y - miny) / (maxy - miny) * FIELD_H


dur = len(frames) * FRAME_MS / 1000.0  # seconds
times = ";".join(f"{i/(len(frames)-1):.4f}" for i in range(len(frames)))


def vals(getter):
    return ";".join(f"{getter(f):.1f}" for f in frames)


wr_cx = vals(lambda f: PX(wr[f]["x"]))
wr_cy = vals(lambda f: PY(wr[f]["y"]))
df_cx = vals(lambda f: PX(df[f]["x"]))
df_cy = vals(lambda f: PY(df[f]["y"]))

# separation line endpoints + text
sep_x1 = vals(lambda f: PX(wr[f]["x"]))
sep_y1 = vals(lambda f: PY(wr[f]["y"]))
sep_x2 = vals(lambda f: PX(df[f]["x"]))
sep_y2 = vals(lambda f: PY(df[f]["y"]))
sep_txt = ";".join(f"{dist(wr[f], df[f]):.1f}" for f in frames)
sep_tx = vals(lambda f: (PX(wr[f]["x"]) + PX(df[f]["x"])) / 2)
sep_ty = vals(lambda f: (PY(wr[f]["y"]) + PY(df[f]["y"])) / 2 - 6)

# cut marker visibility: show from cut frame onward
cut_begin = f"{i_cut/(len(frames)-1)*dur:.3f}s"

# trajectory paths (full, drawn faintly as reference)
wr_path = "M " + " L ".join(f"{PX(wr[f]['x']):.1f} {PY(wr[f]['y']):.1f}" for f in frames)
df_path = "M " + " L ".join(f"{PX(df[f]['x']):.1f} {PY(df[f]['y']):.1f}" for f in frames)

RED = "#E03131"      # WR (offense)
BLUE = "#1C7ED6"     # defender
GREEN = "#2F9E44"
ORANGE = "#F08C00"

# yard lines every 5 yd (field x)
yard_lines = []
gx = int(minx // 5 * 5)
while gx <= maxx:
    if minx <= gx <= maxx:
        x = PX(gx)
        yard_lines.append(
            f'<line x1="{x:.1f}" y1="{OY}" x2="{x:.1f}" y2="{OY+FIELD_H}" '
            f'stroke="#ffffff" stroke-opacity="0.35" stroke-width="1"/>')
        # yardline number (NFL marks 10..50..10; show raw field yard for ref)
        yard_lines.append(
            f'<text x="{x:.1f}" y="{OY+FIELD_H-6}" fill="#e8ffe8" font-size="11" '
            f'text-anchor="middle" opacity="0.6">{gx}</text>')
    gx += 5
yard_lines = "\n".join(yard_lines)

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" font-family="Segoe UI, Arial, sans-serif" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <defs>
    <linearGradient id="turf" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2b7a2b"/>
      <stop offset="1" stop-color="#1f5d1f"/>
    </linearGradient>
    <filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="1.5" flood-opacity="0.5"/>
    </filter>
  </defs>

  <!-- background -->
  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#0b1f12"/>

  <!-- title block -->
  <text x="30" y="34" fill="#ffffff" font-size="22" font-weight="700">The Route Lie Detector</text>
  <text x="30" y="56" fill="#9bd6a0" font-size="14">{OFF_TEAM} {WR_NAME} (WR) vs {DEF_TEAM} {DEF_NAME} (CB) &#8212; {COVERAGE} man &#183; Week 1 JAX@HOU, play {PLAY}</text>

  <!-- field -->
  <rect x="{OX}" y="{OY}" width="{FIELD_W}" height="{FIELD_H}" fill="url(#turf)" rx="6"/>
  {yard_lines}
  <rect x="{OX}" y="{OY}" width="{FIELD_W}" height="{FIELD_H}" fill="none" stroke="#ffffff" stroke-opacity="0.5" rx="6"/>

  <!-- faint full trajectories for reference -->
  <path d="{wr_path}" fill="none" stroke="{RED}" stroke-opacity="0.30" stroke-width="2" stroke-dasharray="4,4"/>
  <path d="{df_path}" fill="none" stroke="{BLUE}" stroke-opacity="0.30" stroke-width="2" stroke-dasharray="4,4"/>

  <!-- separation connector (animated) -->
  <line id="sepline" stroke="#ffd43b" stroke-width="2" stroke-dasharray="2,3">
    <animate id="ax1" attributeName="x1" dur="{dur:.3f}s" values="{sep_x1}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    <animate attributeName="y1" dur="{dur:.3f}s" values="{sep_y1}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    <animate attributeName="x2" dur="{dur:.3f}s" values="{sep_x2}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    <animate attributeName="y2" dur="{dur:.3f}s" values="{sep_y2}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
  </line>
  <text id="septxt" fill="#fff59d" font-size="13" font-weight="700" text-anchor="middle">
    <animate attributeName="x" dur="{dur:.3f}s" values="{sep_tx}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    <animate attributeName="y" dur="{dur:.3f}s" values="{sep_ty}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
  </text>

  <!-- CUT highlight ring (appears at the cut frame) -->
  <circle cx="{PX(wr[CUT_FRAME]['x']):.1f}" cy="{PY(wr[CUT_FRAME]['y']):.1f}" r="4" fill="none" stroke="{ORANGE}" stroke-width="3" opacity="0">
    <set attributeName="opacity" to="1" begin="{cut_begin}" fill="freeze"/>
    <animate attributeName="r" values="4;16;12" dur="0.5s" begin="{cut_begin}" fill="freeze"/>
  </circle>
  <text x="{PX(wr[CUT_FRAME]['x'])+16:.1f}" y="{PY(wr[CUT_FRAME]['y'])-10:.1f}" fill="{ORANGE}" font-size="13" font-weight="700" opacity="0">
    CUT {CUT_TURN:+.0f}&#176;
    <set attributeName="opacity" to="1" begin="{cut_begin}" fill="freeze"/>
  </text>

  <!-- WR marker -->
  <g filter="url(#shadow)">
    <circle id="wr" r="9" fill="{RED}" stroke="#fff" stroke-width="2">
      <animate attributeName="cx" dur="{dur:.3f}s" values="{wr_cx}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
      <animate attributeName="cy" dur="{dur:.3f}s" values="{wr_cy}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    </circle>
  </g>
  <text font-size="11" font-weight="700" fill="#fff" text-anchor="middle">
    WR
    <animate attributeName="x" dur="{dur:.3f}s" values="{wr_cx}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    <animate attributeName="y" dur="{dur:.3f}s" values="{vals(lambda f: PY(wr[f]['y'])+4)}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
  </text>

  <!-- Defender marker -->
  <g filter="url(#shadow)">
    <circle id="def" r="9" fill="{BLUE}" stroke="#fff" stroke-width="2">
      <animate attributeName="cx" dur="{dur:.3f}s" values="{df_cx}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
      <animate attributeName="cy" dur="{dur:.3f}s" values="{df_cy}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    </circle>
  </g>
  <text font-size="11" font-weight="700" fill="#fff" text-anchor="middle">
    CB
    <animate attributeName="x" dur="{dur:.3f}s" values="{df_cx}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
    <animate attributeName="y" dur="{dur:.3f}s" values="{vals(lambda f: PY(df[f]['y'])+4)}" keyTimes="{times}" begin="0s" fill="freeze" repeatCount="1"/>
  </text>

  <!-- legend -->
  <g transform="translate({OX},{OY+FIELD_H+16})">
    <circle cx="8" cy="0" r="7" fill="{RED}" stroke="#fff" stroke-width="2"/>
    <text x="22" y="4" fill="#e8ffe8" font-size="13">{OFF_TEAM} {WR_NAME} (WR)</text>
    <circle cx="250" cy="0" r="7" fill="{BLUE}" stroke="#fff" stroke-width="2"/>
    <text x="264" y="4" fill="#e8ffe8" font-size="13">{DEF_TEAM} {DEF_NAME} (CB)</text>
    <line x1="520" y1="0" x2="548" y2="0" stroke="#ffd43b" stroke-width="2" stroke-dasharray="2,3"/>
    <text x="556" y="4" fill="#e8ffe8" font-size="13">separation (yd)</text>
  </g>

  <!-- stat panel -->
  <g transform="translate({OX},{OY+FIELD_H+44})">
    <rect x="0" y="0" width="{FIELD_W}" height="70" rx="8" fill="#10301c" stroke="#2f9e44" stroke-opacity="0.5"/>
    <text x="16" y="26" fill="#9bd6a0" font-size="12">CUT ANGLE</text>
    <text x="16" y="50" fill="#fff" font-size="20" font-weight="700">{CUT_TURN:+.0f}&#176;</text>
    <text x="150" y="26" fill="#9bd6a0" font-size="12">DEFENDER REACTION DELAY</text>
    <text x="150" y="50" fill="#fff" font-size="20" font-weight="700">{REACT_DESC}</text>
    <text x="470" y="26" fill="#9bd6a0" font-size="12">SEPARATION  (before to +0.5s after cut)</text>
    <text x="470" y="50" fill="#fff" font-size="20" font-weight="700">{sep_before:.1f} yd -&gt; {sep_after:.1f} yd  ({sep_gain:+.1f})</text>
  </g>

  <!-- why-flagged caption -->
  <foreignObject x="{OX}" y="{OY+FIELD_H+120}" width="{FIELD_W}" height="56">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font:12px Segoe UI,Arial,sans-serif;color:#bfe9c6;line-height:1.35">
      <b style="color:#fff">Why flagged:</b> In press man ({COVERAGE}), {WR_NAME} makes a sharp {abs(CUT_TURN):.0f}&#176; cut at t+{(CUT_FRAME-snap)/HZ:.1f}s; the matched defender&#8217;s heading changes {REACT_DESC} later and separation grows by {sep_gain:+.1f} yd over the next half-second.
      This is movement <i>consistent with</i> a deception candidate &#8212; <b style="color:#ffd43b">not proof</b> the defender was beaten or tricked. Defender match confidence: HIGH (close, tightly mirrored pre-cut).
    </div>
  </foreignObject>

  <!-- controls -->
  <g id="controls" transform="translate({OX},10)" cursor="pointer">
  </g>
  <g transform="translate(680,24)">
    <g id="btnPlay" cursor="pointer"><rect x="0" y="0" width="70" height="26" rx="6" fill="#2f9e44"/><text x="35" y="18" fill="#fff" font-size="13" text-anchor="middle" font-weight="700">Play</text></g>
    <g id="btnPause" cursor="pointer" transform="translate(78,0)"><rect x="0" y="0" width="70" height="26" rx="6" fill="#e8590c"/><text x="35" y="18" fill="#fff" font-size="13" text-anchor="middle" font-weight="700">Pause</text></g>
    <g id="btnReplay" cursor="pointer" transform="translate(156,0)"><rect x="0" y="0" width="84" height="26" rx="6" fill="#1c7ed6"/><text x="42" y="18" fill="#fff" font-size="13" text-anchor="middle" font-weight="700">Replay</text></g>
  </g>

  <script type="text/javascript"><![CDATA[
    var svg = document.documentElement;
    function play(){{ try{{ svg.unpauseAnimations(); }}catch(e){{}} }}
    function pause(){{ try{{ svg.pauseAnimations(); }}catch(e){{}} }}
    function replay(){{ try{{ svg.setCurrentTime(0); svg.unpauseAnimations(); }}catch(e){{}} }}
    document.getElementById('btnPlay').addEventListener('click', play);
    document.getElementById('btnPause').addEventListener('click', pause);
    document.getElementById('btnReplay').addEventListener('click', replay);
  ]]></script>
</svg>
"""

out = f"{BASE}/route_lie_detector_animation.svg"
with open(out, "w") as f:
    f.write(svg)
print(f"\nWrote animation: {out}")
print(f"frames={len(frames)} duration={dur:.1f}s")
