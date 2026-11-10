# The Route Lie Detector

**Insight:** Using NFL player-tracking data, we automatically surface *candidate deception moments* — plays in **man coverage** where a wide receiver makes a sharp directional cut and then gains separation from the defender assigned to him, a "watch this" moment for broadcasters and coaches.

**Methodology:** For every man-coverage passing play (verified via `plays.csv` `pff_passCoverageType == "Man"`) across 7 Week-1 games, we smooth each receiver's tracking path, compute travel heading from velocity (wraparound-safe), and flag sharp speed-gated cuts between the snap and the throw. Each cut's receiver is matched to the most plausible covering defender by proximity **and** movement mirroring over a pre-cut window (with an explicit high/medium/low confidence flag), and we measure the defender's reaction delay plus the change in separation from just before the cut to 0.5s after.

**Assumptions:** We treat `pff_role` (`Pass Route` / `Coverage`) and the coverage labels as ground truth, infer the receiver↔defender assignment from movement geometry (the data has no explicit assignment), and use 10 Hz tracking with light 3-frame smoothing.

**Limitations:** This is a **candidate finder, not proof** — a defender's movement near a cut is *consistent with* reacting, **not** evidence he was deceived or beaten; positive separation alone does not prove a "win"; the sample is only 7 games (~353 man-coverage routes, ~21 valid cuts); and defender matching is heuristic, so low-confidence matchups are labeled as such and excluded from the "validated" set.

**Strongest validated example:** JAX @ HOU, play 1404 — WR **Nico Collins** (HOU) vs CB **C.J. Henderson** (JAX), Cover-1 man: a **+99° cut** at t+4.0s, defender heading change **0.2s** later, separation **1.0 → 1.7 yd** (+0.7). All figures computed directly from the tracking coordinates.

---

## Repository contents

| File | Purpose |
|---|---|
| `route_lie_detector_animation.html` | **The final animated replay** (open this). |
| `route_lie_detector_animation.svg`  | The standalone animated SVG (embedded by the HTML). |
| `build_animation.py`     | Builds the animation from verified tracking data. |
| `stage4_multigame.py`    | Multi-game man-coverage detector (auto-discovers all `tracking_*.csv`). |
| `stage3_man_coverage.py` | Single-game man-coverage detector + confidence-flagged matching. |
| `stage2_deception.py`    | Cut detection + defender matching + deception ranking (all coverages). |
| `route_lie_detector.py`  | Stage-1: verify we can track one WR + defender through one play. |
| `prototype_stage.py` / `diag_turns.py` | Early data-verification / threshold-calibration diagnostics. |
| `stage4_candidate*.svg` / `stage3_man_best_*.svg` / `play97_*.svg` | Static charts for the top candidates. |
| `games.csv`, `players.csv`, `plays.csv`, `pffScoutingData.csv` | Reference data. |
| `tracking_<gameId>.csv` | Per-game player tracking (10 Hz x/y/speed/dir). |
| `requirements.txt` | Dependencies (none beyond Python 3.8+ stdlib). |

## How to view the final animation

1. **Download/clone the repo** and open **`route_lie_detector_animation.html`** in any modern web browser (Chrome, Firefox, Safari, Edge) by double-clicking it.
2. Use the **Play / Pause / Replay** buttons (top-right of the graphic) to control the replay.
3. `route_lie_detector_animation.svg` can also be opened directly in a browser if you prefer the standalone graphic.

> No server or installation is required — the animation is self-contained (SMIL + inline JavaScript).

## How to reproduce the analysis

```bash
# from the repo folder, with the tracking_*.csv files present:
python3 stage4_multigame.py      # prints counts + ranked candidates, writes candidate SVGs
python3 build_animation.py       # regenerates the animated SVG for the top validated candidate
```

More `tracking_<gameId>.csv` files (same format) can be dropped into the folder;
`stage4_multigame.py` picks them up automatically and keeps thresholds constant.
