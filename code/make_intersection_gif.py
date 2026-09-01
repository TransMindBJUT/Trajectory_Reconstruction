# -*- coding: utf-8 -*-
"""
GitHub homepage GIF — two interaction examples from the paper (Fig11):
  (a) Intersection (trajectory-crossing) interaction
  (b) Proximity (closest_only) interaction
Layout (identical for both):
  TOP    : ±5 s trajectories around the interaction — the two involved vehicles (ego
           blue / target orange) drive as oriented rectangles, pausing at the flag=2 (ISMD)
           point and at the crossing / closest-approach point. Other vehicles genuinely
           co-present in the scene (window extended by ±3 s before/after) also drive as
           light-grey rectangles. The road-density background and full trajectory are
           removed.
  BOTTOM : 4 metric panels (speed / ax / ay / jerk), all 5 models overlaid in each,
           synced to the drive, with comfort (green dashed) / extreme (red dash-dot)
           thresholds. Proposed (ours) is bold; the other models are faded.
Each scenario is rendered to its own GIF.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from PIL import Image

BASE = r"F:\Rrajectory_Reconstruction\raw data preprocessing\5_all_data"
OUT = os.path.join(BASE, "github_demo")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_cache_both.npz")
DENSITY = os.path.join(OUT, "_density.npz")

# intersection RANGE boundary (straight road edges + trajectory-generated fillet
# arcs) built by _range_boundary.py, stored in the LOCAL frame anchored at fp.
RANGE_NPZ = r"F:\Rrajectory_Reconstruction\project_page\code\_range.npz"

RAW_FILE = os.path.join(BASE, "5_Data_vehicle_need_spatiotemporal_4metrics_30m.csv")

EARTH_RADIUS_M = 6378137.0
FUTURE_S = 5.0
WIN_S = 5.0
EXT_S = 3.0
FPS = 8

# intersection road-boundary edges (lon/lat) — from the paper's fig10/fig11
EDGES = [
    np.array([[116.520172733, 39.785225236], [116.520292092, 39.78526955]]),
    np.array([[116.520459730, 39.785226266], [116.520502645, 39.785172676]]),
    np.array([[116.520521421, 39.785010877], [116.520453024, 39.784975837]]),
    np.array([[116.520238447, 39.784967593], [116.520196873, 39.785023244]]),
]

MODELS = ["RAW", "Proposed", "KCO", "LSTM", "SG"]
MODEL_FILES = {
    "Proposed": os.path.join(BASE, "our_model.csv"),
    "KCO":      os.path.join(BASE, "Zhao_model.csv"),
    "LSTM":     os.path.join(BASE, "LSTM_model.csv"),
    "SG":       os.path.join(BASE, "4metrics_4model", "5_Data_vehicle_features_SG.csv"),
}
MODEL_COLOR = {"RAW": "#555555", "Proposed": "#E53935", "KCO": "#089099",
               "LSTM": "#1F77B4", "SG": "#E28E2C"}
MODEL_LABEL = {"RAW": "RAW", "Proposed": "Proposed", "KCO": "KCO",
               "LSTM": "LSTM", "SG": "SG"}

# map vehicle colors (RGB) — ordinary / target / ego, all with black borders
C_OTHER = "#AEB5C0"   # (174, 181, 192)
C_TGT = "#FB5607"     # (251, 86, 7)
C_EGO = "#3A86FF"     # (58, 134, 255)

METRICS = ["speed", "ax", "ay", "jerk"]
METRIC_TITLE = {"speed": "Speed (m/s)", "ax": r"$a_x$ (m/s$^2$)",
                "ay": r"$a_y$ (m/s$^2$)", "jerk": r"Jerk (m/s$^3$)"}
COMFORT = {"speed": (0, 23), "ax": (-5, 5), "ay": (-1.764, 1.764), "jerk": (-10, 10)}
EXTREME = {"speed": (0, 41.67), "ax": (-10.6, 6.87), "ay": (-2.5, 2.5), "jerk": (-23, 23)}

# Bottom-panel layout:
#   "metrics" -> 4 metric panels (speed / ax / ay / jerk), all 5 models overlaid,
#                Proposed bold. (kept as the original version)
#   "models"  -> 5 model boxes (RAW / Proposed / KCO / LSTM / SG), each box showing
#                a single BOX_METRIC curve for that model, with the comfort/extreme
#                limit lines plus an ISMD-preserved / smoothed annotation (mirroring
#                the jerk row of the original side-by-side version).
BOTTOM_MODE = "metrics"
BOX_METRIC = "jerk"

RAW_COLS = ["uuid", "targetsOrgTimes", "X", "Y", "longitude", "latitude",
            "speed", "ax", "ay", "jerk", "point_flag", "pet_seconds",
            "conflict_distance_m", "interaction_mode", "interaction_target_uuid",
            "len", "width", "heading"]
MODEL_COLS = ["uuid", "targetsOrgTimes", "speed", "ax", "ay", "jerk"]

# (a) crossing example and (b) proximity example — from fig11_interaction_preservation_two_examples.ipynb
SCENARIOS = [
    {"tag": "U5", "label": "(a) Intersection scenario I", "uuid": "3d800ae44cb9cb12b9dfae0c", "ts": 1768350004541, "boundary": True},
    {"tag": "U6", "label": "(b) Intersection scenario II", "uuid": "2ec4ef6e1d829d11ba0cde68", "ts": 1768352995298, "boundary": True},
    {"tag": "C3", "label": "(c) Proximity interaction scenario I", "uuid": "cb86891e6121e600b9cd16ac", "ts": 1768348805315, "boundary": False},
    {"tag": "C8", "label": "(d) Proximity interaction scenario II", "uuid": "25644568253f79dfb9c18f80", "ts": 1768348049611, "boundary": False},
    {"tag": "U8", "label": "(e) Intersection scenario III", "uuid": "2e3f9dd6113cde57b9e592a4", "ts": 1768350406009, "boundary": True},
]


def line_intersection(l1, l2):
    x1, y1 = l1[0]; x2, y2 = l1[1]
    x3, y3 = l2[0]; x4, y4 = l2[1]
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-12:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    return np.array([x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)])


def project_lonlat_to_xy(lon, lat, lon0, lat0, x0, y0):
    lon_r, lat_r = math.radians(lon), math.radians(lat)
    lon0_r, lat0_r = math.radians(lon0), math.radians(lat0)
    x = x0 + (lon_r - lon0_r) * math.cos(lat0_r) * EARTH_RADIUS_M
    y = y0 + (lat_r - lat0_r) * EARTH_RADIUS_M
    return x, y


def segment_intersection_pt(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    det = dx2 * dy1 - dx1 * dy2
    if abs(det) < 1e-12:
        return None
    t = (dx2 * (y3 - y1) - dy2 * (x3 - x1)) / det
    s = (dx1 * (y3 - y1) - dy1 * (x3 - x1)) / det
    if 0 <= t <= 1 and 0 <= s <= 1:
        return np.array([x1 + t * dx1, y1 + t * dy1])
    return None


def load_range_poly():
    """Intersection range boundary polygon in UTM XY (the GIF map frame), plus the
    local->UTM frame (fp, d_B, n_B) needed to draw lane markings."""
    z = np.load(RANGE_NPZ)
    fp = z["fp"]; d_B = z["d_B"]; n_B = z["n_B"]
    x = z["poly_x"]; y = z["poly_y"]
    utm_x = fp[0] + x * d_B[0] + y * n_B[0]
    utm_y = fp[1] + x * d_B[1] + y * n_B[1]
    return fp, d_B, n_B, np.column_stack([utm_x, utm_y])


def draw_lane_cross_section(ax, fp, d_B, n_B):
    """Reference-style lane-level cross-section with rounded corners: each road is a
    stack of colored lane strips — driving (grey), biking (light blue, 非机动车道) —
    matching the paper's lane-level digital-twin basemap. The 4 intersection corners
    are circular curb fillets (radius fitted from the real turning data), and a
    double-yellow center line separates opposing traffic, ending at the white stop
    line (停车线) placed at the MEASURED stopping position per approach (the median
    min-speed point from the full trajectory data, not the geometric tangent point).
    Local frame: +x = road B (horizontal), +y = road A (vertical)."""
    B_center, A_center = 2.24, 12.665
    HALF = 3.6                       # half driving-road width (1 lane per direction)
    ARM = 22.0
    R_DRIVE = 12.0                   # corner curb fillet radius (m), from turning data
    W_BIKE = 2.4                     # biking (非机动车道) width, m
    RAW = {"driving": "#d9d9d9", "biking": "#d7e8ff"}
    EDGE_RAW = "#a5a5a5"
    ALPHA = 0.35

    # blend each colour toward white so the opaque painter's-algorithm fills reproduce
    # the reference's alpha=0.35 look without cross-band blending.
    def blend(hexc):
        r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
        k = ALPHA
        return (r * k + 255 * (1 - k)) / 255, (g * k + 255 * (1 - k)) / 255, (b * k + 255 * (1 - k)) / 255

    COLORS = {k: blend(v) for k, v in RAW.items()}
    EDGE = blend(EDGE_RAW)

    def utm(x, y):
        return fp[0] + x * d_B[0] + y * n_B[0], fp[1] + x * d_B[1] + y * n_B[1]

    def rounded_plus(w, R):
        """Vertices (UTM XY) of a cross: central box of half-width `w` with the 4
        corners filleted by radius `R`, and 4 arms extending to `ARM` from center."""
        x_lo, x_hi = A_center - w, A_center + w
        y_lo, y_hi = B_center - w, B_center + w
        xa, xb = A_center - ARM, A_center + ARM
        ya, yb = B_center - ARM, B_center + ARM
        n = 20
        def arc(cx0, cy0, t0, t1):
            return [(cx0 + R * math.cos(t0 + (t1 - t0) * i / n),
                     cy0 + R * math.sin(t0 + (t1 - t0) * i / n)) for i in range(n + 1)]
        pts = []
        pts += arc(x_lo - R, y_hi + R, 0.0, -math.pi / 2)          # NW fillet
        pts += [(x_lo - R, y_hi), (xa, y_hi), (xa, y_lo), (x_lo - R, y_lo)]
        pts += arc(x_lo - R, y_lo - R, math.pi / 2, 0.0)           # SW fillet
        pts += [(x_lo, y_lo - R), (x_lo, ya), (x_hi, ya), (x_hi, y_lo - R)]
        pts += arc(x_hi + R, y_lo - R, math.pi, math.pi / 2)       # SE fillet
        pts += [(x_hi + R, y_lo), (xb, y_lo), (xb, y_hi), (x_hi + R, y_hi)]
        pts += arc(x_hi + R, y_hi + R, 3 * math.pi / 2, math.pi)   # NE fillet
        pts += [(x_hi, y_hi + R), (x_hi, yb), (x_lo, yb), (x_lo, y_hi + R)]
        return [utm(x, y) for x, y in pts]

    # outermost band first, innermost last: each concentric band is a clean
    # non-overlapping ring and the driving lanes sit on top.
    for w, R, typ, z in (
        (HALF + W_BIKE, R_DRIVE + W_BIKE, "biking", 2),
        (HALF, R_DRIVE, "driving", 3),
    ):
        ax.add_patch(Polygon(rounded_plus(w, R), closed=True, facecolor=COLORS[typ],
                             edgecolor=EDGE, alpha=1.0, linewidth=0.45, zorder=z))

    # driving-lane box edges.
    x_lo, x_hi = A_center - HALF, A_center + HALF
    y_lo, y_hi = B_center - HALF, B_center + HALF
    xa, xb = A_center - ARM, A_center + ARM
    ya, yb = B_center - ARM, B_center + ARM

    # white stop lines (停车线) at the MEASURED stopping position per approach
    # (median min-speed point from the full data), instead of the geometric tangent
    # point (R_DRIVE + HALF = 15.6 m). Each approach stops on its entry side:
    # SW-bound on -x, NE-bound on +x, SE-bound on -y, NW-bound on +y.
    STOP = {"SW": 13.35, "NE": 13.26, "SE": 12.30, "NW": 17.47}
    sW = A_center - STOP["SW"]     # SW-bound stop-line x (SW traffic enters from -x)
    sE = A_center + STOP["NE"]     # NE-bound stop-line x (NE traffic enters from +x)
    sS = B_center - STOP["SE"]     # SE-bound stop-line y (SE traffic enters from -y)
    sN = B_center + STOP["NW"]     # NW-bound stop-line y (NW traffic enters from +y)

    def line(p0, p1, color, lw, z=5):
        (x0, y0), (x1, y1) = utm(*p0), utm(*p1)
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.9,
                zorder=z, solid_capstyle="round")

    # double-yellow center line (light yellow, double solid) separating opposing
    # directions, running from the arm end to the stop line (not into the box).
    for y0 in (B_center - 0.13, B_center + 0.13):
        line((xa, y0), (sW, y0), "#e2c14a", 2.0)
        line((sE, y0), (xb, y0), "#e2c14a", 2.0)
    for x0 in (A_center - 0.13, A_center + 0.13):
        line((x0, ya), (x0, sS), "#e2c14a", 2.0)
        line((x0, sN), (x0, yb), "#e2c14a", 2.0)

    # white stop lines (停车线) — one per approach, across that direction's FORWARD
    # DRIVING lane only (NOT extending into the biking/非机动车道 strip outside it).
    # Lane/direction from the density:  Road B: SW(+x)->lower (y<B_center),
    # NE(-x)->upper (y>B_center);  Road A: SE(+y)->upper-x (x>A_center),
    # NW(-y)->lower-x (x<A_center).
    for p0, p1 in (
        ((sW, y_lo), (sW, B_center)),               # SW-bound on Road B (ego's lane)
        ((sE, B_center), (sE, y_hi)),               # NE-bound on Road B
        ((A_center, sS), (x_hi, sS)),               # SE-bound on Road A
        ((x_lo, sN), (A_center, sN)),               # NW-bound on Road A
    ):
        (sx0, sy0), (sx1, sy1) = utm(*p0), utm(*p1)
        ax.plot([sx0, sx1], [sy0, sy1], color="#ffffff", lw=3.0, alpha=1.0,
                zorder=4, solid_capstyle="butt")


def veh_corners(x, y, heading_deg, length, width):
    """4 corners of a vehicle rectangle centered at (x, y), oriented by compass heading."""
    th = math.radians(heading_deg)
    fwd = np.array([math.sin(th), math.cos(th)])   # forward (X = east, Y = north)
    lat = np.array([math.cos(th), -math.sin(th)])  # lateral (right side)
    c = np.array([x, y])
    return np.array([
        c + 0.5 * length * fwd + 0.5 * width * lat,
        c + 0.5 * length * fwd - 0.5 * width * lat,
        c - 0.5 * length * fwd - 0.5 * width * lat,
        c - 0.5 * length * fwd + 0.5 * width * lat,
    ])


def ribbon_quads(x, y, width):
    """Tile the ribbon swept by a `width`-wide rectangle along the (x, y) centerline
    into non-overlapping quads. Returns (quads, alphas) with alphas fading from 0 at the
    tail (oldest point) to 1 at the head (newest point)."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if len(pts) < 2:
        return [], np.array([])
    d = np.diff(pts, axis=0)
    seglen = np.linalg.norm(d, axis=1)
    seglen[seglen < 1e-9] = 1e-9
    d = d / seglen[:, None]
    perp = np.column_stack([-d[:, 1], d[:, 0]])          # unit normal (right side)
    mid = (pts[:-1] + pts[1:]) / 2.0
    off = perp * (width / 2.0)
    quads = []
    for i in range(len(mid) - 1):
        quads.append(np.array([
            mid[i]     + off[i],
            mid[i + 1] + off[i + 1],
            mid[i + 1] - off[i + 1],
            mid[i]     - off[i],
        ]))
    alphas = np.linspace(0.0, 1.0, len(quads))
    return quads, alphas


def update_ribbon(coll, x, y, width, color, i0, i1):
    """Update a PolyCollection to show the traversed path x[i0:i1+1], y[i0:i1+1] as a
    fading ribbon of `color` (bright at the head, transparent at the tail)."""
    if i1 <= i0:
        coll.set_verts([])
        return
    xs = np.asarray(x, float)[i0:i1 + 1]
    ys = np.asarray(y, float)[i0:i1 + 1]
    quads, alphas = ribbon_quads(xs, ys, width)
    if not quads:
        coll.set_verts([])
        return
    base = to_rgba(color)
    rgba = np.tile(base[:3], (len(quads), 1))
    rgba = np.column_stack([rgba, alphas])
    coll.set_verts(quads)
    coll.set_facecolors(rgba)
    coll.set_edgecolors("none")


# ---------------- build or load the combined two-scenario cache ----------------
def load_scenarios():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        data = {}
        for scn in SCENARIOS:
            tag = scn["tag"]
            d = {}
            for f in ["ego_x", "ego_y", "t_rel", "tgt_x", "tgt_y", "tgt_t_rel",
                      "ego_len", "ego_width", "ego_heading", "tgt_len", "tgt_width", "tgt_heading"]:
                d[f] = z[f"{tag}_{f}"]
            d["flag2_idx"] = int(z[f"{tag}_flag2_idx"])
            d["fp_x"] = float(z[f"{tag}_fp_x"]); d["fp_y"] = float(z[f"{tag}_fp_y"])
            d["fp_jerk"] = float(z[f"{tag}_fp_jerk"])
            d["pet"] = float(z[f"{tag}_pet"]); d["dmin"] = float(z[f"{tag}_dmin"])
            d["isect"] = z[f"{tag}_isect"] if z[f"{tag}_isect"].size else None
            d["cross_ego"] = int(z[f"{tag}_cross_ego"]); d["cross_tgt"] = int(z[f"{tag}_cross_tgt"])
            d["closest_ego"] = int(z[f"{tag}_closest_ego"]); d["closest_tgt"] = int(z[f"{tag}_closest_tgt"])
            d["raw_full"] = {m: z[f"{tag}_raw_{m}"] for m in METRICS}
            d["model_full"] = {mn: {m: z[f"{tag}_mod_{mn}_{m}"] for m in METRICS}
                               for mn in MODELS if mn != "RAW"}
            d["model_full"]["RAW"] = {m: d["raw_full"][m].copy() for m in METRICS}
            if f"{tag}_n_others" in z.files:
                _n = int(z[f"{tag}_n_others"])
                d["others"] = [(z[f"{tag}_others_{i}_x"], z[f"{tag}_others_{i}_y"],
                                z[f"{tag}_others_{i}_t"], z[f"{tag}_others_{i}_h"],
                                float(z[f"{tag}_others_{i}_len"]), float(z[f"{tag}_others_{i}_wid"]))
                               for i in range(_n)]
            else:
                d["others"] = []
            data[tag] = d
        corners_xy = z["corners_xy"]
        print("cache loaded.")
        return data, corners_xy

    print("reading large CSVs (one-time) ...")
    orig = pd.read_csv(RAW_FILE, usecols=RAW_COLS, low_memory=False)
    orig["uuid"] = orig["uuid"].astype(str).str.strip()
    ref_lon0 = float(orig["longitude"].iloc[0]); ref_lat0 = float(orig["latitude"].iloc[0])
    ref_x0 = float(orig["X"].iloc[0]); ref_y0 = float(orig["Y"].iloc[0])
    corners_lonlat = [line_intersection(EDGES[0], EDGES[3]),
                      line_intersection(EDGES[0], EDGES[1]),
                      line_intersection(EDGES[1], EDGES[2]),
                      line_intersection(EDGES[2], EDGES[3])]
    corners_xy = np.array([project_lonlat_to_xy(c[0], c[1], ref_lon0, ref_lat0, ref_x0, ref_y0)
                           for c in corners_lonlat])

    data = {}
    save = {"corners_xy": corners_xy}
    for scn in SCENARIOS:
        tag, UUID, TS = scn["tag"], scn["uuid"], scn["ts"]
        g = orig[orig["uuid"] == UUID].sort_values("targetsOrgTimes").reset_index(drop=True)
        ego_x = g["X"].to_numpy(float); ego_y = g["Y"].to_numpy(float)
        ego_ts = g["targetsOrgTimes"].to_numpy(float)
        t_rel = (ego_ts - TS) / 1000.0
        flag2_idx = int(g.index[g["targetsOrgTimes"] == TS][0])
        fp_row = g.iloc[flag2_idx]
        fp_x, fp_y = float(fp_row["X"]), float(fp_row["Y"])
        fp_jerk = float(fp_row["jerk"])
        pet = float(fp_row["pet_seconds"]) if pd.notna(fp_row["pet_seconds"]) else np.nan
        dmin = float(fp_row["conflict_distance_m"]) if pd.notna(fp_row["conflict_distance_m"]) else np.nan

        tgt_uuid = str(fp_row["interaction_target_uuid"]).strip()
        t = orig[orig["uuid"] == tgt_uuid].sort_values("targetsOrgTimes").reset_index(drop=True)
        tgt_x = t["X"].to_numpy(float); tgt_y = t["Y"].to_numpy(float)
        tgt_t_rel = (t["targetsOrgTimes"].to_numpy(float) - TS) / 1000.0

        # other vehicles genuinely co-present in the interaction window (the ±5 s scene
        # extended by ±EXT_S before it starts / after it ends), each with its full
        # trajectory inside that window, animated in real time — no spatial sampling.
        _scene_min = max(float(t_rel[0]), -WIN_S) - EXT_S
        _scene_max = min(float(t_rel[-1]), WIN_S) + EXT_S
        _t0_ms = TS + _scene_min * 1000.0
        _t1_ms = TS + _scene_max * 1000.0
        _others = orig[(orig["targetsOrgTimes"] >= _t0_ms) & (orig["targetsOrgTimes"] <= _t1_ms)]
        _others = _others[~_others["uuid"].isin([UUID, tgt_uuid])]
        others = []
        for _ou, _grp in _others.groupby("uuid", sort=False):
            _grp = _grp.sort_values("targetsOrgTimes")
            if len(_grp) < 2:
                continue
            _ox = _grp["X"].to_numpy(float)
            _oy = _grp["Y"].to_numpy(float)
            _ots = _grp["targetsOrgTimes"].to_numpy(float)
            _oh = _grp["heading"].to_numpy(float)
            _len = float(_grp["len"].iloc[0]) / 100.0
            _wid = float(_grp["width"].iloc[0]) / 100.0
            others.append((_ox, _oy, (_ots - TS) / 1000.0, _oh, _len, _wid))

        # vehicle geometry (cm -> m) + heading
        ego_len = float(g["len"].iloc[0]) / 100.0
        ego_width = float(g["width"].iloc[0]) / 100.0
        ego_heading = g["heading"].to_numpy(float)
        tgt_len = float(t["len"].iloc[0]) / 100.0
        tgt_width = float(t["width"].iloc[0]) / 100.0
        tgt_heading = t["heading"].to_numpy(float)

        # crossing point (intersection) or closest-approach pair (proximity)
        isect = None; cross_ego = len(ego_x) - 1; cross_tgt = len(tgt_x) - 1
        closest_ego = 0; closest_tgt = 0
        if scn["boundary"]:
            isects = []
            for i in range(len(ego_x) - 1):
                for j in range(len(tgt_x) - 1):
                    r = segment_intersection_pt((ego_x[i], ego_y[i]), (ego_x[i + 1], ego_y[i + 1]),
                                                (tgt_x[j], tgt_y[j]), (tgt_x[j + 1], tgt_y[j + 1]))
                    if r is not None:
                        isects.append((r, (ego_ts[i] + t["targetsOrgTimes"].to_numpy(float)[j]) / 2))
            if isects:
                isect = min(isects, key=lambda x: abs(x[1] - TS))[0]
                cross_ego = int(np.argmin((ego_x - isect[0]) ** 2 + (ego_y - isect[1]) ** 2))
                cross_tgt = int(np.argmin((tgt_x - isect[0]) ** 2 + (tgt_y - isect[1]) ** 2))
        else:
            # closest ego/target points within the 5 s forward window (spatial, fig11-style)
            ego_fwd = (t_rel >= 0) & (t_rel <= FUTURE_S)
            tgt_fwd = (tgt_t_rel >= 0) & (tgt_t_rel <= FUTURE_S)
            if ego_fwd.any() and tgt_fwd.any():
                ep = np.column_stack([ego_x[ego_fwd], ego_y[ego_fwd]])
                tp = np.column_stack([tgt_x[tgt_fwd], tgt_y[tgt_fwd]])
                D = np.linalg.norm(ep[:, None, :] - tp[None, :, :], axis=2)
                ii, jj = np.unravel_index(np.argmin(D), D.shape)
                closest_ego = int(np.flatnonzero(ego_fwd)[ii])
                closest_tgt = int(np.flatnonzero(tgt_fwd)[jj])

        # raw metric series (ego)
        raw_full = {m: g[m].to_numpy(float) for m in METRICS}
        data[tag] = {
            "ego_x": ego_x, "ego_y": ego_y, "t_rel": t_rel,
            "tgt_x": tgt_x, "tgt_y": tgt_y, "tgt_t_rel": tgt_t_rel,
            "ego_len": ego_len, "ego_width": ego_width, "ego_heading": ego_heading,
            "tgt_len": tgt_len, "tgt_width": tgt_width, "tgt_heading": tgt_heading,
            "flag2_idx": flag2_idx, "fp_x": fp_x, "fp_y": fp_y, "fp_jerk": fp_jerk,
            "pet": pet, "dmin": dmin,
            "isect": isect, "cross_ego": cross_ego, "cross_tgt": cross_tgt,
            "closest_ego": closest_ego, "closest_tgt": closest_tgt,
            "raw_full": raw_full, "model_full": {},
            "others": others,
        }
        save.update({
            f"{tag}_ego_x": ego_x, f"{tag}_ego_y": ego_y, f"{tag}_t_rel": t_rel,
            f"{tag}_tgt_x": tgt_x, f"{tag}_tgt_y": tgt_y, f"{tag}_tgt_t_rel": tgt_t_rel,
            f"{tag}_ego_len": ego_len, f"{tag}_ego_width": ego_width, f"{tag}_ego_heading": ego_heading,
            f"{tag}_tgt_len": tgt_len, f"{tag}_tgt_width": tgt_width, f"{tag}_tgt_heading": tgt_heading,
            f"{tag}_flag2_idx": flag2_idx, f"{tag}_fp_x": fp_x, f"{tag}_fp_y": fp_y,
            f"{tag}_fp_jerk": fp_jerk, f"{tag}_pet": pet, f"{tag}_dmin": dmin,
            f"{tag}_isect": isect if isect is not None else np.array([]),
            f"{tag}_cross_ego": cross_ego, f"{tag}_cross_tgt": cross_tgt,
            f"{tag}_closest_ego": closest_ego, f"{tag}_closest_tgt": closest_tgt,
        })
        for m in METRICS:
            save[f"{tag}_raw_{m}"] = raw_full[m]
        save[f"{tag}_n_others"] = len(others)
        for _i, (_ox, _oy, _ot, _oh, _olen, _owid) in enumerate(others):
            save[f"{tag}_others_{_i}_x"] = _ox
            save[f"{tag}_others_{_i}_y"] = _oy
            save[f"{tag}_others_{_i}_t"] = _ot
            save[f"{tag}_others_{_i}_h"] = _oh
            save[f"{tag}_others_{_i}_len"] = _olen
            save[f"{tag}_others_{_i}_wid"] = _owid
        print(f"[{tag}] ego={len(ego_x)} pts  tgt={len(tgt_x)} pts  others={len(others)}  "
              f"flag2_idx={flag2_idx} jerk={fp_jerk:.3f} PET={pet:.2f}s d_min={dmin:.2f}m")

    # model reconstructed series (read each model CSV once, filter both egos)
    for mn in ["Proposed", "KCO", "LSTM", "SG"]:
        md = pd.read_csv(MODEL_FILES[mn], usecols=MODEL_COLS, low_memory=False)
        md["uuid"] = md["uuid"].astype(str).str.strip()
        for scn in SCENARIOS:
            tag, UUID = scn["tag"], scn["uuid"]
            sub = md[md["uuid"] == UUID].sort_values("targetsOrgTimes").reset_index(drop=True)
            data[tag]["model_full"][mn] = {m: sub[m].to_numpy(float) for m in METRICS}
            for m in METRICS:
                save[f"{tag}_mod_{mn}_{m}"] = data[tag]["model_full"][mn][m]
            print(f"  [{tag}] {mn}: {len(sub)} pts")

    # RAW model = raw metric series
    for scn in SCENARIOS:
        data[scn["tag"]]["model_full"]["RAW"] = {m: data[scn["tag"]]["raw_full"][m].copy() for m in METRICS}

    np.savez_compressed(CACHE, **save)
    print("cache written.")
    return data, corners_xy


# ---------------- build one scenario GIF ----------------
def make_scenario_gif(scn, d, map_lim, range_geo=None):
    tag = scn["tag"]

    ego_x, ego_y, t_rel = d["ego_x"], d["ego_y"], d["t_rel"]
    tgt_x, tgt_y, tgt_t_rel = d["tgt_x"], d["tgt_y"], d["tgt_t_rel"]
    ego_len, ego_width, ego_heading = d["ego_len"], d["ego_width"], d["ego_heading"]
    tgt_len, tgt_width, tgt_heading = d["tgt_len"], d["tgt_width"], d["tgt_heading"]
    flag2_idx, fp_x, fp_y, fp_jerk = d["flag2_idx"], d["fp_x"], d["fp_y"], d["fp_jerk"]
    isect, cross_ego, cross_tgt = d["isect"], d["cross_ego"], d["cross_tgt"]
    closest_ego, closest_tgt = d["closest_ego"], d["closest_tgt"]
    raw_full, model_full = d["raw_full"], d["model_full"]

    # ±5 s display window around the ISMD (flag=2) moment at t=0 (drop full trajectory
    # for the two involved vehicles); the ANIMATION window extends it by ±EXT_S so the
    # vehicles genuinely co-present just before/after the scene are also seen driving.
    disp_min = max(float(t_rel[0]), -WIN_S)
    disp_max = min(float(t_rel[-1]), WIN_S)
    T_MIN = disp_min - EXT_S
    T_MAX = disp_max + EXT_S

    # ±5 s trajectory masks + per-scenario map extent (ego/target + nearby vehicles)
    ego_win = (t_rel >= disp_min) & (t_rel <= disp_max)
    tgt_win = (tgt_t_rel >= disp_min) & (tgt_t_rel <= disp_max)
    _wx = [ego_x[ego_win], tgt_x[tgt_win]]
    _wy = [ego_y[ego_win], tgt_y[tgt_win]]
    for _ox, _oy, _ot, _oh, _olen, _owid in d["others"]:
        _wx.append(_ox); _wy.append(_oy)
    _wx = np.concatenate(_wx)
    _wy = np.concatenate(_wy)
    # proximity scenarios stop short of the crossing, so their vehicles alone would
    # frame a blank patch of road — pull the intersection RANGE polygon into the
    # extent so the road cross-section (交叉口) stays visible in the basemap.
    if not scn["boundary"] and range_geo is not None:
        _wx = np.concatenate([_wx, range_geo[3][:, 0]])
        _wy = np.concatenate([_wy, range_geo[3][:, 1]])
    _pad = 12.0
    _cx, _cy = (_wx.min() + _wx.max()) / 2.0, (_wy.min() + _wy.max()) / 2.0
    _half = max(_wx.max() - _wx.min(), _wy.max() - _wy.min()) / 2.0 + _pad
    map_lim = ((_cx - _half, _cx + _half), (_cy - _half, _cy + _half))

    # per-metric y-limits
    YLIM = {}
    for m in METRICS:
        allv = np.concatenate([raw_full[m]] + [model_full[mn][m] for mn in MODELS if mn != "RAW"])
        allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, [2, 98])
        rng = max(hi - lo, 1e-6)
        YLIM[m] = (lo - 0.18 * rng, hi + 0.18 * rng)

    # ---- figure ----
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8})
    fig = plt.figure(figsize=(13.0, 10.17), dpi=110)
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.10,
                             left=0.05, right=0.985, top=0.92, bottom=0.11)
    ax_map = fig.add_subplot(outer[0, 0])
    n_panels = len(MODELS) if BOTTOM_MODE == "models" else len(METRICS)
    inner = outer[1, 0].subgridspec(1, n_panels, wspace=0.30)
    axs = [fig.add_subplot(inner[0, c]) for c in range(n_panels)]

    # ---- LEFT map: paths are NOT pre-drawn — each appears only as a fading ribbon
    # (filled band, width = vehicle width) trailing the driving vehicle.

    # flag=2 point (always visible, top layer); its text caption is placed on the
    # right side of the map panel below (once the panel bbox is known)
    flag2_dot = ax_map.scatter([fp_x], [fp_y], c="black", s=30, zorder=11,
                               edgecolor="white", linewidth=0.7)

    # crossing point (intersection) / closest-approach segment (proximity)
    # dots sit on the top layer (above the moving vehicles) so they stay visible
    t1_ann = t2_ann = None
    if scn["boundary"] and isect is not None:
        ax_map.scatter([isect[0]], [isect[1]], color="#E28E2C", edgecolor="black",
                       linewidth=0.5, s=46, zorder=11)
        # t1 (target crosses) / t2 (ego crosses) at the crossing point — hidden until
        # each vehicle reaches the crossing (shown only during the crossing pause)
        t1_ann = ax_map.annotate(r"$t_1$", xy=(isect[0], isect[1]),
                                 xytext=(isect[0] - 3.5, isect[1] + 2.5),
                                 fontsize=14, color="#7A3E00", fontweight="bold",
                                 ha="center", zorder=12,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none", alpha=1.0))
        t2_ann = ax_map.annotate(r"$t_2$", xy=(isect[0], isect[1]),
                                 xytext=(isect[0] + 3.5, isect[1] - 2.5),
                                 fontsize=14, color="#7A3E00", fontweight="bold",
                                 ha="center", zorder=12,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none", alpha=1.0))
        t1_ann.set_visible(False)
        t2_ann.set_visible(False)
    elif not scn["boundary"]:
        ex0, ey0 = float(ego_x[closest_ego]), float(ego_y[closest_ego])
        tx0, ty0 = float(tgt_x[closest_tgt]), float(tgt_y[closest_tgt])
        ax_map.plot([ex0, tx0], [ey0, ty0], color="#E28E2C", linewidth=1.0,
                    linestyle="--", alpha=0.8, zorder=8)
        ax_map.scatter([ex0, tx0], [ey0, ty0], color="#E28E2C", edgecolor="black",
                       linewidth=0.5, s=30, zorder=11)
        # t1 = red car (target) reaches its closest point first; t2 = blue car (ego)
        # reaches its closest point second — labels stay hidden until each arrives
        seg = np.array([ex0 - tx0, ey0 - ty0])
        n = np.linalg.norm(seg)
        seg = seg / n if n > 1e-9 else np.array([1.0, 0.0])
        t1_ann = ax_map.annotate(r"$t_1$", xy=(tx0, ty0),
                                 xytext=(tx0 - seg[0] * 3.5, ty0 - seg[1] * 3.5),
                                 fontsize=14, color="#7A3E00", fontweight="bold",
                                 ha="center", zorder=12,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none", alpha=1.0))
        t2_ann = ax_map.annotate(r"$t_2$", xy=(ex0, ey0),
                                 xytext=(ex0 + seg[0] * 3.5, ey0 + seg[1] * 3.5),
                                 fontsize=14, color="#7A3E00", fontweight="bold",
                                 ha="center", zorder=12,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none", alpha=1.0))
        t1_ann.set_visible(False)
        t2_ann.set_visible(False)

    # scenario label + color legend — true top-left of the map panel (figure
    # coords, so they stay above the white space set_aspect('equal') leaves
    # around the square map)
    _sub = ax_map.get_subplotspec().get_position(fig)
    fig.text(_sub.x0, _sub.y1, scn["label"], ha="left", va="top",
             fontsize=16, fontweight="bold", color="black", zorder=20)
    fig.text(_sub.x0, _sub.y1 - 0.034, "ego (blue) & target (orange)",
             ha="left", va="top", fontsize=12, color="#555555", zorder=20)

    # map limits — unified square extent shared by both scenarios
    ax_map.set_xlim(*map_lim[0])
    ax_map.set_ylim(*map_lim[1])
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # intersection lane-level cross-section (driving/biking/shoulder/sidewalk) —
    # drawn for every scenario so proximity examples still show the road crossing
    if range_geo is not None:
        fp, d_B, n_B, _ = range_geo
        draw_lane_cross_section(ax_map, fp, d_B, n_B)

    # dynamic map artists — fading ribbons (paths) + moving rectangles, all black-bordered
    ego_trail = PolyCollection([], facecolors="none", zorder=5)
    tgt_trail = PolyCollection([], facecolors="none", zorder=5)
    ax_map.add_collection(ego_trail)
    ax_map.add_collection(tgt_trail)
    ego_rect = Polygon([[0, 0]], closed=True, facecolor=C_EGO, edgecolor="black",
                       linewidth=1.0, zorder=10)
    tgt_rect = Polygon([[0, 0]], closed=True, facecolor=C_TGT, edgecolor="black",
                       linewidth=1.0, zorder=10)
    ax_map.add_patch(ego_rect)
    ax_map.add_patch(tgt_rect)

    # context vehicles: grey-blue rectangles with black borders + their own fading ribbons
    other_rects = []
    other_trails = []
    for _ox, _oy, _ot, _oh, _olen, _owid in d["others"]:
        _p = Polygon([[0, 0]], closed=True, facecolor=C_OTHER, edgecolor="black",
                     linewidth=0.8, zorder=9)
        ax_map.add_patch(_p)
        _p.set_visible(False)
        other_rects.append((_p, _ox, _oy, _ot, _oh, _olen, _owid))
        _tr = PolyCollection([], facecolors="none", zorder=4)
        ax_map.add_collection(_tr)
        other_trails.append(_tr)

    # ---- right-side callouts: dashed leader lines from the flag=2 point and the
    # t1 (red target) / t2 (blue ego) points to labels placed in the white space to
    # the right of the square map (figure coords), so they never overlap the map.
    # In the intersection case the two t's share one crossing point, so they form a
    # side-by-side row: red (t1) LEFT of blue (t2), with the t2 - t1 box directly
    # below whose left border aligns with red's left and right border with blue's.
    from matplotlib.text import Text
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()  # finalize layout so the data -> figure transform is valid
    _renderer = fig.canvas.get_renderer()

    def _fig_xy(x, y):
        return fig.transFigure.inverted().transform(ax_map.transData.transform((x, y)))

    def _tw(s, fs=11):
        t = Text(0, 0, s, fontsize=fs, fontweight="bold")
        t.set_figure(fig)
        return t.get_window_extent(_renderer).width / (fig.dpi * fig.get_size_inches()[0])

    label_x = _sub.x1 - 0.012   # right edge of the label column

    if scn["boundary"]:
        cxx, cyy = (float(isect[0]), float(isect[1])) if isect is not None else (0.0, 0.0)
        t1_pt = t2_pt = (cxx, cyy)
        t1_time = float(tgt_t_rel[cross_tgt])
        t2_time = float(t_rel[cross_ego])
    else:
        t1_pt = (float(tgt_x[closest_tgt]), float(tgt_y[closest_tgt]))
        t2_pt = (float(ego_x[closest_ego]), float(ego_y[closest_ego]))
        t1_time = float(tgt_t_rel[closest_tgt])
        t2_time = float(t_rel[closest_ego])

    # figure-coordinate anchors (leader lines stay horizontal on screen)
    fpx, fpy = _fig_xy(float(fp_x), float(fp_y))
    t1fx, t1fy = _fig_xy(t1_pt[0], t1_pt[1])
    t2fx, t2fy = _fig_xy(t2_pt[0], t2_pt[1])
    same_pt = abs(t2fy - t1fy) < 0.005   # intersection: both t's at one crossing point

    f2_s = r"flag=2 (ISMD) jerk %.1f" % fp_jerk
    t1_s = r"$t_1$ (target) = %.2f s" % t1_time
    t2_s = r"$t_2$ (ego) = %.2f s" % t2_time
    gap_s = r"$t_2 - t_1$ = %.2f s" % (t2_time - t1_time)
    dmin_s = r"$d_{min}$ (min distance) = %.2f m" % d["dmin"]
    w_f2, w_t1, w_t2, w_dm = _tw(f2_s), _tw(t1_s), _tw(t2_s), _tw(dmin_s)

    def _hline(x0, y0, color, x1, y1=None):
        if y1 is None:
            y1 = y0
        ln = Line2D([x0, x1], [y0, y1], transform=fig.transFigure,
                    color=color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=30)
        fig.add_artist(ln)
        return ln

    # ---- per-scenario layout --------------------------------------------
    if scn["boundary"]:
        # intersection: red (t1) left of blue (t2); the t2 - t1 box spans exactly
        # red-left -> blue-right.
        GAP_H = 0.035
        blue_right = label_x
        blue_left = label_x - w_t2
        red_right = blue_left - GAP_H
        red_left = red_right - w_t1
        block_left = red_left
        block_cx = (red_left + label_x) / 2.0

        t1_x = red_right if same_pt else label_x
        t2_x = blue_right if same_pt else label_x
        t1_line_end = (block_left - 0.010) if same_pt else (label_x - w_t1 - 0.012)
        t2_line_end = (block_left - 0.010) if same_pt else (label_x - w_t2 - 0.012)
        t1_y, t2_y = t1fy, t2fy
        gap_x = block_cx if same_pt else label_x
        gap_ha = "center" if same_pt else "right"
        gap_y = (t2fy if same_pt else min(t1fy, t2fy)) - 0.040
        flag2_label_x = label_x
        flag2_label_y = max(fpy, t2fy + 0.055)
        flag2_line_end = label_x - w_f2 - 0.012
        dmin_line = None
        dmin_label = None
    else:
        # proximity (closest-approach): flag=2 keeps its own horizontal leader at the
        # top; the closest-approach quantities (t1, d_min, t2, t2 - t1) share ONE
        # horizontal leader line feeding a right-aligned stack, because t1 and t2 are
        # only ~1.5 m apart (visually coincident) so separate lines would overlap.
        _f2_end_old = label_x - w_f2 - 0.012
        f2_end = fpx + 0.8 * (_f2_end_old - fpx)
        black_right = f2_end + 0.012 + w_f2

        flag2_label_x = black_right
        flag2_label_y = fpy            # horizontal leader line
        flag2_line_end = f2_end

        # closest-approach group: right-aligned column whose top box sits on the
        # leader line; each box is one row lower.
        dm_mid = ((t1_pt[0] + t2_pt[0]) / 2.0, (t1_pt[1] + t2_pt[1]) / 2.0)
        dmfx, dmfy = _fig_xy(dm_mid[0], dm_mid[1])
        w_gap = _tw(gap_s)
        w_group = max(w_t1, w_t2, w_dm, w_gap)
        group_right = black_right
        group_left = group_right - w_group

        dh = 0.034
        t1_x = group_right
        t1_y = dmfy
        dmin_x = group_right
        dmin_y = dmfy - dh
        t2_x = group_right
        t2_y = dmfy - 2.0 * dh
        gap_x = group_right
        gap_ha = "right"
        gap_y = dmfy - 3.0 * dh

    # ---- shared callout artists -----------------------------------------
    flag2_line = _hline(fpx, fpy, "#888888", flag2_line_end, flag2_label_y)
    flag2_label = fig.text(flag2_label_x, flag2_label_y, f2_s, fontsize=11, color="#333333",
                           fontweight="bold", ha="right", va="center", zorder=31,
                           bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                     ec="#888888", lw=0.8, alpha=0.98))
    flag2_line.set_visible(False)
    flag2_label.set_visible(False)

    if scn["boundary"]:
        # t1 (red) / t2 (blue) — leader lines stop at the label block's left edge so
        # the blue line never crosses onto the red box.
        t1_line = _hline(t1fx, t1fy, C_TGT, t1_line_end, t1_y)
        t1_label = fig.text(t1_x, t1_y, t1_s, fontsize=11, color=C_TGT, fontweight="bold",
                            ha="right", va="center", zorder=31,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                      ec=C_TGT, lw=0.8, alpha=0.98))
        t1_line.set_visible(False)
        t1_label.set_visible(False)

        t2_line = _hline(t2fx, t2fy, C_EGO, t2_line_end, t2_y)
        t2_label = fig.text(t2_x, t2_y, t2_s, fontsize=11, color=C_EGO, fontweight="bold",
                            ha="right", va="center", zorder=31,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                      ec=C_EGO, lw=0.8, alpha=0.98))
        t2_line.set_visible(False)
        t2_label.set_visible(False)
    else:
        # proximity: one shared horizontal leader line for the whole closest-approach
        # stack; it appears with t1 and stays. (No separate t2 / d_min lines.)
        t1_line = _hline(dmfx, dmfy, "#888888", group_left - 0.012, dmfy)
        t1_line.set_visible(False)
        t2_line = None

        t1_label = fig.text(t1_x, t1_y, t1_s, fontsize=11, color=C_TGT, fontweight="bold",
                            ha="right", va="center", zorder=31,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                      ec=C_TGT, lw=0.8, alpha=0.98))
        t1_label.set_visible(False)

        t2_label = fig.text(t2_x, t2_y, t2_s, fontsize=11, color=C_EGO, fontweight="bold",
                            ha="right", va="center", zorder=31,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                      ec=C_EGO, lw=0.8, alpha=0.98))
        t2_label.set_visible(False)

        dmin_label = fig.text(dmin_x, dmin_y, dmin_s, fontsize=11, color="#7A3E00",
                              fontweight="bold", ha="right", va="center", zorder=31,
                              bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                        ec="#7A3E00", lw=0.8, alpha=0.98))
        dmin_label.set_visible(False)
        dmin_line = None

    # t2 - t1 box: in the intersection case it spans red-left -> blue-right (exact
    # border alignment with the row above); otherwise a simple right-aligned box.
    gap_label = fig.text(gap_x, gap_y, gap_s, fontsize=11, color="#333333",
                         fontweight="bold", ha=gap_ha, va="center", zorder=31)
    if scn["boundary"] and same_pt:
        gap_box = FancyBboxPatch((red_left, gap_y - 0.010), label_x - red_left, 0.020,
                                 boxstyle="round,pad=0,rounding_size=0.006",
                                 transform=fig.transFigure, fc="white", ec="#333333",
                                 lw=0.8, alpha=0.98, zorder=30)
        fig.add_artist(gap_box)
        gap_box.set_visible(False)
    else:
        gap_box = None
        gap_label.set_bbox(dict(boxstyle="round,pad=0.25", fc="white",
                                ec="#333333", lw=0.8, alpha=0.98))
    gap_label.set_visible(False)

    # ---- BOTTOM panels ----
    curve_lines = {}
    flag2_rings = []   # ring around the t=0 ISMD point (shown after t>=0), models mode
    flag2_labels = []  # "ISMD preserved / smoothed" caption above each box, models mode
    if BOTTOM_MODE == "models":
        # 5 model boxes (one per model), each showing a single BOX_METRIC curve,
        # with a model-colored border and the ISMD-preserved / smoothed ring + caption.
        # the ISMD is a deviation in ONE specific metric (the one sitting in the
        # moderate band at the flag=2 moment) — not necessarily jerk — so show that.
        m = BOX_METRIC
        for _cand in METRICS:
            _v = float(raw_full[_cand][flag2_idx])
            _cl, _ch = COMFORT[_cand]; _el, _eh = EXTREME[_cand]
            if not (_cl <= _v <= _ch) and not (_v < _el or _v > _eh):
                m = _cand
                break
        cl, ch = COMFORT[m]; el, eh = EXTREME[m]
        # a baseline "overshoots" if its peak |metric| over the display window EXCEEDS
        # the raw peak (amplifies the deviation); otherwise it smooths it away.
        raw_peak = float(np.abs(raw_full[m][ego_win]).max())
        for c, mname in enumerate(MODELS):
            ax = axs[c]
            # model-colored box border (each model is a distinct box)
            for sp in ax.spines.values():
                sp.set_color(MODEL_COLOR[mname])
                sp.set_linewidth(1.3)
            ax.axhline(cl, color="#2E9E44", ls="--", lw=0.8, alpha=0.9, zorder=1)
            ax.axhline(ch, color="#2E9E44", ls="--", lw=0.8, alpha=0.9, zorder=1)
            ax.axhline(el, color="#E53935", ls="-.", lw=0.8, alpha=0.9, zorder=1)
            ax.axhline(eh, color="#E53935", ls="-.", lw=0.8, alpha=0.9, zorder=1)
            ax.axvline(0, color="black", lw=0.6, alpha=0.4, zorder=1)
            # single curve per box (uniform line style across all models)
            curve_lines[mname], = ax.plot([], [], color=MODEL_COLOR[mname],
                                          lw=1.7, alpha=1.0, zorder=3, solid_capstyle="round")
            # ISMD-moment marker at t=0 (always visible) + ring (revealed at t>=0)
            ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=28,
                       color=MODEL_COLOR[mname], edgecolor="white", linewidth=0.7, zorder=6)
            ring = ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=150,
                              facecolors="none", edgecolors=MODEL_COLOR[mname],
                              linewidth=1.8, zorder=7)
            ring.set_visible(False)
            flag2_rings.append(ring)
            if mname == "RAW":
                lbl, lc = "", "black"
            elif mname == "Proposed":
                lbl, lc = "ISMD preserved\ninteraction information retained", "#2E7D32"
            elif float(np.abs(model_full[mname][m][ego_win]).max()) > raw_peak:
                lbl, lc = "ISMD overshoot\ninteraction information lost", "#C0392B"
            else:
                lbl, lc = "ISMD smoothed\ninteraction information lost", "#C0392B"
            # caption INSIDE the box (top, centered, white boxed label) so it never
            # overlaps the model-name title above
            txt = ax.text(0.5, 0.97, lbl, transform=ax.transAxes, fontsize=7.5,
                          color=lc, va="top", ha="center", fontweight="bold",
                          zorder=8,
                          bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=lc,
                                    lw=0.8, alpha=0.92))
            txt.set_visible(False)
            flag2_labels.append(txt)
            ax.set_ylim(*YLIM[m]); ax.set_xlim(disp_min, disp_max)
            title = mname + (" (ours)" if mname == "Proposed" else "")
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color=MODEL_COLOR[mname])
            ax.tick_params(labelsize=7)
            ax.set_xlabel("t (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel(METRIC_TITLE[m], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.grid(False)
        handles = [
            Line2D([0], [0], color="#2E9E44", ls="--", lw=0.9, label="Comfort limit"),
            Line2D([0], [0], color="#E53935", ls="-.", lw=0.9, label="Extreme limit"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   ncol=2, frameon=False, fontsize=8.5)
    else:
        # 4 metric panels (one per metric), all models overlaid, Proposed bold
        for c, m in enumerate(METRICS):
            ax = axs[c]
            cl, ch = COMFORT[m]; el, eh = EXTREME[m]
            ax.axhline(cl, color="#2E9E44", ls="--", lw=0.7, alpha=0.9, zorder=1)
            ax.axhline(ch, color="#2E9E44", ls="--", lw=0.7, alpha=0.9, zorder=1)
            ax.axhline(el, color="#E53935", ls="-.", lw=0.7, alpha=0.9, zorder=1)
            ax.axhline(eh, color="#E53935", ls="-.", lw=0.7, alpha=0.9, zorder=1)
            ax.axvline(0, color="black", lw=0.6, alpha=0.4, zorder=1)
            for mname in MODELS:
                lw = 3.0 if mname == "Proposed" else 1.3
                alpha = 1.0 if mname == "Proposed" else 0.40
                curve_lines[(m, mname)], = ax.plot([], [], color=MODEL_COLOR[mname],
                                                   lw=lw, alpha=alpha, zorder=3, solid_capstyle="round")
            # ISMD-moment markers at t=0 (Proposed emphasized, others faded)
            for mname in MODELS:
                s = 26 if mname == "Proposed" else 12
                alpha = 1.0 if mname == "Proposed" else 0.40
                ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=s,
                           color=MODEL_COLOR[mname], alpha=alpha, edgecolor="white", linewidth=0.7, zorder=6)
            ax.set_ylim(*YLIM[m]); ax.set_xlim(disp_min, disp_max)
            ax.set_title(METRIC_TITLE[m], fontsize=13, fontweight="bold", pad=5)
            ax.tick_params(labelsize=9)
            ax.set_xlabel("t (s)", fontsize=10)
            if c == 0:
                ax.set_ylabel("value", fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.grid(False)
        handles = [
            Line2D([0], [0], color="#555555", lw=1.8, alpha=0.7, label="RAW"),
            Line2D([0], [0], color="#E53935", lw=3.0, label="Proposed (ours)"),
            Line2D([0], [0], color="#089099", lw=1.8, alpha=0.7, label="KCO"),
            Line2D([0], [0], color="#1F77B4", lw=1.8, alpha=0.7, label="LSTM"),
            Line2D([0], [0], color="#E28E2C", lw=1.8, alpha=0.7, label="SG"),
            Line2D([0], [0], color="#2E9E44", ls="--", lw=0.9, label="Comfort limit"),
            Line2D([0], [0], color="#E53935", ls="-.", lw=0.9, label="Extreme limit"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   ncol=7, frameon=False, fontsize=11)

    # ---------------- animation (time-synchronized) ----------------
    n_ego = len(ego_x)
    n_tgt = len(tgt_x)
    t_start = T_MIN
    t_end = T_MAX
    t_tgt_first = float(tgt_t_rel[0])
    t_tgt_end = float(tgt_t_rel[-1])
    ego_t_first = float(t_rel[0])
    ego_t_last = float(t_rel[-1])
    ego_t0 = int(np.searchsorted(t_rel, T_MIN))
    tgt_t0 = int(np.searchsorted(tgt_t_rel, T_MIN))

    def ego_idx(t):
        return int(np.clip(np.searchsorted(t_rel, t), 0, n_ego - 1))

    def tgt_idx(t):
        return int(np.clip(np.searchsorted(tgt_t_rel, t), 0, n_tgt - 1))

    def add_time(t_a, t_b, nf, state="drive"):
        for f in range(nf):
            t = t_a + (t_b - t_a) * f / (nf - 1)
            frames.append((t, state))

    # shared pacing — constant simulated-seconds-per-frame so every GIF (crossing AND
    # proximity) plays at the same speed; the key moment (crossing / closest approach)
    # is slowed further since it is the key moment.
    DT_DRIVE = 0.18
    DT_CLOSE = 0.085

    def nd(span, dt):
        return max(2, int(round(span / dt)))

    # The animation spans the FULL recorded trajectory (ego_t_first/ego_t_last and
    # t_tgt_* are the raw endpoints), so vehicles drive in and out to their natural
    # ends. Consistent speed comes from the dt-driven frame counts below, not from
    # clipping to the ±WIN_S display window.
    if scn["boundary"]:
        # intersection: crossing point -> PET pauses (same as before).
        # cross_* are indices into the ego / target arrays respectively.
        t_ego_cross = float(t_rel[cross_ego])
        t_tgt_cross = float(tgt_t_rel[cross_tgt])

        frames = []
        add_time(ego_t_first, 0.0, nd(0.0 - ego_t_first, DT_DRIVE), "drive")
        for _ in range(22):
            frames.append((0.0, "flag2"))
        add_time(0.0, t_tgt_cross, nd(t_tgt_cross - 0.0, DT_CLOSE), "drive")
        for _ in range(14):
            frames.append((t_tgt_cross, "tgt_cross"))
        add_time(t_tgt_cross, t_ego_cross, nd(t_ego_cross - t_tgt_cross, DT_CLOSE), "drive")
        for _ in range(16):
            frames.append((t_ego_cross, "ego_cross"))
        add_time(t_ego_cross, ego_t_last, nd(ego_t_last - t_ego_cross, DT_DRIVE), "drive")
        # short hold at the end so the final PET value (t2 - t1) stays on screen
        for _ in range(18):
            frames.append((ego_t_last, "drive"))
    else:
        # proximity: flag=2 highlight + closest-approach pauses for both vehicles.
        # Uses the same dt-driven frame counts as the crossing branch so C3/C8 play at
        # the same speed and advance smoothly (the old fixed-count version let each
        # candidate's different trajectory length set its own speed, and rewound when
        # ego_t_first fell before the ±8 s scene window).
        t_tgt_closest = float(tgt_t_rel[closest_tgt])  # red reaches closest (t1)
        t_ego_closest = float(t_rel[closest_ego])      # blue reaches closest (t2)
        frames = []
        # clip the (often long, mostly-stationary) pre-interaction approach to the
        # ±5 s display window so the GIF stays short and focuses on the interaction
        add_time(disp_min, 0.0, nd(0.0 - disp_min, DT_DRIVE), "drive")
        for _ in range(22):
            frames.append((0.0, "flag2"))
        add_time(0.0, t_tgt_closest, nd(t_tgt_closest - 0.0, DT_CLOSE), "drive")
        for _ in range(14):
            frames.append((t_tgt_closest, "tgt_closest"))
        add_time(t_tgt_closest, t_ego_closest, nd(t_ego_closest - t_tgt_closest, DT_CLOSE), "drive")
        for _ in range(16):
            frames.append((t_ego_closest, "ego_closest"))
        add_time(t_ego_closest, ego_t_last, nd(ego_t_last - t_ego_closest, DT_DRIVE), "drive")
        for _ in range(18):
            frames.append((ego_t_last, "drive"))

    def update(step):
        t, state = frames[step]
        k = ego_idx(t)
        j = tgt_idx(t)
        update_ribbon(ego_trail, ego_x, ego_y, ego_width, C_EGO, ego_t0, k)
        update_ribbon(tgt_trail, tgt_x, tgt_y, tgt_width, C_TGT, tgt_t0, j)
        ego_rect.set_xy(veh_corners(ego_x[k], ego_y[k], ego_heading[k], ego_len, ego_width))
        ego_rect.set_visible(ego_t_first <= t <= ego_t_last)
        if t_tgt_first <= t <= t_tgt_end:
            tgt_rect.set_xy(veh_corners(tgt_x[j], tgt_y[j], tgt_heading[j], tgt_len, tgt_width))
            tgt_rect.set_visible(True)
        else:
            tgt_rect.set_visible(False)
        # context vehicles drive in real time (visible only within their own window)
        for _i, (_p, _ox, _oy, _ot, _oh, _olen, _owid) in enumerate(other_rects):
            if _ot[0] <= t <= _ot[-1]:
                _j = int(np.clip(np.searchsorted(_ot, t), 0, len(_ot) - 1))
                _p.set_xy(veh_corners(_ox[_j], _oy[_j], _oh[_j], _olen, _owid))
                _p.set_visible(True)
                update_ribbon(other_trails[_i], _ox, _oy, _owid, C_OTHER, 0, _j)
            else:
                _p.set_visible(False)
                other_trails[_i].set_verts([])
        # highlight at key moments
        ego_rect.set_edgecolor("black"); ego_rect.set_linewidth(1.0)
        tgt_rect.set_edgecolor("black"); tgt_rect.set_linewidth(1.0)
        if state == "flag2":
            ego_rect.set_edgecolor("#F4C20D"); ego_rect.set_linewidth(1.8)
        elif state == "tgt_cross" or state == "tgt_closest":
            tgt_rect.set_edgecolor("#E28E2C"); tgt_rect.set_linewidth(1.8)
        elif state == "ego_cross" or state == "ego_closest":
            ego_rect.set_edgecolor("#E28E2C"); ego_rect.set_linewidth(1.8)

        # flag=2 callout (gray leader + label) reveals once the ISMD moment passes
        flag2_line.set_visible(t >= 0)
        flag2_label.set_visible(t >= 0)
        # ISMD rings + preserved/smoothed captions reveal once the ISMD moment passes
        for ring in flag2_rings:
            ring.set_visible(t >= 0)
        for lbl in flag2_labels:
            if lbl.get_text():
                lbl.set_visible(t >= 0)

        # t1 / t2 labels persist once each vehicle reaches its point; the leader
        # lines are transient (red during the t1 pause, blue during the t2 pause)
        t1_label.set_visible(t >= t1_time)
        t2_label.set_visible(t >= t2_time)
        gap_label.set_visible(t >= t2_time)
        if gap_box is not None:
            gap_box.set_visible(t >= t2_time)
        # d_min (brown) off-map callout reveals only after the blue car passes the
        # closest point (proximity scenarios only)
        if dmin_line is not None:
            dmin_line.set_visible(t >= t2_time)
        if dmin_label is not None:
            dmin_label.set_visible(t >= t2_time)
        if scn["boundary"]:
            t1_line.set_visible(state == "tgt_cross")
            t2_line.set_visible(state == "ego_cross")
        else:
            # proximity: one shared leader line stays once the closest-approach group
            # begins (t1 appears); there is no separate t2 leader line
            t1_line.set_visible(t >= t1_time)
            if t2_line is not None:
                t2_line.set_visible(False)

        # t1/t2 map labels appear only while each vehicle crosses / reaches closest
        if scn["boundary"]:
            t1_ann.set_visible(state == "tgt_cross")
            t2_ann.set_visible(state == "ego_cross")
        else:
            t1_ann.set_visible(state == "tgt_closest")
            t2_ann.set_visible(state == "ego_closest")

        arts = [ego_trail, tgt_trail, ego_rect, tgt_rect, flag2_dot,
                flag2_line, flag2_label, t1_line, t1_label, t2_line, t2_label, gap_label]
        if gap_box is not None:
            arts.append(gap_box)
        if BOTTOM_MODE == "models":
            m = BOX_METRIC
            for mname in MODELS:
                nk = min(k, len(model_full[mname][m]) - 1)
                curve_lines[mname].set_data(t_rel[:nk + 1], model_full[mname][m][:nk + 1])
                arts.append(curve_lines[mname])
        else:
            for m in METRICS:
                for mname in MODELS:
                    nk = min(k, len(model_full[mname][m]) - 1)
                    curve_lines[(m, mname)].set_data(t_rel[:nk + 1], model_full[mname][m][:nk + 1])
                    arts.append(curve_lines[(m, mname)])
        return arts

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    suffix = "_models" if BOTTOM_MODE == "models" else ""
    out_gif = os.path.join(OUT, f"{tag}{suffix}.gif")
    anim.save(out_gif, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"[{tag}] saved {out_gif} ({len(frames)} frames)")
    return out_gif


if __name__ == "__main__":
    data, _ = load_scenarios()
    # unified square map extent shared by both scenarios (same base-map size)
    all_x = np.concatenate([data[s["tag"]]["ego_x"] for s in SCENARIOS] +
                           [data[s["tag"]]["tgt_x"] for s in SCENARIOS])
    all_y = np.concatenate([data[s["tag"]]["ego_y"] for s in SCENARIOS] +
                           [data[s["tag"]]["tgt_y"] for s in SCENARIOS])
    PAD = 8.0
    x_mid = (all_x.min() + all_x.max()) / 2.0
    y_mid = (all_y.min() + all_y.max()) / 2.0
    half = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2.0 + PAD
    map_lim = ((x_mid - half, x_mid + half), (y_mid - half, y_mid + half))
    range_geo = load_range_poly()
    for s in SCENARIOS:
        make_scenario_gif(s, data[s["tag"]], map_lim, range_geo=range_geo)
    print("done")
