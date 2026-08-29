# -*- coding: utf-8 -*-
"""
GitHub homepage GIF — two interaction examples from the paper (Fig11):
  (a) Intersection (trajectory-crossing) interaction
  (b) Proximity (closest_only) interaction
Layout (identical for both):
  LEFT  : road DENSITY HEATMAP (all vehicles -> road layout); ego (blue) / target (red)
          DRIVE along their paths as oriented rectangles, pause at the flag=2 (ISMD)
          point and at the crossing / closest-approach point; 5 s forward windows bold.
  RIGHT : 5 models (RAW / Proposed / KCO / LSTM / SG) x 4 metrics (speed/ax/ay/jerk),
          revealed in real time, synced to the left vehicle drive, with comfort (green
          dashed) / extreme (red dash-dot) thresholds.
The two parts are rendered to separate GIFs and concatenated into one.
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
from PIL import Image

BASE = r"F:\Rrajectory_Reconstruction\raw data preprocessing\5_all_data"
OUT = os.path.join(BASE, "github_demo")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_cache_both.npz")
DENSITY = os.path.join(OUT, "_density.npz")

RAW_FILE = os.path.join(BASE, "5_Data_vehicle_need_spatiotemporal_4metrics_30m.csv")

EARTH_RADIUS_M = 6378137.0
FUTURE_S = 5.0
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

METRICS = ["speed", "ax", "ay", "jerk"]
METRIC_TITLE = {"speed": "Speed (m/s)", "ax": r"$a_x$ (m/s$^2$)",
                "ay": r"$a_y$ (m/s$^2$)", "jerk": r"Jerk (m/s$^3$)"}
COMFORT = {"speed": (0, 23), "ax": (-5, 5), "ay": (-1.764, 1.764), "jerk": (-10, 10)}
EXTREME = {"speed": (0, 41.67), "ax": (-10.6, 6.87), "ay": (-2.5, 2.5), "jerk": (-23, 23)}

RAW_COLS = ["uuid", "targetsOrgTimes", "X", "Y", "longitude", "latitude",
            "speed", "ax", "ay", "jerk", "point_flag", "pet_seconds",
            "conflict_distance_m", "interaction_mode", "interaction_target_uuid",
            "len", "width", "heading"]
MODEL_COLS = ["uuid", "targetsOrgTimes", "speed", "ax", "ay", "jerk"]

# (a) crossing example and (b) proximity example — from fig11_interaction_preservation_two_examples.ipynb
SCENARIOS = [
    {"tag": "intersection", "label": "(a) Intersection interaction scenario",
     "uuid": "0d84f99d801d52e1ba739748", "ts": 1768359723902, "boundary": True},
    {"tag": "proximity", "label": "(b) Proximity interaction scenario",
     "uuid": "ea4232fdf4bfdc47bcb5ad20", "ts": 1768397602456, "boundary": False},
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
        print(f"[{tag}] ego={len(ego_x)} pts  tgt={len(tgt_x)} pts  flag2_idx={flag2_idx} "
              f"jerk={fp_jerk:.3f} PET={pet:.2f}s d_min={dmin:.2f}m")

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
def make_scenario_gif(scn, d, map_lim):
    tag = scn["tag"]
    ego_x, ego_y, t_rel = d["ego_x"], d["ego_y"], d["t_rel"]
    tgt_x, tgt_y, tgt_t_rel = d["tgt_x"], d["tgt_y"], d["tgt_t_rel"]
    ego_len, ego_width, ego_heading = d["ego_len"], d["ego_width"], d["ego_heading"]
    tgt_len, tgt_width, tgt_heading = d["tgt_len"], d["tgt_width"], d["tgt_heading"]
    flag2_idx, fp_x, fp_y, fp_jerk = d["flag2_idx"], d["fp_x"], d["fp_y"], d["fp_jerk"]
    isect, cross_ego, cross_tgt = d["isect"], d["cross_ego"], d["cross_tgt"]
    closest_ego, closest_tgt = d["closest_ego"], d["closest_tgt"]
    raw_full, model_full = d["raw_full"], d["model_full"]

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
    fig = plt.figure(figsize=(16.5, 8.4), dpi=110)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.9], wspace=0.06,
                             left=0.03, right=0.985, top=0.90, bottom=0.075)
    ax_map = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(4, 5, hspace=0.55, wspace=0.30)
    axs = np.empty((4, 5), dtype=object)
    for r in range(4):
        for c in range(5):
            axs[r, c] = fig.add_subplot(inner[r, c])

    # ---- LEFT map: density heatmap + trajectories ----
    dz = np.load(DENSITY)
    dens = dz["density"]; dXmin = float(dz["Xmin"]); dXmax = float(dz["Xmax"])
    dYmin = float(dz["Ymin"]); dYmax = float(dz["Ymax"])
    ax_map.imshow(dens, extent=[dXmin, dXmax, dYmin, dYmax], origin="lower", cmap="Greys",
                  vmin=0, vmax=np.percentile(dens, 95), alpha=0.85, zorder=0, aspect="auto")

    # full trajectories (uniform faint guide, same color + width as the drive trail)
    ax_map.plot(ego_x, ego_y, color="#0F4D92", linewidth=2.0, alpha=0.4, zorder=2)
    ax_map.plot(tgt_x, tgt_y, color="#E53935", linewidth=2.0, alpha=0.4, zorder=2)
    # direction arrows
    ax_map.annotate("", xy=(ego_x[-1], ego_y[-1]), xytext=(ego_x[-2], ego_y[-2]),
                    arrowprops=dict(arrowstyle="->", color="#0F4D92", lw=1.6), zorder=5)
    ax_map.annotate("", xy=(tgt_x[-1], tgt_y[-1]), xytext=(tgt_x[-2], tgt_y[-2]),
                    arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.6), zorder=5)

    # flag=2 point (always visible, top layer) + label (below the point)
    flag2_dot = ax_map.scatter([fp_x], [fp_y], c="black", s=30, zorder=11,
                               edgecolor="white", linewidth=0.7)
    flag2_lbl = "flag=2 (ISMD)\ninteraction information (jerk %.1f)" % fp_jerk
    flag2_ann = ax_map.annotate(flag2_lbl, xy=(fp_x, fp_y), xytext=(fp_x, fp_y - 6),
                                fontsize=12, color="#0F4D92", ha="center", va="top", fontweight="bold",
                                arrowprops=dict(arrowstyle="-", color="#0F4D92", lw=0.8), zorder=11,
                                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=1.0))
    flag2_ann.set_visible(False)

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
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1.0))
        t2_ann = ax_map.annotate(r"$t_2$", xy=(isect[0], isect[1]),
                                 xytext=(isect[0] + 3.5, isect[1] - 2.5),
                                 fontsize=14, color="#7A3E00", fontweight="bold",
                                 ha="center", zorder=12,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1.0))
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
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1.0))
        t2_ann = ax_map.annotate(r"$t_2$", xy=(ex0, ey0),
                                 xytext=(ex0 + seg[0] * 3.5, ey0 + seg[1] * 3.5),
                                 fontsize=14, color="#7A3E00", fontweight="bold",
                                 ha="center", zorder=12,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1.0))
        t1_ann.set_visible(False)
        t2_ann.set_visible(False)
        # d_min label with a leader line pointing to the closest-approach segment
        mid_x, mid_y = (ex0 + tx0) / 2.0, (ey0 + ty0) / 2.0
        dmin_ann = ax_map.annotate(r"$d_{min}$ = %.2f m" % d["dmin"],
                                   xy=(mid_x, mid_y),
                                   xytext=(mid_x, mid_y + 5.0),
                                   fontsize=13, color="#7A3E00", fontweight="bold",
                                   ha="center", va="bottom", zorder=12,
                                   arrowprops=dict(arrowstyle="->", color="#E28E2C", lw=1.2),
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1.0))

    # scenario label + color legend — true top-left of the map panel (figure
    # coords, so they stay above the white space set_aspect('equal') leaves
    # around the square map)
    _sub = ax_map.get_subplotspec().get_position(fig)
    fig.text(_sub.x0, _sub.y1, scn["label"], ha="left", va="top",
             fontsize=16, fontweight="bold", color="black", zorder=20)
    fig.text(_sub.x0, _sub.y1 - 0.034, "ego (blue) & interacting vehicle (red)",
             ha="left", va="top", fontsize=12, color="#555555", zorder=20)

    # map limits — unified square extent shared by both scenarios
    ax_map.set_xlim(*map_lim[0])
    ax_map.set_ylim(*map_lim[1])
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # dynamic map artists (moving rectangles)
    ego_trail, = ax_map.plot([], [], color="#0F4D92", linewidth=2.0, zorder=6)
    tgt_trail, = ax_map.plot([], [], color="#E53935", linewidth=2.0, zorder=6)
    ego_rect = Polygon([[0, 0]], closed=True, facecolor="#0F4D92", edgecolor="white",
                       linewidth=0.9, zorder=10)
    tgt_rect = Polygon([[0, 0]], closed=True, facecolor="#E53935", edgecolor="white",
                       linewidth=0.9, zorder=10)
    ax_map.add_patch(ego_rect)
    ax_map.add_patch(tgt_rect)

    # d_min annotation for proximity (orange box, like the PET box in intersection)
    if scn["boundary"]:
        pet_anchor_x = float(tgt_x.max()) + 3.0
        pet_anchor_y = float(ego_y.min()) - 7.0
        pet_text = ax_map.text(pet_anchor_x, pet_anchor_y, "", fontsize=12, color="#7A3E00",
                               ha="left", va="bottom",
                               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#E28E2C",
                                         lw=1.0, alpha=0.92), zorder=12)
    else:
        # time box anchored to the right side of the map (bottom-right)
        dmin_anchor_x = map_lim[0][1] - 3.0
        dmin_anchor_y = map_lim[1][0] + 3.0
        pet_text = ax_map.text(dmin_anchor_x, dmin_anchor_y, "", fontsize=12, color="#7A3E00",
                               ha="right", va="bottom",
                               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#E28E2C",
                                         lw=1.0, alpha=0.92), zorder=12)

    # ---- RIGHT: 5 models x 4 metrics, real-time reveal ----
    curve_lines = np.empty((4, 5), dtype=object)
    flag2_rings = [None] * len(MODELS)
    flag2_labels = [None] * len(MODELS)
    for r, m in enumerate(METRICS):
        for c, mname in enumerate(MODELS):
            ax = axs[r, c]
            cl, ch = COMFORT[m]; el, eh = EXTREME[m]
            ax.axhline(cl, color="#2E9E44", ls="--", lw=0.7, alpha=0.9, zorder=1)
            ax.axhline(ch, color="#2E9E44", ls="--", lw=0.7, alpha=0.9, zorder=1)
            ax.axhline(el, color="#E53935", ls="-.", lw=0.7, alpha=0.9, zorder=1)
            ax.axhline(eh, color="#E53935", ls="-.", lw=0.7, alpha=0.9, zorder=1)
            ax.axvline(0, color="black", lw=0.5, alpha=0.35, zorder=1)
            ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=16,
                       color=MODEL_COLOR[mname], edgecolor="white", linewidth=0.6, zorder=6)
            if r == 3:
                ring = ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=150,
                                  facecolors="none", edgecolors=MODEL_COLOR[mname],
                                  linewidth=1.8, zorder=7)
                ring.set_visible(False)
                flag2_rings[c] = ring
                if mname == "RAW":
                    lbl, lc = "", "black"
                elif mname == "Proposed":
                    lbl, lc = "ISMD preserved\ninteraction information retained", "#2E7D32"
                elif mname == "LSTM":
                    lbl, lc = "ISMD smoothed\ninteraction information lost", "#C0392B"
                else:
                    lbl, lc = "ISMD smoothed\ninteraction information lost", "#C0392B"
                txt = ax.text(0.5, 1.22, lbl, transform=ax.transAxes, fontsize=8.5,
                              color=lc, va="center", ha="center", fontweight="bold",
                              zorder=8, clip_on=False)
                txt.set_visible(False)
                flag2_labels[c] = txt
            curve_lines[r, c], = ax.plot([], [], color=MODEL_COLOR[mname], lw=1.5, zorder=3)
            ax.set_ylim(*YLIM[m]); ax.set_xlim(t_rel[0], t_rel[-1])
            ax.tick_params(labelsize=6)
            ax.grid(False)
            if r == 0:
                ax.set_title(MODEL_LABEL[mname], fontsize=9, fontweight="bold",
                             color=MODEL_COLOR[mname], pad=3)
            if r == 3:
                ax.set_xlabel("t (s)", fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(METRIC_TITLE[m], fontsize=7)
            else:
                ax.set_yticklabels([])

    # legend
    handles = [
        Line2D([0], [0], color="#555555", lw=1.5, label="RAW"),
        Line2D([0], [0], color="#E53935", lw=1.5, label="Proposed"),
        Line2D([0], [0], color="#089099", lw=1.5, label="KCO"),
        Line2D([0], [0], color="#1F77B4", lw=1.5, label="LSTM"),
        Line2D([0], [0], color="#E28E2C", lw=1.5, label="SG"),
        Line2D([0], [0], color="#2E9E44", ls="--", lw=0.8, label="Comfort limit"),
        Line2D([0], [0], color="#E53935", ls="-.", lw=0.8, label="Extreme limit"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=7, frameon=False, fontsize=8)

    # ---------------- animation (time-synchronized) ----------------
    n_ego = len(ego_x)
    n_tgt = len(tgt_x)
    t_start = float(t_rel[0])
    t_end = float(t_rel[-1])
    t_tgt_end = float(tgt_t_rel[-1])

    def ego_idx(t):
        return int(np.clip(np.searchsorted(t_rel, t), 0, n_ego - 1))

    def tgt_idx(t):
        return int(np.clip(np.searchsorted(tgt_t_rel, t), 0, n_tgt - 1))

    def add_time(t_a, t_b, nf, state="drive"):
        for f in range(nf):
            t = t_a + (t_b - t_a) * f / (nf - 1)
            frames.append((t, state))

    if scn["boundary"]:
        # intersection: crossing point -> PET pauses (same as before)
        t_tgt_cross = float(t_rel[cross_tgt]) if scn["boundary"] else 0.0
        # NOTE: cross_* are indices into ego/tgt arrays; t_rel[cross_ego] is ego crossing time
        t_ego_cross = float(t_rel[cross_ego])
        t_tgt_cross = float(tgt_t_rel[cross_tgt])
        frames = []
        add_time(t_start, 0.0, 26, "drive")
        for _ in range(22):
            frames.append((0.0, "flag2"))
        add_time(0.0, t_tgt_cross, 18, "drive")
        for _ in range(14):
            frames.append((t_tgt_cross, "tgt_cross"))
        add_time(t_tgt_cross, t_ego_cross, 32, "drive")
        for _ in range(16):
            frames.append((t_ego_cross, "ego_cross"))
        add_time(t_ego_cross, t_end, 12, "drive")
        for _ in range(12):
            frames.append((t_end, "drive"))
    else:
        # proximity: flag=2 highlight + closest-approach pauses for both vehicles
        # (slowed to ~10 fps-of-simulated-time to match the intersection pacing)
        t_tgt_closest = float(tgt_t_rel[closest_tgt])  # red reaches closest (t1)
        t_ego_closest = float(t_rel[closest_ego])      # blue reaches closest (t2)
        frames = []
        add_time(t_start, 0.0, 48, "drive")
        for _ in range(24):
            frames.append((0.0, "flag2"))
        for _ in range(14):
            frames.append((t_tgt_closest, "tgt_closest"))
        add_time(t_tgt_closest, t_ego_closest, 37, "drive")
        for _ in range(16):
            frames.append((t_ego_closest, "ego_closest"))
        add_time(t_ego_closest, t_end, 27, "drive")
        for _ in range(14):
            frames.append((t_end, "drive"))

    def update(step):
        t, state = frames[step]
        k = ego_idx(t)
        j = tgt_idx(t)
        ego_trail.set_data(ego_x[:k + 1], ego_y[:k + 1])
        tgt_trail.set_data(tgt_x[:j + 1], tgt_y[:j + 1])
        ego_rect.set_xy(veh_corners(ego_x[k], ego_y[k], ego_heading[k], ego_len, ego_width))
        if t <= t_tgt_end:
            tgt_rect.set_xy(veh_corners(tgt_x[j], tgt_y[j], tgt_heading[j], tgt_len, tgt_width))
            tgt_rect.set_visible(True)
        else:
            tgt_rect.set_visible(False)
        # highlight at key moments
        ego_rect.set_edgecolor("white"); ego_rect.set_linewidth(0.9)
        tgt_rect.set_edgecolor("white"); tgt_rect.set_linewidth(0.9)
        if state == "flag2":
            ego_rect.set_edgecolor("#F4C20D"); ego_rect.set_linewidth(1.8)
        elif state == "tgt_cross" or state == "tgt_closest":
            tgt_rect.set_edgecolor("#E28E2C"); tgt_rect.set_linewidth(1.8)
        elif state == "ego_cross" or state == "ego_closest":
            ego_rect.set_edgecolor("#E28E2C"); ego_rect.set_linewidth(1.8)

        # flag=2 annotation + metric-panel rings (the black dot is always visible)
        if t >= 0:
            flag2_ann.set_visible(True)
            for ring in flag2_rings:
                if ring is not None:
                    ring.set_visible(True)
            for lbl in flag2_labels:
                if lbl is not None:
                    lbl.set_visible(True)
        else:
            flag2_ann.set_visible(False)
            for ring in flag2_rings:
                if ring is not None:
                    ring.set_visible(False)
            for lbl in flag2_labels:
                if lbl is not None:
                    lbl.set_visible(False)

        # annotation: PET (intersection) / d_min (proximity)
        if scn["boundary"]:
            t_tgt_cross = float(tgt_t_rel[cross_tgt])
            t_ego_cross = float(t_rel[cross_ego])
            if t < t_tgt_cross:
                pet_text.set_text("")
            elif t < t_ego_cross:
                pet_text.set_text(r"$t_1$ = %.2f s" % t_tgt_cross)
            elif t < t_end:
                pet_text.set_text(r"$t_1$ = %.2f s" % t_tgt_cross + "\n"
                                  r"$t_2$ = %.2f s" % t_ego_cross)
            else:
                pet_text.set_text(r"$t_1$ = %.2f s" % t_tgt_cross + "\n"
                                  r"$t_2$ = %.2f s" % t_ego_cross + "\n"
                                  r"$t_2 - t_1$ = %.2f s" % (t_ego_cross - t_tgt_cross))
            # t1/t2 map labels appear only while each vehicle crosses
            t1_ann.set_visible(state == "tgt_cross")
            t2_ann.set_visible(state == "ego_cross")
        else:
            t_tgt = float(tgt_t_rel[closest_tgt])   # red car reaches closest first (t1)
            t_ego = float(t_rel[closest_ego])       # blue car reaches closest second (t2)
            # reveal t1 (red) -> t2 (blue) -> t2 - t1 as the animation progresses;
            # the time gap appears only at the very end
            if t < t_tgt:
                pet_text.set_text("")
            elif t < t_ego:
                pet_text.set_text(r"$t_1$ = %.2f s" % t_tgt)
            elif t < t_end:
                pet_text.set_text(r"$t_1$ = %.2f s" % t_tgt + "\n"
                                  r"$t_2$ = %.2f s" % t_ego)
            else:
                pet_text.set_text(r"$t_1$ = %.2f s" % t_tgt + "\n"
                                  r"$t_2$ = %.2f s" % t_ego + "\n"
                                  r"$t_2 - t_1$ = %.2f s" % (t_ego - t_tgt))
            # map labels t1/t2 appear only during the closest-approach pauses
            t1_ann.set_visible(state == "tgt_closest")
            t2_ann.set_visible(state == "ego_closest")

        arts = [ego_trail, tgt_trail, ego_rect, tgt_rect, flag2_dot, flag2_ann, pet_text]
        for r, m in enumerate(METRICS):
            for c, mname in enumerate(MODELS):
                nk = min(k, len(model_full[mname][m]) - 1)
                curve_lines[r, c].set_data(t_rel[:nk + 1], model_full[mname][m][:nk + 1])
                arts.append(curve_lines[r, c])
        return arts

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    out_gif = os.path.join(OUT, f"{tag}.gif")
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
    for s in SCENARIOS:
        make_scenario_gif(s, data[s["tag"]], map_lim)
    print("done")
