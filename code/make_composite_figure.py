# -*- coding: utf-8 -*-
"""
ONE animated composite GIF (一张图) combining U8 (intersection / crossing) and
C3 (proximity / closest-approach) in the "models" layout, using ALL 7 paper models:

  TOP    : 7 SQUARE model boxes for U8 — Raw / Proposed / KCO / LSTM / KF / SG /
           Wavelet-SG — each box showing that model's ISMD-trigger metric curve over the
           ±5 s window, with comfort / extreme limit lines and the ISMD preserved /
           overshoot / smoothed annotation.
  MIDDLE : two LARGE square maps side-by-side — U8 (left, crossing) and C3 (right,
           proximity) — both drive in sync, with the intersection cross-section basemap,
           fading trajectories, the flag=2 (ISMD) point and the crossing / closest-approach
           points. The t1 / t2 / d_min / flag=2 labels sit INLINE next to the points
           (no leader lines, 不要拉出线) and appear once each moment is reached.
  BOTTOM : the same 7 square model boxes for C3.

Both scenarios share t=0 as the flag=2 (ISMD) moment, so they are driven through the same
story (approach -> flag=2 -> t1 -> t2 -> leave) in lockstep, each interpolating its own
t1 / t2 timing. Every tile is square; the maps are simply 3.5× the small boxes.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_intersection_gif as mif

METRICS = mif.METRICS
METRIC_TITLE = mif.METRIC_TITLE
COMFORT = mif.COMFORT
EXTREME = mif.EXTREME
C_EGO = mif.C_EGO
C_TGT = mif.C_TGT
C_OTHER = mif.C_OTHER
WIN_S = mif.WIN_S
FPS = mif.FPS

# the full 7-model lineup from the paper (Raw + 6 reconstruction methods)
COMP_MODELS = ["RAW", "Proposed", "KCO", "LSTM", "KF", "SG", "WaveletSG"]
COMP_LABEL = {"RAW": "RAW", "Proposed": "Proposed", "KCO": "KCO", "LSTM": "LSTM",
              "KF": "KF", "SG": "SG", "WaveletSG": "Wavelet-SG"}
COMP_COLOR = {"RAW": "#555555", "Proposed": "#E53935", "KCO": "#089099",
              "LSTM": "#1F77B4", "KF": "#7B1FA2", "SG": "#E28E2C", "WaveletSG": "#C2185B"}

OUT_GIF = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "docs", "static", "videos",
                                       "composite_U8_C3.gif"))

_MNAME = {"jerk": "jerk", "ax": "ax", "ay": "ay", "speed": "speed"}


def trigger_metric(d):
    """The ISMD is a deviation in ONE specific metric (the one sitting in the moderate
    band at the flag=2 moment) — not necessarily jerk. Return (metric, value@flag2)."""
    raw_full = d["raw_full"]
    flag2_idx = d["flag2_idx"]
    m = "jerk"
    for cand in METRICS:
        v = float(raw_full[cand][flag2_idx])
        cl, ch = COMFORT[cand]
        el, eh = EXTREME[cand]
        if not (cl <= v <= ch) and not (v < el or v > eh):
            m = cand
            break
    return m, float(raw_full[m][flag2_idx])


def load_extra_models(data):
    """Load KF and Wavelet-SG (they live in 4metrics_4model/, not in mif.MODEL_FILES)
    and attach them to U8/C3's model_full dict."""
    BASE = mif.BASE
    MODEL_COLS = mif.MODEL_COLS
    extra = {
        "KF": os.path.join(BASE, "4metrics_4model", "5_Data_vehicle_features_KF.csv"),
        "WaveletSG": os.path.join(BASE, "4metrics_4model", "5_Data_vehicle_features_WaveletSG.csv"),
    }
    for tag in ["U8", "C3"]:
        uuid = [s for s in mif.SCENARIOS if s["tag"] == tag][0]["uuid"]
        for mn, path in extra.items():
            md = pd.read_csv(path, usecols=MODEL_COLS, low_memory=False)
            md["uuid"] = md["uuid"].astype(str).str.strip()
            sub = md[md["uuid"] == uuid].sort_values("targetsOrgTimes").reset_index(drop=True)
            data[tag]["model_full"][mn] = {m: sub[m].to_numpy(float) for m in METRICS}
            print("[%s] %s: %d pts" % (tag, mn, len(sub)))
    return data


def setup_boxes(axes, d, disp_min, disp_max):
    """Per-model SQUARE boxes (one model per box, single ISMD-trigger metric curve)."""
    raw_full = d["raw_full"]
    model_full = d["model_full"]
    flag2_idx = d["flag2_idx"]
    t_rel = d["t_rel"]

    m, _ = trigger_metric(d)
    ego_win = (t_rel >= disp_min) & (t_rel <= disp_max)
    raw_peak = float(np.abs(raw_full[m][ego_win]).max())

    YLIM = {}
    for mm in METRICS:
        allv = np.concatenate([raw_full[mm]] + [model_full[mn][mm] for mn in COMP_MODELS if mn != "RAW"])
        allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, [2, 98])
        rng = max(hi - lo, 1e-6)
        YLIM[mm] = (lo - 0.18 * rng, hi + 0.18 * rng)

    cl, ch = COMFORT[m]
    el, eh = EXTREME[m]

    curve_lines = {}
    rings = []
    labels = []
    for c, mname in enumerate(COMP_MODELS):
        ax = axes[c]
        ax.set_box_aspect(1.0)   # square tile
        for sp in ax.spines.values():
            sp.set_color(COMP_COLOR[mname])
            sp.set_linewidth(1.3)
        ax.axhline(cl, color="#2E9E44", ls="--", lw=0.7, alpha=0.9, zorder=1)
        ax.axhline(ch, color="#2E9E44", ls="--", lw=0.7, alpha=0.9, zorder=1)
        ax.axhline(el, color="#E53935", ls="-.", lw=0.7, alpha=0.9, zorder=1)
        ax.axhline(eh, color="#E53935", ls="-.", lw=0.7, alpha=0.9, zorder=1)
        ax.axvline(0, color="black", lw=0.6, alpha=0.4, zorder=1)
        curve_lines[mname], = ax.plot([], [], color=COMP_COLOR[mname], lw=1.5,
                                      alpha=1.0, zorder=3, solid_capstyle="round")
        ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=22, color=COMP_COLOR[mname],
                   edgecolor="white", linewidth=0.6, zorder=6)
        ring = ax.scatter([0.0], [model_full[mname][m][flag2_idx]], s=130, facecolors="none",
                          edgecolors=COMP_COLOR[mname], linewidth=1.6, zorder=7)
        ring.set_visible(False)
        rings.append(ring)

        ax.set_ylim(*YLIM[m])
        ax.set_xlim(disp_min, disp_max)
        ax.set_title(COMP_LABEL[mname], fontsize=11, fontweight="bold", pad=4,
                     color=COMP_COLOR[mname])
        ax.tick_params(labelsize=10)
        ax.set_xlabel("t (s)", fontsize=10)
        if c == 0:
            ax.set_ylabel(METRIC_TITLE[m], fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.grid(False)

    return {"curve_lines": curve_lines, "rings": rings, "labels": labels,
            "m": m, "t_rel": t_rel, "model_full": model_full}


def update_boxes(B, t):
    m = B["m"]
    t_rel = B["t_rel"]
    model_full = B["model_full"]
    nk = int(np.clip(np.searchsorted(t_rel, t), 0, len(t_rel) - 1))
    for mname in COMP_MODELS:
        B["curve_lines"][mname].set_data(t_rel[:nk + 1], model_full[mname][m][:nk + 1])
    for ring in B["rings"]:
        ring.set_visible(t >= 0)
    for lbl in B["labels"]:
        if lbl.get_text():
            lbl.set_visible(t >= 0)


def setup_map(ax, scn, d, range_geo, shared_lim):
    """One map: basemap + dynamic trajectories/rectangles + static points + INLINE labels."""
    ego_x, ego_y, t_rel = d["ego_x"], d["ego_y"], d["t_rel"]
    tgt_x, tgt_y, tgt_t_rel = d["tgt_x"], d["tgt_y"], d["tgt_t_rel"]
    flag2_idx = d["flag2_idx"]
    fp_x, fp_y = d["fp_x"], d["fp_y"]
    isect = d["isect"]
    cross_ego, cross_tgt = d["cross_ego"], d["cross_tgt"]
    closest_ego, closest_tgt = d["closest_ego"], d["closest_tgt"]

    # basemap
    fp, d_B, n_B, _ = range_geo
    mif.draw_lane_cross_section(ax, fp, d_B, n_B)

    # dynamic trails + rectangles
    ego_trail = PolyCollection([], facecolors="none", zorder=5)
    tgt_trail = PolyCollection([], facecolors="none", zorder=5)
    ax.add_collection(ego_trail)
    ax.add_collection(tgt_trail)
    ego_rect = Polygon([[0, 0]], closed=True, facecolor=C_EGO, edgecolor="black",
                       linewidth=1.0, zorder=10)
    tgt_rect = Polygon([[0, 0]], closed=True, facecolor=C_TGT, edgecolor="black",
                       linewidth=1.0, zorder=10)
    ax.add_patch(ego_rect)
    ax.add_patch(tgt_rect)

    other_rects = []
    other_trails = []
    for _ox, _oy, _ot, _oh, _olen, _owid in d["others"]:
        p = Polygon([[0, 0]], closed=True, facecolor=C_OTHER, edgecolor="black",
                    linewidth=0.8, zorder=9)
        ax.add_patch(p)
        p.set_visible(False)
        other_rects.append((p, _ox, _oy, _ot, _oh, _olen, _owid))
        tr = PolyCollection([], facecolors="none", zorder=4)
        ax.add_collection(tr)
        other_trails.append(tr)

    # flag=2 point (always visible)
    ax.scatter([fp_x], [fp_y], c="black", s=40, zorder=11, edgecolor="white", linewidth=0.8)

    # inline label offsets scaled to the map extent
    xrng = shared_lim[0][1] - shared_lim[0][0]
    yrng = shared_lim[1][1] - shared_lim[1][0]
    ox = 0.06 * xrng
    oy = 0.06 * yrng

    def lab(x, y, s, color, ha, va):
        return ax.text(x, y, s, ha=ha, va=va, fontsize=11, color=color, fontweight="bold",
                       zorder=14,
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                                 lw=0.8, alpha=0.5))

    m, fval = trigger_metric(d)
    f2_s = "flag=2 (ISMD) %s %.1f" % (_MNAME[m], fval)

    labels = []  # (artist, reveal_time)

    if scn["boundary"] and isect is not None:
        cx, cy = float(isect[0]), float(isect[1])
        t1_time = float(tgt_t_rel[cross_tgt])
        t2_time = float(t_rel[cross_ego])
        ax.scatter([cx], [cy], color="#E28E2C", edgecolor="black", linewidth=0.5,
                   s=46, zorder=11)
        L1 = lab(cx - ox, cy, r"$t_1$ (target) = %.2f s" % t1_time, C_TGT, "right", "center")
        L2 = lab(cx + ox, cy, r"$t_2$ (ego) = %.2f s" % t2_time, C_EGO, "left", "center")
        Lg = lab(cx, cy + 1.6 * oy, r"$t_2 - t_1$ = %.2f s" % (t2_time - t1_time),
                 "#333333", "center", "bottom")
        for L, tt in [(L1, t1_time), (L2, t2_time), (Lg, t2_time)]:
            L.set_visible(False)
            labels.append((L, tt))
    else:
        ex, ey = float(ego_x[closest_ego]), float(ego_y[closest_ego])
        tx, ty = float(tgt_x[closest_tgt]), float(tgt_y[closest_tgt])
        t1_time = float(tgt_t_rel[closest_tgt])
        t2_time = float(t_rel[closest_ego])
        ax.plot([ex, tx], [ey, ty], color="#E28E2C", ls="--", lw=1.0, alpha=0.8, zorder=8)
        ax.scatter([tx, ex], [ty, ey], color="#E28E2C", edgecolor="black", linewidth=0.5,
                   s=30, zorder=11)
        L1 = lab(tx - ox, ty, r"$t_1$ (target) = %.2f s" % t1_time, C_TGT, "right", "center")
        L2 = lab(ex + ox, ey, r"$t_2$ (ego) = %.2f s" % t2_time, C_EGO, "left", "center")
        mx, my = (ex + tx) / 2.0, (ey + ty) / 2.0
        dmin_v = d["dmin"] if np.isfinite(d["dmin"]) else 0.0
        Lg = lab(mx, my - 1.8 * oy, r"$t_2 - t_1$ = %.2f s" % (t2_time - t1_time),
                 "#333333", "center", "top")
        # dmin: horizontal leader line to the closest-approach midpoint; box left
        # edge aligned with the flag=2 label's left edge (x = fp_x + ox)
        Ld = lab(fp_x + ox, my, r"$d_{min}$ = %.2f m" % dmin_v, "#7A3E00", "left", "center")
        Ld_line, = ax.plot([fp_x + ox, mx], [my, my], color="#3E1F00", ls="--", lw=2.0,
                           alpha=1.0, zorder=8)
        for L, tt in [(L1, t1_time), (L2, t2_time), (Ld, t2_time),
                      (Ld_line, t2_time), (Lg, t2_time)]:
            L.set_visible(False)
            labels.append((L, tt))

    Lf = lab(fp_x + ox, fp_y, f2_s, "#333333", "left", "center")
    Lf.set_visible(False)
    labels.append((Lf, 0.0))

    title = "(a) Intersection interaction scenario" if scn["boundary"] else "(b) Proximity interaction scenario"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8, loc="left")
    ax.set_xlim(*shared_lim[0])
    ax.set_ylim(*shared_lim[1])
    ax.set_aspect("equal")
    ax.axis("off")

    a0 = max(float(t_rel[0]), -WIN_S)
    return {
        "ego_x": ego_x, "ego_y": ego_y, "t_rel": t_rel,
        "ego_heading": d["ego_heading"], "ego_len": d["ego_len"], "ego_width": d["ego_width"],
        "tgt_x": tgt_x, "tgt_y": tgt_y, "tgt_t_rel": tgt_t_rel,
        "tgt_heading": d["tgt_heading"], "tgt_len": d["tgt_len"], "tgt_width": d["tgt_width"],
        "ego_i0": int(np.searchsorted(t_rel, a0)),
        "tgt_i0": int(np.searchsorted(tgt_t_rel, a0)),
        "ego_trail": ego_trail, "tgt_trail": tgt_trail,
        "ego_rect": ego_rect, "tgt_rect": tgt_rect,
        "other_rects": other_rects, "other_trails": other_trails,
        "labels": labels,
    }


def update_map(S, t, state):
    ego_x, ego_y, t_rel = S["ego_x"], S["ego_y"], S["t_rel"]
    tgt_x, tgt_y, tgt_t_rel = S["tgt_x"], S["tgt_y"], S["tgt_t_rel"]
    ego_heading, tgt_heading = S["ego_heading"], S["tgt_heading"]
    ego_len, ego_width = S["ego_len"], S["ego_width"]
    tgt_len, tgt_width = S["tgt_len"], S["tgt_width"]

    k = int(np.clip(np.searchsorted(t_rel, t), 0, len(ego_x) - 1))
    j = int(np.clip(np.searchsorted(tgt_t_rel, t), 0, len(tgt_x) - 1))

    mif.update_ribbon(S["ego_trail"], ego_x, ego_y, ego_width, C_EGO, S["ego_i0"], k)
    mif.update_ribbon(S["tgt_trail"], tgt_x, tgt_y, tgt_width, C_TGT, S["tgt_i0"], j)
    S["ego_rect"].set_xy(mif.veh_corners(ego_x[k], ego_y[k], ego_heading[k], ego_len, ego_width))
    S["tgt_rect"].set_xy(mif.veh_corners(tgt_x[j], tgt_y[j], tgt_heading[j], tgt_len, tgt_width))

    # appear / disappear strictly by each vehicle's own data window — never freeze
    # outside it (the target's data typically ends before the animation window does)
    ego_on = t_rel[0] <= t <= t_rel[-1]
    tgt_on = tgt_t_rel[0] <= t <= tgt_t_rel[-1]
    S["ego_rect"].set_visible(ego_on)
    S["ego_trail"].set_visible(ego_on)
    S["tgt_rect"].set_visible(tgt_on)
    S["tgt_trail"].set_visible(tgt_on)

    for (_p, _ox, _oy, _ot, _oh, _olen, _owid), _tr in zip(S["other_rects"], S["other_trails"]):
        if _ot[0] <= t <= _ot[-1]:
            _j = int(np.clip(np.searchsorted(_ot, t), 0, len(_ot) - 1))
            _p.set_xy(mif.veh_corners(_ox[_j], _oy[_j], _oh[_j], _olen, _owid))
            _p.set_visible(True)
            mif.update_ribbon(_tr, _ox, _oy, _owid, C_OTHER, 0, _j)
        else:
            _p.set_visible(False)
            _tr.set_verts([])

    S["ego_rect"].set_edgecolor("black")
    S["ego_rect"].set_linewidth(1.0)
    S["tgt_rect"].set_edgecolor("black")
    S["tgt_rect"].set_linewidth(1.0)
    if state == "flag2":
        S["ego_rect"].set_edgecolor("#F4C20D")
        S["ego_rect"].set_linewidth(1.8)
    elif state == "t1":
        S["tgt_rect"].set_edgecolor("#E28E2C")
        S["tgt_rect"].set_linewidth(1.8)
    elif state == "t2":
        S["ego_rect"].set_edgecolor("#E28E2C")
        S["ego_rect"].set_linewidth(1.8)

    for L, tt in S["labels"]:
        L.set_visible(t >= tt)


def shared_extent(data, tags, range_geo):
    xs = []
    ys = []
    for tag in tags:
        d = data[tag]
        xs.append(d["ego_x"])
        ys.append(d["ego_y"])
        xs.append(d["tgt_x"])
        ys.append(d["tgt_y"])
    if range_geo is not None:
        xs.append(range_geo[3][:, 0])
        ys.append(range_geo[3][:, 1])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 + 10.0
    return ((cx - half, cx + half), (cy - half, cy + half))


def make_composite_gif(data, range_geo, smooth=False):
    u8 = data["U8"]
    c3 = data["C3"]
    scn_u8 = [s for s in mif.SCENARIOS if s["tag"] == "U8"][0]
    scn_c3 = [s for s in mif.SCENARIOS if s["tag"] == "C3"][0]
    shared_lim = shared_extent(data, ["U8", "C3"], range_geo)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8})
    fig = plt.figure(figsize=(17.8, 14.0), dpi=110)
    # three explicit horizontal bands; the maps band is shifted down a touch
    # (top/bottom model boxes keep their original positions)
    top = fig.add_gridspec(1, 7, wspace=0.20, left=0.05, right=0.985, top=0.96, bottom=0.8165)
    maps = fig.add_gridspec(1, 2, wspace=0.10, left=0.05, right=0.985, top=0.739, bottom=0.165)
    bot = fig.add_gridspec(1, 7, wspace=0.20, left=0.05, right=0.985, top=0.1735, bottom=0.03)

    top_axes = [fig.add_subplot(top[0, c]) for c in range(7)]
    ax_u8 = fig.add_subplot(maps[0, 0])
    ax_c3 = fig.add_subplot(maps[0, 1])
    bot_axes = [fig.add_subplot(bot[0, c]) for c in range(7)]

    fig.text(0.012, 0.888, "(a) Intersection interaction", rotation=90,
             ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.012, 0.102, "(b) Proximity interaction", rotation=90,
             ha="center", va="center", fontsize=12, fontweight="bold")

    du8_min = max(float(u8["t_rel"][0]), -WIN_S)
    du8_max = min(float(u8["t_rel"][-1]), WIN_S)
    dc3_min = max(float(c3["t_rel"][0]), -WIN_S)
    dc3_max = min(float(c3["t_rel"][-1]), WIN_S)

    B_top = setup_boxes(top_axes, u8, du8_min, du8_max)
    B_bot = setup_boxes(bot_axes, c3, dc3_min, dc3_max)
    S_u8 = setup_map(ax_u8, scn_u8, u8, range_geo, shared_lim)
    S_c3 = setup_map(ax_c3, scn_c3, c3, range_geo, shared_lim)

    # legend inside each map's top-left corner so it never overlaps the tick labels
    leg_handles = [
        Line2D([0], [0], color=C_EGO, lw=3, label="ego (blue)"),
        Line2D([0], [0], color=C_TGT, lw=3, label="target (orange)"),
    ]
    for axm in (ax_u8, ax_c3):
        axm.legend(handles=leg_handles, loc="upper left", fontsize=12,
                   framealpha=0.5, borderpad=0.6, handlelength=1.6)

    # ---- synchronized frame schedule (both scenarios share t=0 = flag=2) ----
    def key_times(d, scn):
        t_rel = d["t_rel"]
        tgt_t_rel = d["tgt_t_rel"]
        a0 = max(float(t_rel[0]), -WIN_S)
        a1 = min(float(t_rel[-1]), WIN_S)
        if scn["boundary"]:
            t1 = float(tgt_t_rel[d["cross_tgt"]])
            t2 = float(t_rel[d["cross_ego"]])
        else:
            t1 = float(tgt_t_rel[d["closest_tgt"]])
            t2 = float(t_rel[d["closest_ego"]])
        a0 = min(a0, 0.0)
        t1 = max(t1, 0.0)
        t2 = max(t2, t1)
        a1 = max(a1, t2)
        return a0, t1, t2, a1

    counts = ([26, 2, 18, 2, 18, 2, 22, 2] if smooth
              else [26, 20, 18, 14, 18, 16, 22, 14])

    def build(times):
        a0, t1, t2, a1 = times
        ts = []
        sts = []

        def phase(s, e, n, st):
            for f in range(n):
                fr = f / max(n - 1, 1)
                ts.append(s + (e - s) * fr)
                sts.append(st)

        phase(a0, 0.0, counts[0], "drive")
        phase(0.0, 0.0, counts[1], "flag2")
        phase(0.0, t1, counts[2], "drive")
        phase(t1, t1, counts[3], "t1")
        phase(t1, t2, counts[4], "drive")
        phase(t2, t2, counts[5], "t2")
        phase(t2, a1, counts[6], "drive")
        phase(a1, a1, counts[7], "drive")
        return ts, sts

    t_u8, states = build(key_times(u8, scn_u8))
    t_c3, _ = build(key_times(c3, scn_c3))
    frames = list(zip(t_u8, t_c3, states))

    def update(step):
        tu, tc, st = frames[step]
        update_map(S_u8, tu, st)
        update_map(S_c3, tc, st)
        update_boxes(B_top, tu)
        update_boxes(B_bot, tc)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    out_path = OUT_GIF if not smooth else OUT_GIF.replace(".gif", "_smooth.gif")
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("saved %s (%d frames)" % (out_path, len(frames)))
    return out_path


if __name__ == "__main__":
    data, _ = mif.load_scenarios()
    data = load_extra_models(data)
    range_geo = mif.load_range_poly()
    make_composite_gif(data, range_geo, smooth=True)    # smooth, no pauses
