#!/usr/bin/env python3
"""Waterfall plots per cluster: waveforms aligned on the P pick and sorted by RELOCATED DEPTH,
with the observed S pick and the model-predicted S-P overlaid.

The question these answer: within a cluster, does the S-P interval actually vary with the
depth hypoDD assigned, or is the depth spread an inversion artifact?

S-P depends on hypocentral distance and not at all on origin time, so it is an independent
check on relative depth. If a cluster's depth spread is real, its traces -- sorted by depth --
show S walking systematically later down the page, tracking the predicted curve. If the depth
spread came out of the inversion rather than the data, S sits at a constant lag while the
depth axis changes underneath it. That contrast is visible directly, without any statistic.

Each figure is one cluster at one station:
  left   waterfall, one trace per event, sorted by depth, aligned on P (t=0)
  right  observed S-P against relocated depth, with the model prediction

Usage:
    python full_catalog_pipeline/plot_cluster_sp_waterfall.py --array T2 --clusters 1 3 6 7
"""

# --- pipeline path bootstrap (added by the 2026-09 reorganization) ---------------
# Scripts live in stage subdirectories but import shared modules (catalog_paths, config,
# vels1d_model, lib.*) and occasionally each other. Python only puts a script's OWN
# directory on sys.path, so add the pipeline root and every stage directory. Invoke from
# the REPOSITORY root -- data paths inside these scripts are relative to it.
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# --- end bootstrap ---------------------------------------------------------------

import argparse
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from obspy import UTCDateTime
from scipy.signal import butter, sosfiltfilt, hilbert

import catalog_paths
import config
from hypodd_tune import RELOC_COLS
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp
from pyrocko import cake

SR = config.SAMPLE_RATE_HZ                 # 200 Hz -- do NOT hard-code, the archive sets it
WINDOW_PRE = 0.05
WINDOW_PAD = 0.35                          # seconds of record kept past the latest S pick
BAND = (20.0, 90.0)                        # same band the cross-correlation used
MIN_TRACES = 25
MAX_TRACES = 220

SERIES = ["#2a78d6", "#eb6834"]
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e4e3de"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def bandpass(x, sr=SR, band=BAND):
    sos = butter(4, [band[0] / (sr / 2), band[1] / (sr / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def load_picks(phase_dat, want_ids):
    """{event_id: {"origin": UTCDateTime, "P": {sta: tt}, "S": {sta: tt}}}"""
    out, cur = {}, None
    with open(phase_dat) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                eid = int(p[-1])
                cur = eid if eid in want_ids else None
                if cur is not None:
                    out[cur] = {"origin": UTCDateTime(int(p[1]), int(p[2]), int(p[3]),
                                                      int(p[4]), int(p[5]), float(p[6])),
                                "P": {}, "S": {}}
            elif cur is not None:
                q = line.split()
                if len(q) == 4 and q[3] in ("P", "S"):
                    out[cur][q[3]][q[0].split(".")[-1]] = float(q[1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, required=True)
    ap.add_argument("--stations", nargs="*", default=None)
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    reloc = pd.read_csv(catalog_paths.reloc(args.array), sep=r"\s+", header=None,
                        names=RELOC_COLS)
    sta = pd.read_csv(catalog_paths.station_sel(args.array), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = to_ps.transform(sta["lon"].values, sta["lat"].values)
    sta["x"], sta["y"] = sx, sy

    print("building traveltime grids ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    SUMMARY = []

    for cl in args.clusters:
        ids_file = os.path.join(work, f"{args.array.lower()}_cluster{cl}_event_ids.txt")
        if not os.path.exists(ids_file):
            print(f"cluster {cl}: no id list at {ids_file}")
            continue
        ids = {int(v) for v in open(ids_file).read().split()}
        picks = load_picks(catalog_paths.phase_dat(args.array), ids)
        ev = reloc[reloc["id"].isin(ids)].set_index("id")
        if ev.empty:
            continue

        # Rank stations by how many cluster members have BOTH a P and an S pick there.
        counts = {}
        for eid, p in picks.items():
            if eid not in ev.index:
                continue
            for s in set(p["P"]) & set(p["S"]):
                counts[s] = counts.get(s, 0) + 1
        # Rank by DISTANCE, not by pick count. S-P responds to depth as dz*(z/r), so its
        # sensitivity is greatest at the closest station and asymptotically nil at a far one:
        # a far station's flat S-P says nothing about whether the depths are right.
        ctr = ev[["lat", "lon"]].median()
        cxx, cyy = to_ps.transform(ctr["lon"], ctr["lat"])
        sta_ok = sta[sta["code"].map(lambda c: counts.get(c, 0)) >= MIN_TRACES].copy()
        sta_ok["dist_km"] = np.hypot(sta_ok["x"] - cxx, sta_ok["y"] - cyy) / 1000.0
        order = list(sta_ok.sort_values("dist_km")["code"])
        stations = args.stations or order[:2]
        dist_of = dict(zip(sta_ok["code"], sta_ok["dist_km"]))

        for station in stations:
            if counts.get(station, 0) < MIN_TRACES:
                print(f"cluster {cl} / {station}: only {counts.get(station,0)} P+S events, skipping")
                continue
            srow = sta[sta["code"] == station]
            if srow.empty:
                continue
            srow = srow.iloc[0]
            members = [e for e in picks if e in ev.index
                       and station in picks[e]["P"] and station in picks[e]["S"]]
            members.sort(key=lambda e: ev.loc[e, "depth"])
            if len(members) > MAX_TRACES:      # thin evenly so the depth range is preserved
                members = list(np.array(members)[np.linspace(0, len(members) - 1,
                                                             MAX_TRACES).astype(int)])

            # Size the window from the actual S-P times at THIS station, so a distant
            # station's later S still lands inside the record.
            sp_here = [picks[e]["S"][station] - picks[e]["P"][station] for e in members]
            window_post = float(np.percentile(sp_here, 99)) + WINDOW_PAD
            cache = RollingDayCache(day_index, station)
            traces, depths, sp_obs, sp_pred = [], [], [], []
            for eid in members:
                pt = picks[eid]["origin"] + picks[eid]["P"][station]
                row = {"primary_date": date(pt.year, pt.month, pt.day).isoformat()}
                data, status = extract_window(cache, row, pt - WINDOW_PRE, pt + window_post)
                cache.evict_before(row["primary_date"])
                if data is None or not data["complete"]:
                    continue
                # Horizontals carry S far better than the vertical; use their envelope so a
                # polarity flip between events cannot cancel the arrival visually.
                h = np.hypot(bandpass(data["N"].astype(float)), bandpass(data["E"].astype(float)))
                env = np.abs(hilbert(h))
                if env.max() <= 0 or len(env) < 0.8 * (WINDOW_PRE + window_post) * SR:
                    continue
                e = ev.loc[eid]
                ex, ey = to_ps.transform(e["lon"], e["lat"])
                r = np.hypot(ex - srow["x"], ey - srow["y"]) / 1000.0
                z = float(e["depth"])
                traces.append(env / env.max())
                depths.append(z)
                sp_obs.append(picks[eid]["S"][station] - picks[eid]["P"][station])
                sp_pred.append(float(interp(zs, rs, gs, np.array([z]), np.array([r]))[0]
                                     - interp(zs, rs, gp, np.array([z]), np.array([r]))[0]))
            if len(traces) < MIN_TRACES:
                print(f"cluster {cl} / {station}: only {len(traces)} usable waveforms, skipping")
                continue

            # Trim/pad to the nominal window instead of the shortest trace: one short read
            # would otherwise truncate every trace, and at 200 Hz that cropped the plot before
            # the S arrival even appeared.
            n = int(round((WINDOW_PRE + window_post) * SR))
            M = np.vstack([np.pad(t[:n], (0, max(0, n - len(t)))) for t in traces])
            t_axis = np.arange(n) / SR - WINDOW_PRE
            depths = np.array(depths); sp_obs = np.array(sp_obs); sp_pred = np.array(sp_pred)

            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 7.5),
                                          gridspec_kw={"width_ratios": [1.5, 1]})
            ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=0.6,
                      extent=[t_axis[0], t_axis[-1], len(M) - 0.5, -0.5],
                      interpolation="nearest")
            ax.plot(sp_obs, np.arange(len(M)), ".", color=SERIES[1], ms=3.5,
                    label="observed S pick")
            ax.plot(sp_pred, np.arange(len(M)), "-", color=SERIES[0], lw=1.8,
                    label="predicted S-P for that depth")
            ax.axvline(0, color=INK, lw=1)
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.set_xlabel("time after P pick (s)")
            ax.set_ylabel("event, sorted by relocated depth  (shallow → deep)")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
            ax.grid(False)
            tick = np.linspace(0, len(M) - 1, 6).astype(int)
            ax.set_yticks(tick)
            ax.set_yticklabels([f"{depths[i]:.2f}" for i in tick])
            ax.set_title(f"{args.array} cluster {cl} at {station} — envelope of the horizontals,\n"
                         f"aligned on P, sorted by depth (n={len(M)})", fontsize=10, color=INK)

            ax2.scatter(sp_obs * 1000, depths, s=9, color=SERIES[1], alpha=0.6,
                        label="observed")
            o = np.argsort(depths)
            ax2.plot(sp_pred[o] * 1000, depths[o], color=SERIES[0], lw=2, label="predicted")
            if len(depths) > 5 and depths.ptp() > 0.02:
                # Slope of S-P against depth, with the standard error of the fit. The
                # prediction is what the velocity model REQUIRES if the depth differences are
                # real; a depth/origin-time trade-off inflates depth without moving S-P, so
                # it shows up as an observed slope well below the predicted one.
                sl, _ = np.polyfit(depths, sp_obs * 1000, 1)
                pl, _ = np.polyfit(depths, sp_pred * 1000, 1)
                resid = sp_obs * 1000 - np.polyval(np.polyfit(depths, sp_obs * 1000, 1), depths)
                se = (np.std(resid, ddof=2) / (np.std(depths) * np.sqrt(len(depths))))
                ratio = sl / pl if pl else np.nan
                ax2.set_title(f"observed {sl:+.0f} ± {se:.0f} ms/km   vs   predicted {pl:+.0f} ms/km"
                              f"   (ratio {ratio:.2f})\n"
                              f"ratio ≈ 1 => the depth differences are carried by the data",
                              fontsize=10, color=INK)
                SUMMARY.append((cl, station, dist_of.get(station, np.nan), len(depths),
                                depths.min(), depths.max(), sl, se, pl, ratio))
            ax2.invert_yaxis()
            ax2.set_xlabel("S-P (ms)")
            ax2.set_ylabel("relocated depth (km)")
            ax2.legend(fontsize=8, frameon=False)

            fig.tight_layout()
            out = os.path.join(work, f"{args.array.lower()}_cluster{cl}_{station}_sp_waterfall.png")
            fig.savefig(out, dpi=145, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}   (depth range {depths.min():.2f}-{depths.max():.2f} km)")


    if SUMMARY:
        print(f"\n{'clus':>4s} {'sta':>5s} {'dist':>6s} {'n':>4s} {'depth range':>13s} "
              f"{'observed':>14s} {'predicted':>10s} {'ratio':>6s}")
        for cl, st, di, n, z0, z1, sl, se, pl, ra in SUMMARY:
            print(f"{cl:4d} {st:>5s} {di:5.1f}km {n:4d} {z0:5.2f}-{z1:5.2f} km "
                  f"{sl:+8.0f}±{se:<4.0f} ms/km {pl:+9.0f} {ra:6.2f}")


if __name__ == "__main__":
    main()
