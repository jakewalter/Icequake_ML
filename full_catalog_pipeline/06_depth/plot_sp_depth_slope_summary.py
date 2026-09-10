#!/usr/bin/env python3
"""Does each cluster's depth spread show up in its S-P times? One figure, every cluster,
every station.

S-P depends on hypocentral distance and not on origin time. So if a cluster's relocated depth
spread is real, S-P must increase with depth at the rate the velocity model requires. If the
spread came from the depth/origin-time trade-off instead -- hypoDD moving an event deeper and
the origin time absorbing it -- S-P does not move at all, and the observed slope falls far
below the predicted one.

The ratio observed/predicted is therefore the diagnostic: ~1 means the data carries the depth
differences, ~0 means the inversion invented them.

Sensitivity matters and is why this plots against it. dS-P/dz scales as z/r, so a station far
from a cluster has an almost flat predicted slope and its ratio is meaningless -- a big ratio
there is a small number divided by a smaller one. Only station-cluster pairs whose predicted
slope exceeds MIN_PREDICTED_SLOPE are drawn filled; the rest are shown hollow and excluded
from the verdict.

This needs no waveforms (picks, depths and the traveltime grid suffice), so it is much faster
than the per-cluster waterfalls in plot_cluster_sp_waterfall.py and covers every station.

Usage:
    python full_catalog_pipeline/plot_sp_depth_slope_summary.py --array T2
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from pyrocko import cake

import catalog_paths
from hypodd_tune import RELOC_COLS
from plot_cluster_sp_waterfall import load_picks
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp
from vels1d_model import bed_markers

MIN_EVENTS = 20
MIN_DEPTH_RANGE_KM = 0.05
MIN_PREDICTED_SLOPE = 40.0     # ms/km below which the station cannot resolve depth at all

SERIES = ["#2a78d6", "#eb6834"]
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e4e3de"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()
    work = catalog_paths.work_dir(args.array)
    ice_base = bed_markers(args.array)["ice_base"]

    reloc = pd.read_csv(catalog_paths.reloc(args.array), sep=r"\s+", header=None,
                        names=RELOC_COLS)
    sta = pd.read_csv(catalog_paths.station_sel(args.array), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sta["x"], sta["y"] = to_ps.transform(sta["lon"].values, sta["lat"].values)

    print("building traveltime grids ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])

    rows = []
    for cl in range(20):
        f = os.path.join(work, f"{args.array.lower()}_cluster{cl}_event_ids.txt")
        if not os.path.exists(f):
            continue
        ids = {int(v) for v in open(f).read().split()}
        picks = load_picks(catalog_paths.phase_dat(args.array), ids)
        ev = reloc[reloc["id"].isin(ids)].set_index("id")
        for _, s in sta.iterrows():
            code = s["code"]
            z, sp, r = [], [], []
            for eid, p in picks.items():
                if eid not in ev.index or code not in p["P"] or code not in p["S"]:
                    continue
                e = ev.loc[eid]
                ex, ey = to_ps.transform(e["lon"], e["lat"])
                z.append(float(e["depth"]))
                sp.append(p["S"][code] - p["P"][code])
                r.append(np.hypot(ex - s["x"], ey - s["y"]) / 1000.0)
            z, sp, r = np.array(z), np.array(sp) * 1000.0, np.array(r)
            if len(z) < MIN_EVENTS or z.ptp() < MIN_DEPTH_RANGE_KM:
                continue
            pred = (interp(zs, rs, gs, z, r) - interp(zs, rs, gp, z, r)) * 1000.0
            ok = np.isfinite(pred)
            if ok.sum() < MIN_EVENTS:
                continue
            z, sp, pred = z[ok], sp[ok], pred[ok]
            obs_slope = np.polyfit(z, sp, 1)[0]
            pred_slope = np.polyfit(z, pred, 1)[0]
            resid = sp - np.polyval(np.polyfit(z, sp, 1), z)
            se = np.std(resid, ddof=2) / (np.std(z) * np.sqrt(len(z)))
            rows.append(dict(cluster=cl, station=code, n=len(z),
                             dist_km=float(np.median(r)), z_lo=z.min(), z_hi=z.max(),
                             obs=obs_slope, se=se, pred=pred_slope,
                             ratio=obs_slope / pred_slope if pred_slope else np.nan,
                             resolving=abs(pred_slope) >= MIN_PREDICTED_SLOPE))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no cluster/station pairs met the thresholds")
        return
    csv = os.path.join(work, f"{args.array.lower()}_sp_depth_slopes.csv")
    df.to_csv(csv, index=False)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.6),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: observed vs predicted slope, 1:1 line ----
    res, non = df[df.resolving], df[~df.resolving]
    ax.errorbar(res["pred"], res["obs"], yerr=res["se"], fmt="o", color=SERIES[0], ms=7,
                lw=0, elinewidth=1.2, ecolor=SERIES[0], capsize=2,
                label=f"station resolves depth (predicted ≥ {MIN_PREDICTED_SLOPE:.0f} ms/km)")
    ax.errorbar(non["pred"], non["obs"], yerr=non["se"], fmt="o", mfc="none", ms=7, lw=0,
                mec=MUTED, elinewidth=1, ecolor=MUTED, capsize=2,
                label="station too far to resolve depth — ratio meaningless")
    lim = [0, max(df["pred"].max(), df["obs"].max()) * 1.1]
    ax.plot(lim, lim, color=INK, lw=1.2, ls="--", label="1:1 (data carries the depth spread)")
    for _, r in df.iterrows():
        ax.annotate(f"{int(r.cluster)}·{r.station}", xy=(r["pred"], r["obs"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7.5,
                    color=SERIES[0] if r.resolving else MUTED)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted dS-P/dz (ms/km) — what the velocity model requires")
    ax.set_ylabel("observed dS-P/dz (ms/km)")
    ax.set_title("If a cluster's depth spread is real, it lands on the 1:1 line",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)

    # ---- right: one number per cluster, from its BEST-RESOLVING station ----
    # Sensitivity varies enormously between stations -- for cluster 0, JULA at 1.1 km demands
    # +253 ms/km while WICH at 4.0 km demands only +20. A ratio computed at a low-sensitivity
    # station is a small number over a smaller one and says nothing either way, so each
    # cluster is judged by the station whose predicted slope is largest.
    df["abs_pred"] = df["pred"].abs()
    r2 = (df.sort_values("abs_pred", ascending=False)
            .groupby("cluster", as_index=False).first()
            .sort_values("cluster"))
    ypos = np.arange(len(r2))
    ax2.errorbar(r2["ratio"], ypos, xerr=(r2["se"] / r2["pred"]).abs(), fmt="o",
                 color=SERIES[0], ms=7, lw=0, elinewidth=1.2, capsize=2)
    ax2.axvline(1.0, color=INK, ls="--", lw=1.2)
    ax2.axvline(0.0, color=MUTED, ls=":", lw=1)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"cluster {int(r.cluster)} · {r.station}  "
                         f"({r.z_lo:.2f}–{r.z_hi:.2f} km, n={int(r.n)})" for _, r in r2.iterrows()],
                        fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("observed / predicted slope")
    ax2.set_title("Judged at each cluster's most depth-sensitive station\n"
                  "ratio ≈ 1: depths carried by the data;  ≈ 0: depth/origin-time trade-off",
                  fontsize=11, color=INK)
    ax2.set_xlim(-0.4, 2.6)
    worst = r2.loc[r2["ratio"].idxmin()]
    ax2.annotate(f"cluster {int(worst.cluster)}: S-P flat across a\n"
                 f"{1000*(worst.z_hi-worst.z_lo):.0f} m depth range — artifact",
                 xy=(worst["ratio"], list(r2["cluster"]).index(worst["cluster"])),
                 xytext=(0.15, 0.12), textcoords="axes fraction", fontsize=9, color=SERIES[1],
                 arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1.2))

    fig.suptitle(f"{args.array} — does each cluster's depth spread appear in its S-P times?  "
                 f"(measured ice base {ice_base:.2f} km)", fontsize=13, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(work, f"{args.array.lower()}_sp_depth_slope_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}\nwrote {csv}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
