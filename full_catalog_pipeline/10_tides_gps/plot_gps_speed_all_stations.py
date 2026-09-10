#!/usr/bin/env python3
"""GPS speed at all 7 Thwaites stations: records, spectra, and the tidal band.

Small multiples rather than one crowded axis: the stations differ in mean speed by ~5x
(T01C std 0.50 vs T02A 0.10), so overlaying them on shared limits would flatten most of
them into the baseline.

Column choice is immaterial and was checked, not assumed: cache_disp_v3's vel_sm is a 2 h
per-segment Savitzky-Golay of vel_raw, and its gain across 20-30 h is 0.985-1.002 (10-14 h:
0.974-0.994). Band correlations against the tide agree to +-0.024 and phases to +-8 deg
between the two columns, so vel_sm is used and nothing depends on that.

Panel set:
  A  per-station speed record, 14-day envelope, all on a common time axis
  B  Lomb-Scargle periodogram per station with the tidal bands marked (scipy's, normalized;
     handles the gaps properly -- ~20% of samples are missing and an FFT would alias them)
  C  the 20-30 h band-passed speed for one fortnight, all stations stacked
  D  diurnal-band amplitude per station against distance from the ocean

Usage:
    python full_catalog_pipeline/plot_gps_speed_all_stations.py
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
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.ndimage import uniform_filter1d

from gps_tide_admittance import nearest_wet, tide_at, STATIONS, GPS_DIR
from plot_low_tide_speedup import bandpass

ORDER = ["T02A", "T02B", "T02C", "T01A", "T01B", "T01C", "T01D"]
C_T1, C_T2 = "#2a78d6", "#eb6834"
C_TIDE = "#52514e"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedddA", "#fcfcfb"
BAND = (20.0, 30.0)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)


def col(s):
    return C_T2 if s.startswith("T02") else C_T1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="full_catalog_pipeline/artifacts/full_run/T2_v5")
    ap.add_argument("--field", default="vel_sm")
    args = ap.parse_args()

    ds = nc.Dataset("CATS2008_v2023.nc")
    G, meta = {}, {}
    for s in ORDER:
        g = pd.read_parquet(f"{GPS_DIR}/{s}.parquet")
        y = g[args.field]
        ok = y.notna().values
        t = g.index[ok].tz_localize(None)
        k, dkm, _, _ = nearest_wet(ds, *STATIONS[s])
        z = tide_at(ds, k, t)
        d = (pd.DataFrame({"v": y.values[ok], "z": z}, index=t)
             .resample("1h").mean().interpolate(limit=3).dropna())
        d["vb"] = bandpass(d["v"].values, *BAND)
        d["zb"] = bandpass(d["z"].values, *BAND)
        G[s] = d
        meta[s] = dict(dist_km=dkm, n=len(d), valid=ok.mean(),
                       std=float(d["v"].std()), band_std=float(d["vb"].std()),
                       band_r=float(np.corrcoef(d["vb"], d["zb"])[0, 1]))
        print(f"{s}  {dkm:5.0f} km  n={len(d):5d}  valid={ok.mean():.0%}  "
              f"std={d['v'].std():.4f}  band std={d['vb'].std():.4f}  "
              f"band r={meta[s]['band_r']:+.3f}")

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 1.7, 1.5, 1.1],
                          hspace=0.42, wspace=0.20)

    # ---- A: per-station records, small multiples on a shared time axis -------------
    sub = gs[0, :].subgridspec(7, 1, hspace=0.0)
    t0, t1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-25")
    for i, s in enumerate(ORDER):
        ax = fig.add_subplot(sub[i])
        d = G[s]
        v = d["v"].values
        med = np.median(v)
        mad = np.median(np.abs(v - med)) or v.std()
        lo_y, hi_y = med - 6 * 1.4826 * mad, med + 6 * 1.4826 * mad
        n_clip = int(((v < lo_y) | (v > hi_y)).sum())
        ax.plot(d.index, v, color=col(s), lw=0.5, alpha=0.55, zorder=3)
        ax.plot(d.index, uniform_filter1d(v, 24 * 14), color=INK, lw=1.4, zorder=4)
        ax.set_ylim(lo_y, hi_y)
        ax.set_xlim(t0, t1)
        if n_clip:
            ax.annotate(f"{n_clip} pts off-scale", (0.005, 0.12),
                        xycoords="axes fraction", fontsize=6.5, color=MUTED)
        ax.set_ylabel(s, fontsize=8.5, color=col(s), rotation=0, ha="right", va="center",
                      labelpad=16)
        ax.set_yticks([])
        style(ax)
        ax.grid(False)
        if i < 6:
            ax.set_xticklabels([])
        ax.annotate(f"{meta[s]['dist_km']:.0f} km · robust sd "
                    f"{1.4826 * mad:.3f} · raw sd {meta[s]['std']:.3f}",
                    (0.995, 0.78), xycoords="axes fraction", fontsize=6.5,
                    color=MUTED, ha="right")
        if i == 0:
            ax.set_title("A · speed record per station (thin) with 14-day mean (dark). "
                         "y-limits are median ±6·MAD per station — a few spikes were "
                         "otherwise flattening every trace.",
                         fontsize=11, color=INK, loc="left")

    # ---- B: Lomb-Scargle spectra ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for s in ORDER:
        d = G[s]
        tt = (d.index - d.index[0]).total_seconds().values / 3600.0
        f = np.linspace(1 / 60.0, 1 / 6.0, 4000)          # cycles/hour, 6-60 h
        yv = d["v"].values - d["v"].mean()
        p = lombscargle(tt, yv, 2 * np.pi * f, normalize=True)
        ax.semilogy(1 / f, p, color=col(s), lw=0.9, alpha=0.75, label=s)
    for P, lbl in ((12.42, "M2"), (23.93, "K1"), (25.82, "O1")):
        ax.axvline(P, color=INK, lw=0.8, ls="--", alpha=0.5)
        ax.annotate(lbl, (P, 1.02), xycoords=("data", "axes fraction"), fontsize=8,
                    color=INK, ha="center")
    ax.axvspan(BAND[0], BAND[1], color=C_TIDE, alpha=0.10, zorder=0)
    ax.set_xlabel("period (hours)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("Lomb-Scargle power", fontsize=9.5, color=MUTED)
    ax.set_title("B · spectra — power concentrates in the DIURNAL band, not at M2",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    # ---- B2: the tide's own spectrum, for comparison --------------------------------
    ax = fig.add_subplot(gs[1, 1])
    d = G["T02A"]
    tt = (d.index - d.index[0]).total_seconds().values / 3600.0
    f = np.linspace(1 / 60.0, 1 / 6.0, 4000)
    zv = d["z"].values - d["z"].mean()
    ax.semilogy(1 / f, lombscargle(tt, zv, 2 * np.pi * f, normalize=True),
                color=C_TIDE, lw=1.3, label="CATS2008 tide")
    for P, lbl in ((12.42, "M2"), (23.93, "K1"), (25.82, "O1")):
        ax.axvline(P, color=INK, lw=0.8, ls="--", alpha=0.5)
        ax.annotate(lbl, (P, 1.02), xycoords=("data", "axes fraction"), fontsize=8,
                    color=INK, ha="center")
    ax.axvspan(BAND[0], BAND[1], color=C_TIDE, alpha=0.10, zorder=0)
    ax.set_xlabel("period (hours)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("Lomb-Scargle power", fontsize=9.5, color=MUTED)
    ax.set_title("B2 · the FORCING: tide is diurnal-dominant, M2 is weak\n"
                 "(so the GPS spectrum in B mirrors the tide)",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    # ---- C: band-passed speed, all stations stacked over one fortnight -------------
    ax = fig.add_subplot(gs[2, :])
    w0, w1 = pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-24")
    d0 = G["T02A"]
    m0 = (d0.index >= w0) & (d0.index < w1)
    zs = (d0.zb[m0] - d0.zb[m0].mean()) / d0.zb[m0].std()
    ax.fill_between(d0.index[m0], -1, len(ORDER), where=(zs.values < 0),
                    color=C_TIDE, alpha=0.10, zorder=0, label="LOW tide half-cycles")
    for i, s in enumerate(ORDER):
        d = G[s]
        m = (d.index >= w0) & (d.index < w1)
        if m.sum() < 10:
            continue
        v = d.vb[m]
        ax.plot(d.index[m], np.clip(v / v.std(), -3, 3) * 0.26 + (len(ORDER) - 1 - i),
                color=col(s), lw=1.3, zorder=3)
        ax.annotate(f"{s}  r={meta[s]['band_r']:+.2f}", (w0, len(ORDER) - 1 - i + 0.42),
                    fontsize=7.5, color=col(s))
    ax.plot(d0.index[m0], np.clip(zs, -3, 3) * 0.26 + len(ORDER), color=INK, lw=1.6, zorder=4)
    ax.annotate("ocean tide", (w0, len(ORDER) + 0.42), fontsize=8, color=INK,
                fontweight="bold")
    ax.set_ylim(-0.7, len(ORDER) + 0.9)
    ax.set_yticks([])
    ax.set_xlim(w0, w1)
    ax.set_title("C · 20–30 h band-passed speed, each scaled to unit variance and clipped "
                 "at ±3σ (shape only — amplitude NOT comparable between rows)",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # ---- D: diurnal-band amplitude vs distance -------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    for s in ORDER:
        ax.scatter(meta[s]["dist_km"], meta[s]["band_std"], s=90, color=col(s), zorder=4)
        ax.annotate(s, (meta[s]["dist_km"], meta[s]["band_std"]), fontsize=7.5,
                    color=MUTED, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("distance to nearest wet cell (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("diurnal-band speed amplitude\n(std of 20–30 h band)",
                  fontsize=9.5, color=MUTED)
    ax.set_title("D · absolute band amplitude vs distance", fontsize=11.5, color=INK,
                 loc="left")
    style(ax)

    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    ax.set_title("station summary", fontsize=11.5, color=INK, loc="left")
    ax.text(0.0, 0.86, f"{'sta':6s}{'km':>6s}{'valid':>8s}{'std':>9s}{'band std':>10s}"
                       f"{'band r':>9s}", fontsize=8.5, color=INK, family="monospace",
            transform=ax.transAxes, fontweight="bold")
    yy = 0.74
    for s in ORDER:
        m = meta[s]
        ax.text(0.0, yy, f"{s:6s}{m['dist_km']:6.0f}{m['valid']:8.0%}{m['std']:9.4f}"
                         f"{m['band_std']:10.4f}{m['band_r']:+9.3f}",
                fontsize=8.5, color=col(s), family="monospace", transform=ax.transAxes)
        yy -= 0.105

    fig.suptitle("Thwaites GPS speed at all 7 stations — records, spectra, "
                 "and the diurnal tidal band", fontsize=14, color=INK, y=0.995)
    out = os.path.join(args.out_dir, "gps_speed_all_stations.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    pd.DataFrame(meta).T.to_csv(os.path.join(args.out_dir, "gps_speed_all_stations.csv"))
    print(f"\nwrote {out}")
    print(f"wrote {os.path.join(args.out_dir, 'gps_speed_all_stations.csv')}")


if __name__ == "__main__":
    main()
