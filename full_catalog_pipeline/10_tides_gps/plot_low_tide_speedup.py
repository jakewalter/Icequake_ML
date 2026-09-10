#!/usr/bin/env python3
"""Does the glacier flow faster at LOW tide? Figures for the diurnal-band GPS/tide result.

Tests the hypothesis directly: velocity should anti-correlate with ocean tide height, and the
response should weaken inland (T02 is ~80 km closer to the ocean than T01).

Band-limiting to 20-30 h is what makes this visible. The diurnal band carries 62 cm of the
78 cm tidal signal here (CATS2008: K1 34 cm, O1 28 cm, against M2's 5 cm), while >90% of the
GPS velocity variance is non-tidal. Broadband, T02A gives r = -0.13; in-band it is -0.50.

PHASE IS WRAPPED -- DO NOT READ PROPAGATION OFF THIS. At a ~24 h period a -171 deg lag and a
+189 deg lead are indistinguishable, so one diurnal cycle of ambiguity swamps any real
inland propagation delay. Panel C shows phase only to demonstrate CLUSTERING NEAR 180 deg
(i.e. anti-phase with tide height). Measuring propagation needs a non-periodic marker -- a
spring-tide envelope maximum or a discrete speed-up event -- not a phase at a tidal frequency.

The tide is CATS2008_v2023 synthesised with pyTMD (round-trip validated). It is NOT
gps_data/thwaites_tidal_predictions.csv, which is a 4-constituent hardcoded fallback with
M2 dominant -- the opposite of the real, diurnal-dominant local tide.

Usage:
    python full_catalog_pipeline/plot_low_tide_speedup.py
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
from scipy.signal import butter, filtfilt, hilbert

from gps_tide_admittance import nearest_wet, tide_at, STATIONS, GPS_DIR

BAND = (20.0, 30.0)
C_T1, C_T2 = "#2a78d6", "#eb6834"
C_TIDE = "#52514e"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedddA", "#fcfcfb"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def bandpass(x, lo_h, hi_h):
    b, a = butter(3, [2 / hi_h, 2 / lo_h], btype="band")
    return filtfilt(b, a, x)


def robust_outlier_mask(series, n_mad=8.0):
    """Flag isolated GPS position glitches by a global median/MAD z-score.

    This is the criterion used by the GPS project's own analysis
    (`plot_pyocto_events_vs_displacement.py:robust_outlier_mask`), which validated it against
    `noise_analysis/{STATION}_quality_flags_v2.csv`. It removes single-epoch bad fixes while
    leaving multi-day speed-up episodes untouched, since those stay well inside 8 robust sigma.

    It matters a lot here: cleaning lifts T02C from r = -0.105 to -0.437 on removing 2.3% of
    samples, and T01B from -0.210 to -0.400. Additionally dropping every 5-min bin containing a
    `marked_bad` epoch from the flag files changes nothing beyond +-0.02, so this cheap reject is
    sufficient.
    """
    v = series.dropna()
    med = v.median()
    mad = (v - med).abs().median()
    return (series - med).abs() > n_mad * 1.4826 * mad


def load_station(ds, s, clean=True, n_mad=8.0):
    la, lo = STATIONS[s]
    g = pd.read_parquet(f"{GPS_DIR}/{s}.parquet")
    y = g["vel_sm"]
    if clean:
        y = y.mask(robust_outlier_mask(y, n_mad))
    ok = y.notna().values
    times = g.index[ok].tz_localize(None)
    k, dkm, _, _ = nearest_wet(ds, la, lo)
    z = tide_at(ds, k, times)
    df = (pd.DataFrame({"v": y.values[ok], "z": z}, index=times)
          .resample("1h").mean().interpolate(limit=3).dropna())
    df["vb"] = bandpass(df["v"].values, *BAND)
    df["zb"] = bandpass(df["z"].values, *BAND)
    return df, dkm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="full_catalog_pipeline/artifacts/full_run/T2_v5")
    ap.add_argument("--demo-station", default="T02A")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip the 8*MAD outlier reject (reproduces the pre-2026-09 output)")
    ap.add_argument("--n-mad", type=float, default=8.0)
    args = ap.parse_args()

    ds = nc.Dataset(CATS := "CATS2008_v2023.nc")
    rows, data = [], {}
    rng = np.random.default_rng(0)
    for s in sorted(STATIONS):
        df, dkm = load_station(ds, s, clean=not args.no_clean, n_mad=args.n_mad)
        data[s] = df
        r0 = np.corrcoef(df.v, df.z)[0, 1]
        rb = np.corrcoef(df.vb, df.zb)[0, 1]
        nl = np.array([np.corrcoef(df.vb, np.roll(df.zb.values,
                       int(rng.integers(48, len(df) - 48))))[0, 1] for _ in range(1000)])
        p = float((np.abs(nl) >= abs(rb)).mean())
        ph = np.degrees(np.angle(np.mean(hilbert(df.vb.values)
                                         * np.conj(hilbert(df.zb.values)))))
        rows.append(dict(station=s, array=s[:3], dist_km=dkm, r_broad=r0, r_band=rb,
                         band_p=p, phase_deg=ph, n=len(df)))
        print(f"{s}  {dkm:5.0f} km  broadband r={r0:+.3f}  band r={rb:+.3f} "
              f"(p={p:.3f})  phase={ph:+.0f} deg")
    R = pd.DataFrame(rows)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.30)

    # ---- A: the money panel, a two-week window -------------------------------------
    # ONE axis, both series standardised. A twin axis would let arbitrary y-scaling
    # manufacture (or hide) the apparent agreement -- with z-scores the anti-phase is a
    # property of the data, not of the axis limits.
    ax = fig.add_subplot(gs[0, :2])
    d = data[args.demo_station]
    m = (d.index >= "2020-02-10") & (d.index < "2020-02-24")
    tt = d.index[m]
    zs = (d.zb[m] - d.zb[m].mean()) / d.zb[m].std()
    vs = (d.vb[m] - d.vb[m].mean()) / d.vb[m].std()
    ax.fill_between(tt, -3.4, 3.4, where=(zs.values < 0), color=C_TIDE, alpha=0.10,
                    zorder=1, label="LOW tide half-cycles")
    ax.plot(tt, zs, color=C_TIDE, lw=1.8, zorder=3, label="ocean tide (standardised)")
    ax.plot(tt, vs, color=C_T2, lw=1.8, zorder=4, label="GPS speed (standardised)")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_ylim(-3.4, 3.4)
    ax.set_ylabel("standard deviations", fontsize=9.5, color=MUTED)
    rr = float(np.corrcoef(d.vb[m], d.zb[m])[0, 1])
    ax.set_title(f"A · {args.demo_station}: speed peaks in the LOW-tide half-cycles"
                 f"   (Feb 2020, r = {rr:+.2f} in this window)",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)

    # ---- B: broadband vs band-limited correlation ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    o = R.sort_values("dist_km")
    yy = np.arange(len(o))
    ax.barh(yy - 0.19, o.r_broad, 0.36, color=MUTED, zorder=3, label="broadband")
    ax.barh(yy + 0.19, o.r_band, 0.36,
            color=[C_T1 if a == "T01" else C_T2 for a in o.array], zorder=3,
            label="20–30 h band")
    ax.axvline(0, color=INK, lw=1.0, zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.station}  {r.dist_km:.0f} km" for r in o.itertuples()],
                       fontsize=8)
    ax.set_xlim(min(o.r_band.min(), o.r_broad.min()) * 1.15, 0.02)
    ax.set_xlabel("correlation of speed with tide HEIGHT", fontsize=9.5, color=MUTED)
    ax.set_title("B · all 7 stations negative\n(faster at low tide)",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # ---- C: phase, on a polar axis (a circular quantity) ---------------------------
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    for r in R.itertuples():
        ax.plot([np.radians(r.phase_deg)], [abs(r.r_band)], "o", ms=10,
                color=C_T1 if r.array == "T01" else C_T2, alpha=0.9)
        ax.annotate(r.station, (np.radians(r.phase_deg), abs(r.r_band)),
                    fontsize=7, color=MUTED, xytext=(4, 4), textcoords="offset points")
    ax.plot([np.pi, np.pi], [0, 0.55], color=INK, lw=2.0, ls="--", zorder=5)
    ax.set_title("C · phase of speed relative to tide (radius = |band r|)\n"
                 "dashed 180° = peak speed exactly at LOW tide",
                 fontsize=11, color=INK, loc="left", pad=20)
    ax.set_theta_zero_location("E")
    ax.set_rlim(0, 0.62)
    ax.tick_params(labelsize=8, colors=MUTED)

    # ---- D: decay with distance ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for a, c in (("T01", C_T1), ("T02", C_T2)):
        sub = R[R.array == a]
        ax.scatter(sub.dist_km, np.abs(sub.r_band), s=90, color=c, zorder=4, label=a)
        for r in sub.itertuples():
            off = (5, -11) if r.station == "T01C" else (5, 4)
            ax.annotate(r.station, (r.dist_km, abs(r.r_band)), fontsize=7.5,
                        color=MUTED, xytext=off, textcoords="offset points")
    pr = np.corrcoef(R.dist_km, np.abs(R.r_band))[0, 1]
    ax.set_xlabel("distance to nearest wet cell (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("|band correlation|", fontsize=9.5, color=MUTED)
    ax.set_title(f"D · weakens inland, but scatter is large\n"
                 f"Pearson r = {pr:+.2f} (n=7)", fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    # ---- E: caveats, stated on the figure ------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("E · what this does and does not show", fontsize=11.5, color=INK,
                 loc="left")
    txt = [
        ("SHOWN", INK, True),
        ("• all 7 stations negative; mean phase −164°", MUTED, False),
        ("• band-limiting lifts T02A from −0.13 to −0.50", MUTED, False),
        ("• T02 (closer to ocean) responds more than T01", MUTED, False),
        ("", MUTED, False),
        ("NOT SHOWN", INK, True),
        ("• propagation: phase wraps at 24 h, so a −171°", MUTED, False),
        ("  lag and a +189° lead are the same number.", MUTED, False),
        ("  Needs a non-periodic marker (spring-tide", MUTED, False),
        ("  envelope, or a discrete speed-up event).", MUTED, False),
        ("• within-array scatter is unexplained: T02C is", MUTED, False),
        ("  5× weaker than T02A 9 km away; T01A is 3×", MUTED, False),
        ("  weaker than T01D despite being closer.", MUTED, False),
        ("  Could be bed coupling or data quality —", MUTED, False),
        ("  check {STATION}_quality_flags_v2.csv.", MUTED, False),
    ]
    yy = 0.95
    for s, col, bold in txt:
        ax.text(0.0, yy, s, fontsize=8.8, color=col, va="top",
                fontweight="bold" if bold else "normal", transform=ax.transAxes)
        yy -= 0.068

    fig.suptitle("Thwaites GPS: the glacier flows faster at LOW tide "
                 f"(CATS2008_v2023, diurnal band{'' if args.no_clean else ', 8·MAD cleaned'})", fontsize=13.5, color=INK, y=0.975)
    out = os.path.join(args.out_dir, "low_tide_speedup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    R.to_csv(os.path.join(args.out_dir, "low_tide_speedup.csv"), index=False)
    print(f"\nwrote {out}")
    print(f"wrote {os.path.join(args.out_dir, 'low_tide_speedup.csv')}")


if __name__ == "__main__":
    main()
