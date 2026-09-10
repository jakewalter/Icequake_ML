#!/usr/bin/env python3
"""Do T1 and T2 become seismically active at the SAME times?

First-order question before any tidal modelling: if the two arrays -- ~30 km apart on the
same shear margin -- switch on and off together, the forcing is regional (tides, weather,
ice-stream dynamics). If they are independent, whatever drives each is local to it.

Three things make this easy to get wrong, so each is handled explicitly:

  DATA AVAILABILITY. If both arrays were simply off at the same times, rate correlation is
  an instrument artifact. Checked against artifacts/day_file_index.csv: every station has
  707-728 days of HHZ out of ~730, so availability is near-complete and the quiet days are
  real. Days where either array drops below MIN_STATIONS are excluded anyway.

  AUTOCORRELATION. Both rate series are bursty and strongly autocorrelated. A naive p-value
  on Pearson/Spearman is meaningless -- the effective sample size is far below the number of
  days. The null here is a CIRCULAR SHIFT of one series, which preserves each series' own
  autocorrelation and burstiness exactly while destroying only the alignment between them.

  TIMESCALE. Long-period co-variation (both busier in July) is a different claim from
  tidal-band coherence. Seasonal covariance can be environmental and says nothing about
  tides, so the test is run at daily AND sub-daily resolution, and the daily series is also
  high-passed to remove the seasonal envelope before re-testing.

Usage:
    python full_catalog_pipeline/t1_t2_temporal_coherence.py
    ... --t1-reloc <path> --t2-reloc <path>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]
ART = "full_catalog_pipeline/artifacts"
T1_STA = ["DEEJ", "ELZA", "LILA", "TJTJ", "OTIS", "LOUS", "SQIG"]
T2_STA = ["JULA", "EPJZ", "OKGS", "FRST", "DRSC", "WICH", "BAUM"]
MIN_STATIONS = 5          # of 7; below this the array cannot locate reliably
N_NULL = 2000

C1, C2 = "#2a78d6", "#eb6834"
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


def load_times(path):
    d = pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)
    t = (pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy, hour=d.hr, minute=d.mi))
         + pd.to_timedelta(d.sc.astype(float), unit="s"))
    return t.sort_values().reset_index(drop=True)


def availability():
    idx = pd.read_csv(f"{ART}/day_file_index.csv", parse_dates=["date"])
    a = (idx[idx.station.isin(T1_STA)].groupby("date").has_hhz.sum()
         .rename("n_t1").to_frame()
         .join(idx[idx.station.isin(T2_STA)].groupby("date").has_hhz.sum().rename("n_t2"),
               how="outer").fillna(0))
    return a


def circular_shift_null(x, y, stat, n=N_NULL, rng=None):
    """p-value from circularly shifting y. Preserves y's autocorrelation exactly."""
    rng = rng or np.random.default_rng(0)
    obs = stat(x, y)
    null = np.empty(n)
    for i in range(n):
        null[i] = stat(x, np.roll(y, rng.integers(1, len(y) - 1)))
    # two-sided: how often does a shifted series match or beat the observed magnitude?
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, null, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    R = "full_catalog_pipeline/artifacts/full_run"
    ap.add_argument("--t1-reloc",
                    default=f"{R}/T1_v5/hypodd_vels1d_t1ccstrong/output_files/hypoDD.reloc")
    ap.add_argument("--t2-reloc",
                    default=f"{R}/T2_v5/hypodd_vels1d_ccstrong/output_files/hypoDD.reloc")
    ap.add_argument("--out-dir", default=f"{R}/T2_v5")
    args = ap.parse_args()

    t1, t2 = load_times(args.t1_reloc), load_times(args.t2_reloc)
    avail = availability()

    lo = max(t1.min(), t2.min()).normalize()
    hi = min(t1.max(), t2.max()).normalize()
    days = pd.date_range(lo, hi, freq="D")
    ok = avail.reindex(days).fillna(0)
    good = (ok.n_t1 >= MIN_STATIONS) & (ok.n_t2 >= MIN_STATIONS)

    c1 = pd.Series(1, index=t1).resample("D").sum().reindex(days, fill_value=0)
    c2 = pd.Series(1, index=t2).resample("D").sum().reindex(days, fill_value=0)

    print("=" * 78)
    print("T1 / T2 TEMPORAL COHERENCE")
    print("=" * 78)
    print(f"window {lo.date()} .. {hi.date()}  ({len(days)} days)")
    print(f"days with >= {MIN_STATIONS}/7 stations at BOTH arrays: {int(good.sum())} "
          f"({good.mean():.0%})")
    print(f"events retained: T1 {int(c1[good].sum())}/{len(t1)}, "
          f"T2 {int(c2[good].sum())}/{len(t2)}")
    x, y = c1[good].values.astype(float), c2[good].values.astype(float)

    def sp(a, b):
        return spearmanr(a, b).correlation

    def pe(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    print()
    print("--- 1. DAILY rate coherence (raw) ---")
    for name, f in (("Spearman", sp), ("Pearson", pe)):
        obs, null, p = circular_shift_null(x, y, f)
        print(f"  {name:9s} r = {obs:+.3f}   circular-shift p = {p:.4f}   "
              f"(null |r| p95 = {np.percentile(np.abs(null), 95):.3f})")

    # high-pass: remove the seasonal envelope, keep short-period variability
    win = 29
    xh = x - uniform_filter1d(x, win, mode="nearest")
    yh = y - uniform_filter1d(y, win, mode="nearest")
    print()
    print(f"--- 2. DAILY coherence after removing a {win}-day running mean ---")
    print("    (a seasonal envelope shared by both is NOT evidence of tidal forcing)")
    for name, f in (("Spearman", sp), ("Pearson", pe)):
        obs, null, p = circular_shift_null(xh, yh, f)
        print(f"  {name:9s} r = {obs:+.3f}   circular-shift p = {p:.4f}")

    # lagged cross-correlation on the high-passed series
    lags = np.arange(-20, 21)
    xc = np.array([pe(xh, np.roll(yh, L)) for L in lags])
    best = lags[np.argmax(np.abs(xc))]
    print()
    print(f"--- 3. lagged cross-correlation (high-passed) ---")
    print(f"  peak |r| = {np.abs(xc).max():.3f} at lag {best:+d} days "
          f"(r at lag 0 = {xc[lags == 0][0]:+.3f})")

    # sub-daily: hourly bins, only on days both arrays were active
    both_active = good & (c1 > 0) & (c2 > 0)
    sel_days = days[both_active]
    hours = pd.date_range(sel_days.min(), sel_days.max() + pd.Timedelta("1D"), freq="H")
    h1 = pd.Series(1, index=t1).resample("H").sum().reindex(hours, fill_value=0)
    h2 = pd.Series(1, index=t2).resample("H").sum().reindex(hours, fill_value=0)
    keep = pd.Series(hours.normalize(), index=hours).isin(sel_days)
    hx, hy = h1[keep].values.astype(float), h2[keep].values.astype(float)
    print()
    print(f"--- 4. SUB-DAILY (hourly) coherence, on the {len(sel_days)} days both were active ---")
    print(f"    {len(hx)} hourly bins; T1 {int(hx.sum())} events, T2 {int(hy.sum())} events")
    for name, f in (("Spearman", sp), ("Pearson", pe)):
        obs, null, p = circular_shift_null(hx, hy, f)
        print(f"  {name:9s} r = {obs:+.3f}   circular-shift p = {p:.4f}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(4, 1, figsize=(13, 12),
                             gridspec_kw=dict(height_ratios=[1, 1, 0.9, 0.9], hspace=0.42))

    ax = axes[0]
    ax.bar(days, c1.values, width=1.0, color=C1, label=f"T1 (n={len(t1)})")
    ax.bar(days, -c2.values, width=1.0, color=C2, label=f"T2 (n={len(t2)})")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("events/day\nT1 up · T2 down", fontsize=9.5, color=MUTED)
    ax.set_title("daily event rate, back to back", fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax = axes[1]
    ax.plot(days, uniform_filter1d(c1.values.astype(float), 14), color=C1, lw=2,
            label="T1, 14-day mean")
    ax.plot(days, uniform_filter1d(c2.values.astype(float), 14), color=C2, lw=2,
            label="T2, 14-day mean")
    ax.set_ylabel("events/day (smoothed)", fontsize=9.5, color=MUTED)
    ax.set_title("seasonal envelope — shared long-period variation is NOT tidal evidence",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(lags, xc, color=INK, lw=1.8, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("lag (days, T2 shifted)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("Pearson r", fontsize=9.5, color=MUTED)
    ax.set_title("lagged cross-correlation of the high-passed daily rates",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)

    ax = axes[3]
    obs, null, p = circular_shift_null(xh, yh, pe)
    ax.hist(null, bins=60, color="#b9b8b2", zorder=3,
            label=f"circular-shift null ({N_NULL} draws)")
    ax.axvline(obs, color=C2, lw=2.4, zorder=5, label=f"observed r = {obs:+.3f} (p = {p:.3f})")
    ax.set_xlabel("Pearson r, high-passed daily rates", fontsize=9.5, color=MUTED)
    ax.set_ylabel("draws", fontsize=9.5, color=MUTED)
    ax.set_title("significance against an autocorrelation-preserving null",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Are T1 and T2 seismically active at the same times?",
                 fontsize=13.5, color=INK, y=0.945)
    out = os.path.join(args.out_dir, "t1_t2_temporal_coherence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"\nwrote {out}")

    pd.DataFrame(dict(date=days, t1=c1.values, t2=c2.values,
                      n_sta_t1=ok.n_t1.values, n_sta_t2=ok.n_t2.values,
                      usable=good.values)).to_csv(
        os.path.join(args.out_dir, "t1_t2_daily_rates.csv"), index=False)
    print(f"wrote {os.path.join(args.out_dir, 't1_t2_daily_rates.csv')}")


if __name__ == "__main__":
    main()
