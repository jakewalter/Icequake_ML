#!/usr/bin/env python3
"""Explain WHY interpolating x2 before cross-correlation rescues hypoDD's P differential times.

The confusing part of the fix is that upsampling looks like it invents information. It does not.
The cross-correlation of two band-limited signals is a smooth CONTINUOUS function of lag. obspy's
xcorr_pick_correction only ever evaluates that function on a grid whose spacing is exactly
1/sampling_rate, then -- to get sub-sample timing -- fits a parabola to the run of samples around
the peak over which the curvature stays convex. If fewer than three grid points land on that
convex lobe it raises "Less than 3 samples selected for fit to cross correlation" and the
measurement is thrown away.

So the failure is not about signal quality or about the peak being ill-defined. It is about
sampling the same curve too coarsely to fit a parabola to its top. Correlating over a 20-90 Hz
passband at 200 Hz gives a CC function that oscillates every ~2-3 samples, so the convex lobe
around its peak is 1-2 samples wide and the fit is refused. Measured on real pairs, the refusal
rate is 98.3% for P on Z and 65-76% for S -- it hits BOTH phases hard.

Two things then separate them, and both are measured rather than assumed (panel C):
  * P is refused more often on the same channel: 1.7% of P/Z attempts clear the threshold
    against 18.3% of S/Z, a factor of 10.8. P's coherent energy oscillates a little faster
    (median 70 Hz against S's 60 Hz), so its lobe is the narrower of the two.
  * S gets three shots and P gets one. cc_s_phase_weighting offers Z, E and N and hypoDDpy
    returns on the first channel that clears the threshold; cc_p_phase_weighting offers only Z.
    That is worth a further 2.3x (S 41.3% across three channels vs 18.3% on Z alone).
Together they account for the observed ~24x gap. Interpolating x2 removes the refusal itself,
which is why P -- the phase with no second channel to fall back on -- gains the most.

The figure is built to make that verifiable rather than asserted: the densely-evaluated curve is
drawn through the coarse samples, so you can see the 200 Hz and 400 Hz points lying on ONE curve.

Panels:
  A, D  the two events' waveforms in the correlation window, for P and for S
  B, E  the CC function near its peak, sampled at 200 Hz and at 400 Hz, with the convex lobe
        obspy would find shaded and its sample count labelled -- the mechanism
  C     why the fit is refused, per phase and channel, from the attempts table
  F     the measured consequence, from diagnose_cc_p_deficit.py --fix-sweep

An earlier draft of panel C plotted the two windows' amplitude spectra to argue "P is the
higher-frequency arrival". The rendered figure refuted its own caption -- the example P peaked
at 68 Hz and the S at 73 Hz -- so the claim was dropped for the measured refusal rates, which
say what actually separates the two phases.

Colour means SAMPLING RATE throughout (one entity, one hue); P versus S is carried by panel
position, line style and direct labels, never by hue.

Usage:
    python full_catalog_pipeline/plot_cc_upsampling_explainer.py --array T2
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
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime
from obspy.signal.cross_correlation import correlate
from obspy.signal.invsim import cosine_taper

import catalog_paths
import config
from diagnose_cc_p_deficit import (
    CC_TIME_BEFORE, CC_TIME_AFTER, CC_MAXLAG, CC_FMIN, CC_FMAX, CC_MIN_COEFF,
    load_phase_dat, dt_ct_pairs, grab,
)
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache

# Measured by diagnose_cc_p_deficit.py --fix-sweep on 120 real event pairs (T2).
PASS_RATES = {"P": {200: 0.010, 400: 0.997}, "S": {200: 0.197, 400: 0.563}}

RATE_COLOR = {200: "#eb6834", 400: "#2a78d6"}   # validated pair, light surface
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"
LOBE = "#f0efe9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def prepared_slice(arr, t_start, pick, sr, upsample):
    """One trace put through obspy's exact preparation: interpolate (our shim), then demean,
    10% cosine taper, bandpass, and slice [pick - before - maxlag/2, pick + after + maxlag/2]."""
    tr = Trace(np.asarray(arr, float))
    tr.stats.sampling_rate = sr
    tr.stats.starttime = t_start
    if upsample > 1:
        tr.interpolate(sampling_rate=sr * upsample, method="lanczos", a=20)
    tr.detrend(type="demean")
    tr.data *= cosine_taper(len(tr), 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr.filter("bandpass", freqmin=CC_FMIN, freqmax=CC_FMAX)
    return tr.slice(pick - CC_TIME_BEFORE - CC_MAXLAG / 2.0,
                    pick + CC_TIME_AFTER + CC_MAXLAG / 2.0)


def cc_function(a1, t1s, p1, a2, t2s, p2, sr, upsample):
    """obspy's cross-correlation function of lag, plus the convex lobe it would fit."""
    s1 = prepared_slice(a1, t1s, p1, sr, upsample)
    s2 = prepared_slice(a2, t2s, p2, sr, upsample)
    rate = sr * upsample
    shift_len = int(CC_MAXLAG * rate)
    n = min(len(s1.data), len(s2.data))
    cc = correlate(s1.data[:n], s2.data[:n], shift_len, method="direct")
    lag = np.linspace(-CC_MAXLAG, CC_MAXLAG, shift_len * 2 + 1)
    # obspy's rule, verbatim: walk out from the peak while curvature stays <= 0
    curv = np.concatenate((np.zeros(1), np.diff(cc, 2), np.zeros(1)))
    peak = int(cc.argmax())
    first = peak
    while first > 0 and curv[first - 1] <= 0:
        first -= 1
    last = peak
    while last < len(cc) - 1 and curv[last + 1] <= 0:
        last += 1
    return lag, cc, first, last, peak


def pick_example(ph, pairs, caches, phase, day_index, want_fail_at_200):
    """First real pair at which obspy refuses the fit at 200 Hz but accepts it at 400."""
    for e1, e2 in pairs:
        if e1 not in ph or e2 not in ph:
            continue
        for key in [k for k in ph[e1] if isinstance(k, tuple) and k[1] == phase and k in ph[e2]]:
            sta = key[0]
            caches.setdefault(sta, RollingDayCache(day_index, sta))
            t1 = ph[e1]["origin"] + ph[e1][key]
            t2 = ph[e2]["origin"] + ph[e2][key]
            pad = (max(CC_TIME_BEFORE * 2, 2.0), max(CC_TIME_AFTER * 2, 2.0))
            a1 = grab(caches[sta], t1 - pad[0], t1 + pad[1], "Z")
            a2 = grab(caches[sta], t2 - pad[0], t2 + pad[1], "Z")
            if a1 is None or a2 is None or min(len(a1), len(a2)) < 100:
                continue
            if np.std(a1) == 0 or np.std(a2) == 0:
                continue
            sr = config.SAMPLE_RATE_HZ
            try:
                r200 = cc_function(a1, t1 - pad[0], t1, a2, t2 - pad[0], t2, sr, 1)
                r400 = cc_function(a1, t1 - pad[0], t1, a2, t2 - pad[0], t2, sr, 2)
                rref = cc_function(a1, t1 - pad[0], t1, a2, t2 - pad[0], t2, sr, 16)
            except Exception:
                continue
            n200 = r200[3] - r200[2] + 1
            n400 = r400[3] - r400[2] + 1
            if r200[1].max() < CC_MIN_COEFF:
                continue
            if want_fail_at_200 and not (n200 < 3 <= n400):
                continue
            if not want_fail_at_200 and n200 < 3:
                continue
            return dict(e1=e1, e2=e2, sta=sta, phase=phase, a1=a1, a2=a2,
                        t1=t1, t2=t2, t1s=t1 - pad[0], t2s=t2 - pad[0],
                        r200=r200, r400=r400, rref=rref, n200=n200, n400=n400)
    return None


def draw_waveforms(ax, ex, sr, letter):
    s1 = prepared_slice(ex["a1"], ex["t1s"], ex["t1"], sr, 1)
    s2 = prepared_slice(ex["a2"], ex["t2s"], ex["t2"], sr, 1)
    n = min(len(s1.data), len(s2.data))
    t = np.arange(n) / sr - CC_TIME_BEFORE - CC_MAXLAG / 2.0
    for d, style, lab, col in ((s1.data[:n], "-", "event 1", INK),
                               (s2.data[:n], "--", "event 2", MUTED)):
        ax.plot(t, d / np.abs(d).max(), style, color=col, lw=1.2, label=lab)
    ax.axvline(0, color=SECONDARY, lw=1)
    ax.set_xlabel("time relative to the pick (s)")
    ax.set_yticks([])
    ax.set_ylabel("normalised")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"{letter}. {ex['phase']} window at {ex['sta']} — the two events "
                 f"being correlated", fontsize=9.5, loc="left")


def draw_cc(ax, ex, letter):
    lag_r, cc_r, *_ = ex["rref"]
    ax.plot(lag_r * 1000, cc_r, color=MUTED, lw=1.0, zorder=1,
            label="the underlying CC function")
    for rate, up, ms, marker in ((400, "r400", 5.5, "s"), (200, "r200", 8.0, "o")):
        lag, cc, first, last, peak = ex[up]
        n = last - first + 1
        ax.plot(lag * 1000, cc, marker, color=RATE_COLOR[rate], ms=ms, lw=0,
                zorder=3, label=f"{rate} Hz grid ({n} on the lobe)")
        if rate == 200:
            ax.axvspan(lag[first] * 1000, lag[last] * 1000, color=LOBE, zorder=0)
    verdict = "fit REFUSED" if ex["n200"] < 3 else "fit accepted"
    bb = dict(facecolor="#fcfcfb", edgecolor="none", pad=1.6, alpha=0.9)
    ax.annotate(f"200 Hz → {ex['n200']} on the lobe → {verdict}",
                xy=(0.02, 0.13), xycoords="axes fraction", fontsize=8, bbox=bb, zorder=5,
                color=RATE_COLOR[200], fontweight="bold" if ex["n200"] < 3 else "normal")
    ax.annotate(f"400 Hz → {ex['n400']} on the lobe → fit accepted",
                xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8, bbox=bb, zorder=5,
                color=RATE_COLOR[400])
    # period of the coherent oscillation, which is what sets the lobe width
    lr, cr = ex["rref"][0], ex["rref"][1]
    C = cr - cr.mean()
    f = np.fft.rfftfreq(len(C), lr[1] - lr[0])
    fdom = f[int(np.argmax(np.abs(np.fft.rfft(C))))]
    ax.set_xlim(-30, 30)
    ax.set_xlabel("correlation lag (ms)")
    ax.set_ylabel("correlation coefficient")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.set_title(f"{letter}. {ex['phase']} — one curve, two sampling grids\n"
                 f"     CC oscillates at {fdom:.0f} Hz → lobe ≈ {200/(2*fdom):.1f} samples",
                 fontsize=9.5, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--scan-pairs", type=int, default=4000)
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    sr = config.SAMPLE_RATE_HZ
    ph = load_phase_dat(catalog_paths.phase_dat(args.array))
    pairs = dt_ct_pairs(os.path.join(work, "input_files", "dt.ct"), args.scan_pairs)
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    caches = {}

    print("searching for a P pair obspy refuses at 200 Hz but accepts at 400 ...")
    exP = pick_example(ph, pairs, caches, "P", day_index, True)
    print("searching for an S pair obspy accepts at 200 Hz ...")
    exS = pick_example(ph, pairs, caches, "S", day_index, False)
    if exP is None or exS is None:
        raise SystemExit(f"no suitable example found (P={exP is not None}, S={exS is not None})")
    print(f"  P: events {exP['e1']}/{exP['e2']} at {exP['sta']} — "
          f"lobe {exP['n200']} samples at 200 Hz, {exP['n400']} at 400 Hz")
    print(f"  S: events {exS['e1']}/{exS['e2']} at {exS['sta']} — "
          f"lobe {exS['n200']} samples at 200 Hz, {exS['n400']} at 400 Hz")

    attempts = os.path.join(work, f"{args.array.lower()}_cc_p_deficit_attempts.csv")
    if not os.path.exists(attempts):
        raise SystemExit(f"{attempts} missing -- run diagnose_cc_p_deficit.py first")
    at = pd.read_csv(attempts)
    at["grp"] = at.phase + "/" + at.chan

    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.30)
    draw_waveforms(fig.add_subplot(gs[0, 0]), exP, sr, "A")
    draw_cc(fig.add_subplot(gs[0, 1]), exP, "B")

    # C: why attempts die, per phase and channel -- measured, at 200 Hz
    ax = fig.add_subplot(gs[0, 2])
    order = ["P/Z", "S/Z", "S/E", "S/N"]
    y = np.arange(len(order))[::-1]
    for k, g in enumerate(order):
        h = at[at.grp == g]
        if not len(h):
            continue
        ref = 100 * (h.status != "ok").mean()
        acc = 100 - ref
        ax.barh(y[k], acc, 0.6, color="#d8d7d1", zorder=2)
        ax.barh(y[k], ref, 0.6, left=acc + 0.6, color=RATE_COLOR[200], zorder=2)
        ax.annotate(f"{ref:.0f}% refused", (min(acc + 6, 62), y[k]), va="center",
                    fontsize=8, color=SECONDARY)
        if g == "S/N":   # widest accepted segment -- label it once, in place of a legend
            ax.annotate("fit accepted", (2.5, y[k]), va="center", fontsize=8,
                        color=SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of attempts at 200 Hz (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("C. the fit refuses most attempts on EVERY phase —\n"
                 "     and P has no second channel to fall back on",
                 fontsize=9.5, loc="left")

    draw_waveforms(fig.add_subplot(gs[1, 0]), exS, sr, "D")
    draw_cc(fig.add_subplot(gs[1, 1]), exS, "E")

    # F: measured consequence
    ax = fig.add_subplot(gs[1, 2])
    labels = ["P\n(Z only)", "S\n(Z only)"]
    x = np.arange(2)
    w = 0.34
    for i, rate in enumerate((200, 400)):
        vals = [PASS_RATES["P"][rate] * 100, PASS_RATES["S"][rate] * 100]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=RATE_COLOR[rate],
                      label=f"{rate} Hz", zorder=2, edgecolor="#fcfcfb", linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v + 3),
                        ha="center", fontsize=8, color=SECONDARY, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("attempts clearing CC ≥ 0.4 (%)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.02))
    ax.set_title("F. the consequence, on 120 real event pairs", fontsize=9.5, loc="left")

    fig.suptitle("Why interpolating ×2 rescues the P cross-correlations — obspy needs 3 grid "
                 "points on the CC peak's convex lobe; at 200 Hz it gets 2",
                 fontsize=11.5, color=INK, y=1.00)
    out = os.path.join(work, f"{args.array.lower()}_cc_upsampling_explainer.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
