#!/usr/bin/env python3
"""Why does hypoddpy's cross-correlation stage produce almost no P differential times?

Established by `hypodd_cc_depth_information.py`: T2's dt.cc is 97.8% S (17,015 P vs 743,545 S)
while the catalog picks (16.4k P / 15.5k S) and dt.ct (1.79M P / 1.78M S) are balanced. Since
hypoddpy builds dt.cc by walking dt.ct's event pairs and attempting every pick both events
share, P and S get essentially the same number of ATTEMPTS. So P's success rate is ~44x lower
than S's, and this script measures where those attempts die.

Two asymmetries are visible in hypoddpy's code and both are tested here:

  * P is offered ONE channel and S is offered THREE. `cc_p_phase_weighting={"Z": 1.0}` versus
    `cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0}`, and _perform_cross_correlation
    returns on the FIRST channel clearing cc_min_allowed_cross_corr_coeff. So S gets three
    independent chances per attempt and P gets one. That buys at most 3x, not 44x.
  * The window is short and the correlation is PAIRWISE. cc_time_before=0.05,
    cc_time_after=0.2 -- a 0.25 s window with only 50 samples at 200 Hz -- correlating two
    single noisy traces against each other, with a 0.4 coefficient threshold. Compare the
    stacking work elsewhere in this pipeline, which aligns each trace against a stack of
    hundreds (MCCC) over a 0.6 s window and gets mean P coefficients of 0.60-0.84 on the SAME
    Z channel. If P clears 0.4 there and not here, the parameters are the cause, not the data.

Method. Draw real event pairs from dt.ct that share a pick at a station, then call obspy's
xcorr_pick_correction with hypoddpy's EXACT parameters -- same window, same max lag, same
20-90 Hz bandpass, same threshold -- and tabulate the coefficient distribution and pass rate
per phase and per channel. Channel is reported separately for S so the "is Z itself bad?"
explanation can be separated from "is P starved by having only one channel?": if S also fails
on Z but succeeds on HH1/HH2, P is collateral damage of being Z-only.

A --window-sweep option re-runs the same pairs at wider windows and lags to show what the pass
rate would be under different settings, since the actionable question is which parameter to
change.

Usage:
    python full_catalog_pipeline/diagnose_cc_p_deficit.py --array T2 --pairs 400
    python full_catalog_pipeline/diagnose_cc_p_deficit.py --array T2 --pairs 400 --window-sweep
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
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
from obspy import Trace
from obspy.signal.cross_correlation import xcorr_pick_correction

import catalog_paths
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window

# hypoddpy's settings, from hypodd_relocate.py's HypoDDRelocator(...) call
CC_TIME_BEFORE, CC_TIME_AFTER, CC_MAXLAG = 0.05, 0.2, 0.1
CC_FMIN, CC_FMAX = 20.0, 90.0
CC_MIN_COEFF = 0.4
# hypoddpy maps its "E"/"N" keys onto this deployment's HH1/HH2 via _select_traces_smart;
# lib.windowing.extract_window already returns the same three under those names
# (Z<-HHZ, E<-HH1, N<-HH2), so the key passes straight through.
CHAN_FOR = {"Z": "Z", "E": "E", "N": "N"}
P_CHANNELS = ["Z"]
S_CHANNELS = ["Z", "E", "N"]

# (upsample factor, bandpass upper corner) — the two levers on the parabola-fit width
FIX_SWEEP = [(1, 90.0), (1, 50.0), (1, 35.0), (2, 90.0), (4, 90.0), (8, 90.0), (4, 50.0)]
CC_UPSAMPLE_CHECK = 2   # the factor the shim in hypodd_relocate.py actually installs


def load_phase_dat(path):
    """phase.dat -> {event_id: {"origin": UTCDateTime, (sta, phase): offset}}."""
    from obspy import UTCDateTime
    out, cur = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                cur = int(p[-1])
                yr, mo, dy, hr, mi, sc = p[1:7]
                out[cur] = {"origin": UTCDateTime(int(yr), int(mo), int(dy),
                                                  int(hr), int(mi), float(sc))}
            elif cur is not None:
                p = line.split()
                if len(p) == 4:
                    out[cur][(p[0].split(".")[-1], p[3])] = float(p[1])
    return out


def dt_ct_pairs(path, limit):
    """Event-id pairs from dt.ct's header lines (the same list hypoddpy walks)."""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                pairs.append((int(p[1]), int(p[2])))
                if len(pairs) >= limit:
                    break
    return pairs


def grab(cache, t0, t1, channel):
    r = {"primary_date": t0.date.isoformat()}
    data, _ = extract_window(cache, r, t0, t1)
    if data is None:
        return None
    return data.get(channel)


def collect_work(ph, pairs, limit_pairs):
    """One work item per (event pair, station, phase) both events share."""
    items, used = [], 0
    for e1, e2 in pairs:
        if used >= limit_pairs:
            break
        if e1 not in ph or e2 not in ph:
            continue
        shared = [k for k in ph[e1]
                  if isinstance(k, tuple) and k[1] in ("P", "S") and k in ph[e2]]
        if not shared:
            continue
        for sta, phase in shared:
            items.append(dict(e1=e1, e2=e2, sta=sta, phase=phase,
                              t1=ph[e1]["origin"] + ph[e1][(sta, phase)],
                              t2=ph[e2]["origin"] + ph[e2][(sta, phase)]))
        used += 1
    return items, used


def prefetch_windows(items, day_index):
    """Extract every window the work list needs, ONCE each, in date order.

    ph2dt pairs events that are spatial neighbours, so the two events of a pair are often
    months apart -- which is why hypoDDpy's per-pair waveform loading is so expensive and why
    ordering the PAIRS by date does not help. Ordering the individual (event, station, phase)
    WINDOWS by date does: each station-day file is then read at most once and the rolling
    cache can evict behind the cursor. The extracted windows are a few hundred samples each,
    so the whole work list fits in memory and every later cross-correlation, at every
    upsampling factor, is served from RAM.
    """
    need = {}
    for it in items:
        for tag, t in (("1", it["t1"]), ("2", it["t2"])):
            need[(it["e" + tag], it["sta"], it["phase"])] = t
    pad = (max(CC_TIME_BEFORE * 2, 2.0), max(CC_TIME_AFTER * 2, 2.0))
    chans_for = {"P": P_CHANNELS, "S": S_CHANNELS}
    by_station = {}
    for (ev, sta, phase), t in need.items():
        by_station.setdefault(sta, []).append((t, ev, phase))
    out = {}
    for sta, entries in by_station.items():
        entries.sort(key=lambda x: x[0])          # date order within the station
        cache = RollingDayCache(day_index, sta)
        for t, ev, phase in entries:
            r = {"primary_date": (t - pad[0]).date.isoformat()}
            data, _ = extract_window(cache, r, t - pad[0], t + pad[1])
            cache.evict_before(r["primary_date"])
            if data is None:
                continue
            keep = {c: np.asarray(data[c], float) for c in chans_for[phase]
                    if data.get(c) is not None}
            if keep:
                out[(ev, sta, phase)] = (t - pad[0], keep)
    print(f"  prefetched {len(out):,} of {len(need):,} windows "
          f"({len(by_station)} stations, one pass over the day files)")
    return out


def cc_from_windows(win, ev1, ev2, sta, phase, chan_key, before, after, maxlag,
                    upsample=1, fmax=CC_FMAX):
    """try_cc, but served from prefetched windows instead of re-reading the archive.

    Returns (coeff, status, correction). The correction is obspy's `pick2_corr`, which is what
    decides the differential time hypoDDpy actually writes:
        diff_travel_time = (pick1 - origin1) - (pick2 + pick2_corr - origin2)
    i.e. dt.cc = dt.ct - correction. So a recovered measurement is only USEFUL if that
    correction is small: a cycle-skipped correlation passes the coefficient threshold and the
    parabola fit, then writes a differential time displaced by most of the lag window --
    exactly the kind of entry the |dt| cap exists to remove.
    """
    a = win.get((ev1, sta, phase))
    b = win.get((ev2, sta, phase))
    if a is None or b is None:
        return np.nan, "no_data", np.nan
    if chan_key not in a[1] or chan_key not in b[1]:
        return np.nan, "no_channel", np.nan
    trs = []
    for t0, chans in (a, b):
        arr = chans[chan_key]
        if len(arr) < 10 or np.std(arr) == 0:
            return np.nan, "zero_variance", np.nan
        tr = Trace(np.asarray(arr, float))
        tr.stats.sampling_rate = config.SAMPLE_RATE_HZ
        tr.stats.starttime = t0
        if upsample > 1:
            tr.interpolate(sampling_rate=config.SAMPLE_RATE_HZ * upsample,
                           method="lanczos", a=20)
        trs.append(tr)
    t1 = a[0] + max(CC_TIME_BEFORE * 2, 2.0)
    t2 = b[0] + max(CC_TIME_BEFORE * 2, 2.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, coeff = xcorr_pick_correction(
                t1, trs[0], t2, trs[1], t_before=before, t_after=after,
                cc_maxlag=maxlag, filter="bandpass",
                filter_options={"freqmin": CC_FMIN, "freqmax": fmax}, plot=False)
        if coeff is None or np.isnan(coeff):
            return np.nan, "nan", np.nan
        return float(coeff), "ok", float(corr)
    except Exception as e:
        return np.nan, "exc:" + str(e).strip().split("\n")[0][:70], np.nan


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near 0% and 100%, where normal approximation is not."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def try_cc(cache_by_sta, sta, chan_key, t1, t2, before, after, maxlag,
           upsample=1, fmax=CC_FMAX):
    """hypoddpy's exact call. Returns (coeff, status).

    `upsample` and `fmax` exist to test the fix, not to reproduce production: obspy fits a
    parabola to the convex region around the cross-correlation peak and refuses with "Less
    than 3 samples" when that region is narrower than 3 samples. At 200 Hz a signal near the
    90 Hz top of the band gives a CC function with a period of ~2.2 samples, so the convex
    region is 1-2 samples wide and the fit is refused almost every time. Interpolating to a
    higher rate widens it in samples without touching the data's information content;
    lowering fmax widens it by discarding the highest-frequency content."""
    chan = CHAN_FOR[chan_key]
    pad = max(before * 2, 2.0), max(after * 2, 2.0)
    from obspy import Trace, UTCDateTime
    out = []
    for t in (t1, t2):
        arr = grab(cache_by_sta[sta], t - pad[0], t + pad[1], chan)
        if arr is None or len(arr) < 10:
            return np.nan, "no_data"
        tr = Trace(np.asarray(arr, float))
        tr.stats.sampling_rate = config.SAMPLE_RATE_HZ
        tr.stats.starttime = t - pad[0]
        if upsample > 1:
            tr.interpolate(sampling_rate=config.SAMPLE_RATE_HZ * upsample,
                           method="lanczos", a=20)
        out.append(tr)
    if np.std(out[0].data) == 0 or np.std(out[1].data) == 0:
        return np.nan, "zero_variance"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, coeff = xcorr_pick_correction(
                t1, out[0], t2, out[1], t_before=before, t_after=after,
                cc_maxlag=maxlag, filter="bandpass",
                filter_options={"freqmin": CC_FMIN, "freqmax": fmax}, plot=False)
        if coeff is None or np.isnan(coeff):
            return np.nan, "nan"
        return float(coeff), "ok"
    except Exception as e:
        # The message matters, not the class: obspy raises a bare Exception for every
        # distinct refusal (too-short window, peak at the lag edge, ...), and those are
        # different bugs with different fixes.
        return np.nan, "exc:" + str(e).strip().split("\n")[0][:70]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--scan-pairs", type=int, default=200000,
                    help="how many dt.ct header lines to read before sampling")
    ap.add_argument("--fix-sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    ph = load_phase_dat(catalog_paths.phase_dat(args.array))
    pairs = dt_ct_pairs(os.path.join(work, "input_files", "dt.ct"), args.scan_pairs)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    items, used = collect_work(ph, pairs, args.pairs)
    print(f"{args.array}: {used} event pairs -> {len(items)} (pair, station, phase) "
          f"observations; extracting windows ...")
    win = prefetch_windows(items, day_index)

    # Baseline: hypoDDpy's settings exactly, every channel it would offer for that phase.
    rows = []
    for it in items:
        for chan_key in (P_CHANNELS if it["phase"] == "P" else S_CHANNELS):
            coeff, status, _ = cc_from_windows(win, it["e1"], it["e2"], it["sta"],
                                               it["phase"], chan_key, CC_TIME_BEFORE,
                                               CC_TIME_AFTER, CC_MAXLAG)
            rows.append(dict(e1=it["e1"], e2=it["e2"], sta=it["sta"], phase=it["phase"],
                             chan=chan_key, coeff=coeff, status=status,
                             passed=bool(np.isfinite(coeff) and coeff >= CC_MIN_COEFF)))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no attempts made -- check dt.ct / phase.dat / waveform index")
        return

    print(f"\n{args.array}: {len(df)} channel attempts at hypoDDpy's settings "
          f"(before={CC_TIME_BEFORE}s after={CC_TIME_AFTER}s maxlag={CC_MAXLAG}s "
          f"band={CC_FMIN:.0f}-{CC_FMAX:.0f}Hz thresh={CC_MIN_COEFF})\n")
    print("per-channel attempt outcome")
    print(f"{'phase':6} {'chan':5} {'n':>7} {'ok':>7} {'refused':>8} {'coeff med':>10} "
          f"{'>=0.4':>7} {'95% CI':>16}")
    for (phase, chan), g in df.groupby(["phase", "chan"]):
        ok = g[g.status == "ok"]
        lo, hi = wilson(int(g.passed.sum()), len(g))
        print(f"{phase:6} {chan:5} {len(g):7,} {len(ok):7,} "
              f"{(g.status != 'ok').mean():7.1%} "
              f"{ok.coeff.median() if len(ok) else float('nan'):10.3f} "
              f"{g.passed.mean():6.1%} {f'[{lo:.1%}, {hi:.1%}]':>16}")

    print("\nfailure status breakdown")
    print(pd.crosstab(df.status, [df.phase, df.chan]).to_string())

    print("\nEFFECTIVE per-observation success (any offered channel clears the threshold)")
    eff = df.groupby(["e1", "e2", "sta", "phase"]).passed.any().reset_index()
    effrate = {}
    for phase, g in eff.groupby("phase"):
        lo, hi = wilson(int(g.passed.sum()), len(g))
        effrate[phase] = g.passed.mean()
        print(f"  {phase}: {g.passed.mean():.1%} [{lo:.1%}, {hi:.1%}] of {len(g):,} "
              f"({'1 channel' if phase == 'P' else '3 channels'} offered)")
    sz = df[(df.phase == "S") & (df.chan == "Z")]
    if len(sz) and "P" in effrate:
        szr = sz.groupby(["e1", "e2", "sta"]).passed.any()
        print(f"  S restricted to Z only (as P is): {szr.mean():.1%} of {len(szr):,}")
        print(f"  -> gap decomposition: same-channel P/Z vs S/Z = "
              f"{szr.mean()/max(effrate['P'], 1e-9):.1f}x, "
              f"channel count = {effrate.get('S', 0)/max(szr.mean(), 1e-9):.1f}x, "
              f"total = {effrate.get('S', 0)/max(effrate['P'], 1e-9):.1f}x")

    df.to_csv(os.path.join(work, f"{args.array.lower()}_cc_p_deficit_attempts.csv"),
              index=False)

    if args.fix_sweep:
        print(f"\nFIX SWEEP -- same {len(items)} observations, served from the prefetched "
              f"windows (Z channel, window/lag unchanged)")
        print(f"{'upsample':>9} {'sr_Hz':>7} {'fmax':>6} {'P pass':>8} {'P 95% CI':>16} "
              f"{'P coeff':>8} {'S(Z) pass':>10} {'S 95% CI':>16} {'P refused':>10}")
        sweep_rows = []
        for up, fmax in FIX_SWEEP:
            stat = {}
            for phase in ("P", "S"):
                q = [it for it in items if it["phase"] == phase]
                good = tot = refused = 0
                coefs = []
                for it in q:
                    c, st, _ = cc_from_windows(win, it["e1"], it["e2"], it["sta"], phase, "Z",
                                               CC_TIME_BEFORE, CC_TIME_AFTER, CC_MAXLAG,
                                               upsample=up, fmax=fmax)
                    if st in ("no_data", "no_channel"):
                        continue
                    tot += 1
                    if np.isfinite(c):
                        good += int(c >= CC_MIN_COEFF)
                        coefs.append(c)
                    elif st.startswith("exc:Less than 3"):
                        refused += 1
                stat[phase] = (good, tot, refused, coefs)
            (pg, pt, pr, pc), (sg, st_, sr_, _) = stat["P"], stat["S"]
            plo, phi = wilson(pg, pt)
            slo, shi = wilson(sg, st_)
            print(f"{up:9d} {config.SAMPLE_RATE_HZ*up:7.0f} {fmax:6.0f} "
                  f"{pg/max(pt,1):7.1%} {f'[{plo:.1%}, {phi:.1%}]':>16} "
                  f"{np.median(pc) if pc else float('nan'):8.3f} "
                  f"{sg/max(st_,1):9.1%} {f'[{slo:.1%}, {shi:.1%}]':>16} "
                  f"{pr/max(pt,1):9.1%}")
            sweep_rows.append(dict(upsample=up, sr_hz=config.SAMPLE_RATE_HZ*up, fmax=fmax,
                                   p_n=pt, p_pass=pg/max(pt,1), p_lo=plo, p_hi=phi,
                                   p_coeff_med=float(np.median(pc)) if pc else np.nan,
                                   p_refused=pr/max(pt,1),
                                   s_n=st_, s_pass=sg/max(st_,1), s_lo=slo, s_hi=shi))
        # Are the RECOVERED measurements usable, or merely present? Count is not quality:
        # check the correction that sets the written differential time.
        print(f"\nQuality of the differential times recovered at x{CC_UPSAMPLE_CHECK} "
              f"(P, Z channel, coeff >= {CC_MIN_COEFF})")
        for phase in ("P", "S"):
            rec = []
            for it in [i for i in items if i["phase"] == phase]:
                c, st, corr = cc_from_windows(win, it["e1"], it["e2"], it["sta"], phase, "Z",
                                              CC_TIME_BEFORE, CC_TIME_AFTER, CC_MAXLAG,
                                              upsample=CC_UPSAMPLE_CHECK)
                if st == "ok" and np.isfinite(c) and c >= CC_MIN_COEFF:
                    dt_ct = ph[it["e1"]][(it["sta"], phase)] - ph[it["e2"]][(it["sta"], phase)]
                    rec.append((abs(corr), abs(dt_ct - corr)))
            if not rec:
                print(f"  {phase}: none recovered")
                continue
            ac = np.array([r[0] for r in rec])
            adt = np.array([r[1] for r in rec])
            print(f"  {phase}: n={len(ac):,}  |correction| median={1000*np.median(ac):.1f} ms, "
                  f"p95={1000*np.percentile(ac,95):.1f} ms, max={1000*ac.max():.1f} ms")
            print(f"      near the lag edge (|corr| > 0.9*maxlag): "
                  f"{100*(ac > 0.9*CC_MAXLAG).mean():.2f}%   "
                  f"resulting |dt.cc| > 0.2 s production cap: {100*(adt > 0.2).mean():.2f}%")

        out = os.path.join(work, f"{args.array.lower()}_cc_fix_sweep.csv")
        pd.DataFrame(sweep_rows).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
