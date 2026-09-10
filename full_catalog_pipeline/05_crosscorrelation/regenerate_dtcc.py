#!/usr/bin/env python3
"""Regenerate dt.cc from scratch with the upsampling fix, using a prefetch architecture.

Replaces hypoDDpy's `_cross_correlate_picks` stage. Two independent reasons to do this rather
than just install the shim in hypodd_relocate.py and re-run hypoDDpy:

  * Correctness. At 200 Hz, obspy's xcorr_pick_correction refuses to fit its sub-sample
    parabola for 99.4% of P attempts ("Less than 3 samples selected for fit"), so T2's dt.cc
    came out 97.8% S while the picks and dt.ct are balanced. Interpolating x2 before the
    correlation takes the P pass rate from 0.6% to 99.8% -- see diagnose_cc_p_deficit.py and
    plot_cc_upsampling_explainer.py for the mechanism and the measurements.
  * Cost. hypoDDpy loads waveforms per PICK PAIR across 488,438 pairs, which is what made the
    original run take ~10 h. The I/O is actually bounded by DISTINCT (event, station, phase)
    windows -- exactly the number of picks in phase.dat, 31,937 for T2 -- so extracting each
    once, in date order, reduces the archive pass to minutes.

The same prefetch also lets each window be interpolated, demeaned, tapered and bandpassed ONCE
instead of once per correlation. obspy does that preparation inside every call; hoisting it out
removes ~14M interpolate+filter operations from the inner loop, leaving only the correlation
and the parabola fit. `--verify` proves the hoist is exact by running both paths on real pairs
and comparing.

hypoDDpy's output semantics are reproduced exactly:
  * pairs come from dt.ct's header lines, in that order
  * a pair/station/phase yields at most ONE observation, from the first offered channel whose
    coefficient clears the threshold (P is offered Z; S is offered Z, E, N -- hypoDDpy's
    cc_p/cc_s_phase_weighting)
  * dt = (tt1 - tt2) - pick2_corr, where tt are phase.dat travel times. This is hypoDDpy's
    `(pick1 - origin1) - (pick2 + pick2_corr - origin2)` rewritten in travel-time form, so no
    absolute origin time is needed.
  * line format `NET.STA  dt  coeff  PHASE`, pair header `# ev1  ev2 0.0`

Deliberate differences, both narrowing rather than widening the output:
  * picks come from phase.dat, not the QuakeML catalog. hypoDDpy drops some picks when writing
    phase.dat ("Negative absolute travel time ... will not be used"); those stay dropped here.
  * pairs with no surviving observation are omitted rather than written as a bare header
    (hypoDDpy left 43,308 such headers in T2's dt.cc; hypodd_tune.filter_dtcc drops them too).

Writes to a NEW file and never touches dt.cc or dt.cc.authoritative: the raw dt.cc is
hardlinked across ~27 sweep directories and is deliberately kept unfiltered, and the
authoritative filtered copy is what the promoted relocation reads (see PROVENANCE.md). Applying
the |dt| / coefficient caps stays a separate step -- hypodd_tune.filter_dtcc.

Usage:
    python full_catalog_pipeline/regenerate_dtcc.py --array T2 --verify
    python full_catalog_pipeline/regenerate_dtcc.py --array T2 --limit-pairs 5000
    python full_catalog_pipeline/regenerate_dtcc.py --array T2 --workers 20
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
import time
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from obspy import Trace
from obspy.signal.cross_correlation import xcorr_pick_correction
from obspy.signal.invsim import cosine_taper

import catalog_paths
import config
from diagnose_cc_p_deficit import (
    CC_TIME_BEFORE, CC_TIME_AFTER, CC_MAXLAG, CC_FMIN, CC_FMAX, CC_MIN_COEFF,
    P_CHANNELS, S_CHANNELS, load_phase_dat,
)
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window

PAD_BEFORE = max(CC_TIME_BEFORE * 2, 2.0)     # hypoDDpy's load margins
PAD_AFTER = max(CC_TIME_AFTER * 2, 2.0)
CHUNK = 2000

_WIN = {}          # {(event, sta, phase): {chan: Trace}} -- prepared, shared via fork
_TT = {}           # {(event, sta, phase): travel time (s)}


def prepare(arr, t0, sr, upsample):
    """One window put through obspy's in-call preparation, once and for all.

    Mirrors xcorr_pick_correction's own sequence exactly -- float64, demean, 10% cosine taper,
    bandpass -- with the interpolation ahead of it. Traces prepared this way are passed with
    filter=None, so obspy slices them and does nothing else.
    """
    tr = Trace(np.asarray(arr, float))
    tr.stats.sampling_rate = sr
    tr.stats.starttime = t0
    if upsample > 1:
        tr.interpolate(sampling_rate=sr * upsample, method="lanczos", a=20)
    tr.data = tr.data.astype(np.float64)
    tr.detrend(type="demean")
    tr.data *= cosine_taper(len(tr), 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr.filter("bandpass", freqmin=CC_FMIN, freqmax=CC_FMAX)
    return tr


def prefetch(needed, day_index, upsample, sr):
    """Extract and prepare every needed window once, walking each station in date order."""
    by_station = {}
    for (ev, sta, phase), t in needed.items():
        by_station.setdefault(sta, []).append((t, ev, phase))
    chans_for = {"P": P_CHANNELS, "S": S_CHANNELS}
    out, t_start, done = {}, time.time(), 0
    for sta in sorted(by_station):
        entries = sorted(by_station[sta], key=lambda x: x[0])
        cache = RollingDayCache(day_index, sta)
        got = 0
        for t, ev, phase in entries:
            r = {"primary_date": (t - PAD_BEFORE).date.isoformat()}
            data, _ = extract_window(cache, r, t - PAD_BEFORE, t + PAD_AFTER)
            cache.evict_before(r["primary_date"])
            done += 1
            if data is None:
                continue
            prepared = {}
            for c in chans_for[phase]:
                a = data.get(c)
                if a is not None and len(a) > 10 and np.std(a) > 0:
                    prepared[c] = prepare(a, t - PAD_BEFORE, sr, upsample)
            if prepared:
                out[(ev, sta, phase)] = prepared
                got += 1
        print(f"    {sta}: {got}/{len(entries)} windows  "
              f"({done}/{len(needed)} overall, {time.time()-t_start:.0f}s)", flush=True)
    return out


def _pick_time(key):
    """The pick time inside a prepared window: its start plus the load margin."""
    trs = _WIN[key]
    return next(iter(trs.values())).stats.starttime + PAD_BEFORE


def correlate_pair(args_pair):
    """All observations for one event pair, as dt.cc lines. Runs in a worker process."""
    e1, e2, shared, network = args_pair
    lines = []
    for sta, phase in shared:
        k1, k2 = (e1, sta, phase), (e2, sta, phase)
        if k1 not in _WIN or k2 not in _WIN:
            continue
        t1, t2 = _pick_time(k1), _pick_time(k2)
        for chan in (P_CHANNELS if phase == "P" else S_CHANNELS):
            tr1, tr2 = _WIN[k1].get(chan), _WIN[k2].get(chan)
            if tr1 is None or tr2 is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, coeff = xcorr_pick_correction(
                        t1, tr1, t2, tr2, t_before=CC_TIME_BEFORE, t_after=CC_TIME_AFTER,
                        cc_maxlag=CC_MAXLAG, filter=None, plot=False)
            except Exception:
                continue
            if coeff is None or not np.isfinite(coeff) or not np.isfinite(corr):
                continue
            coeff = max(-1.0, min(1.0, float(coeff)))
            if coeff < CC_MIN_COEFF:
                continue                      # hypoDDpy: try the next offered channel
            dt = (_TT[k1] - _TT[k2]) - float(corr)
            lines.append(f"{network}.{sta} {dt:.6f} {coeff:.4f} {phase}")
            break                             # first passing channel wins
    return e1, e2, lines


def verify(ph, pairs, win, upsample, sr, day_index, n=40):
    """Prove the hoisted preparation is exact: prepared+filter=None must equal raw+filter."""
    print(f"\nverifying the prepared-window shortcut on {n} real correlations ...")
    chans_for = {"P": P_CHANNELS, "S": S_CHANNELS}
    checked = worst = 0
    for e1, e2 in pairs:
        if checked >= n:
            break
        if e1 not in ph or e2 not in ph:
            continue
        for key in [k for k in ph[e1] if isinstance(k, tuple) and k in ph[e2]]:
            if checked >= n:
                break
            sta, phase = key
            k1, k2 = (e1, sta, phase), (e2, sta, phase)
            if k1 not in win or k2 not in win:
                continue
            chan = chans_for[phase][0]
            if chan not in win[k1] or chan not in win[k2]:
                continue
            t1 = win[k1][chan].stats.starttime + PAD_BEFORE
            t2 = win[k2][chan].stats.starttime + PAD_BEFORE
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fast = xcorr_pick_correction(
                        t1, win[k1][chan], t2, win[k2][chan], t_before=CC_TIME_BEFORE,
                        t_after=CC_TIME_AFTER, cc_maxlag=CC_MAXLAG, filter=None, plot=False)
            except Exception:
                continue
            # the reference path: re-extract raw, interpolate only, let obspy filter
            raw = {}
            for tag, (ev, t) in (("1", (e1, t1)), ("2", (e2, t2))):
                cache = RollingDayCache(day_index, sta)
                r = {"primary_date": (t - PAD_BEFORE).date.isoformat()}
                data, _ = extract_window(cache, r, t - PAD_BEFORE, t + PAD_AFTER)
                if data is None or data.get(chan) is None:
                    raw = None
                    break
                tr = Trace(np.asarray(data[chan], float))
                tr.stats.sampling_rate = sr
                tr.stats.starttime = t - PAD_BEFORE
                if upsample > 1:
                    tr.interpolate(sampling_rate=sr * upsample, method="lanczos", a=20)
                raw[tag] = tr
            if not raw:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ref = xcorr_pick_correction(
                        t1, raw["1"], t2, raw["2"], t_before=CC_TIME_BEFORE,
                        t_after=CC_TIME_AFTER, cc_maxlag=CC_MAXLAG, filter="bandpass",
                        filter_options={"freqmin": CC_FMIN, "freqmax": CC_FMAX}, plot=False)
            except Exception:
                continue
            worst = max(worst, abs(fast[0] - ref[0]), abs(fast[1] - ref[1]))
            checked += 1
    print(f"  {checked} correlations compared, worst absolute difference "
          f"{worst:.3e} (correction in s, coefficient dimensionless)")
    if checked and worst > 1e-9:
        raise SystemExit("prepared-window shortcut is NOT equivalent -- aborting")
    print("  equivalent." if checked else "  nothing comparable found")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--upsample", type=int, default=2)
    ap.add_argument("--network", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    sr = config.SAMPLE_RATE_HZ
    work = catalog_paths.work_dir(args.array)
    network = args.network or ("7U" if args.array == "T2" else "2E")
    out_path = args.out or os.path.join(work, "input_files", "dt.cc.upsampled")
    for guard in ("dt.cc", "dt.cc.authoritative"):
        if os.path.abspath(out_path) == os.path.abspath(
                os.path.join(work, "input_files", guard)):
            raise SystemExit(f"refusing to overwrite {guard}")

    ph = load_phase_dat(catalog_paths.phase_dat(args.array))
    dt_ct = os.path.join(work, "input_files", "dt.ct")
    pairs = []
    with open(dt_ct) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                pairs.append((int(p[1]), int(p[2])))
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    print(f"{args.array}: {len(pairs):,} event pairs from dt.ct, "
          f"upsample x{args.upsample}, network {network}, {args.workers} workers")

    # work list + the windows it needs
    work_items, needed = [], {}
    for e1, e2 in pairs:
        if e1 not in ph or e2 not in ph:
            continue
        shared = [k for k in ph[e1]
                  if isinstance(k, tuple) and k[1] in ("P", "S") and k in ph[e2]]
        if not shared:
            continue
        work_items.append((e1, e2, shared, network))
        for sta, phase in shared:
            needed[(e1, sta, phase)] = ph[e1]["origin"] + ph[e1][(sta, phase)]
            needed[(e2, sta, phase)] = ph[e2]["origin"] + ph[e2][(sta, phase)]
    print(f"  {len(work_items):,} pairs with shared picks; "
          f"{len(needed):,} distinct windows to extract")

    t0 = time.time()
    print("\nprefetching and preparing windows (one pass over the archive, date-ordered) ...")
    win = prefetch(needed, load_day_file_index_csv(config.DAY_FILE_INDEX_CSV),
                   args.upsample, sr)
    print(f"  prepared {len(win):,}/{len(needed):,} windows in {time.time()-t0:.0f}s")

    tt = {k: ph[k[0]][(k[1], k[2])] for k in needed}
    if args.verify:
        verify(ph, pairs, win, args.upsample,
               sr, load_day_file_index_csv(config.DAY_FILE_INDEX_CSV))

    print(f"\ncorrelating {len(work_items):,} pairs ...")
    t1 = time.time()
    written = {"P": 0, "S": 0}
    dts, coeffs = [], []
    n_pairs_kept = 0
    # The prepared windows are ~850 MB for a full T2 run. They reach the workers by fork
    # copy-on-write, NOT as initargs: passing them as initargs would pickle the whole dict
    # once per worker (~17 GB at 20 workers). Hence the globals and the explicit fork context
    # -- under "spawn" the children would not inherit them and every pair would return empty.
    global _WIN, _TT
    _WIN, _TT = win, tt
    with open(out_path, "w") as fout, ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("fork")) as pool:
        for i, (e1, e2, lines) in enumerate(
                pool.map(correlate_pair, work_items, chunksize=CHUNK), 1):
            if lines:
                fout.write(f"# {e1}  {e2} 0.0\n")
                fout.write("\n".join(lines) + "\n")
                n_pairs_kept += 1
                for ln in lines:
                    f = ln.split()
                    written[f[3]] += 1
                    dts.append(abs(float(f[1])))
                    coeffs.append(float(f[2]))
            if i % 50000 == 0:
                rate = i / (time.time() - t1)
                print(f"    {i:,}/{len(work_items):,} pairs  ({rate:.0f}/s, "
                      f"eta {(len(work_items)-i)/rate/60:.0f} min)", flush=True)

    total = written["P"] + written["S"]
    print(f"\nwrote {out_path}")
    print(f"  {n_pairs_kept:,} pairs with data, {total:,} observations: "
          f"P={written['P']:,} ({100*written['P']/max(total,1):.1f}%), "
          f"S={written['S']:,} ({100*written['S']/max(total,1):.1f}%)")
    if dts:
        d, c = np.array(dts), np.array(coeffs)
        print(f"  |dt|: median={1000*np.median(d):.1f} ms, p99={1000*np.percentile(d,99):.1f} ms, "
              f"p99.9={1000*np.percentile(d,99.9):.1f} ms, max={d.max():.3f} s")
        print(f"  beyond the 0.2 s production cap: {100*(d>0.2).mean():.4f}%  "
              f"(old dt.cc had entries to 232 s)")
        print(f"  coefficient: median={np.median(c):.3f}, "
              f"fraction >=0.5: {100*(c>=0.5).mean():.1f}%")
    print(f"  correlation stage took {(time.time()-t1)/60:.1f} min; "
          f"total {(time.time()-t0)/60:.1f} min")
    print("\nNOT installed as an input. To use it, filter and point hypoDD.inp at the result:")
    print("  python -c \"import sys; sys.path.insert(0,'full_catalog_pipeline'); "
          "from hypodd_tune import filter_dtcc; "
          f"print(filter_dtcc('{out_path}', '{out_path}.filtered', 0.2, 0.5))\"")


if __name__ == "__main__":
    main()
