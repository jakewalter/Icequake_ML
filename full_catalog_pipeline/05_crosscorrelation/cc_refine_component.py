#!/usr/bin/env python3
"""From-scratch, single-component cross-correlation refinement of each event's S arrival
time at one station -- NOT a reuse of hypoDD's own dt.cc (different quality threshold, and
hypoDD's S dt.cc mixes Z/N/E per its own internal weighting), and NOT an envelope
(sqrt(N^2+E^2), which discards sign/polarity and can smear true onset timing) -- operates on
ONE raw signed component at a time (Z, N, or E; N/E are this deployment's HH1/HH2, NOT
verified geographic components, see cluster3_orientation_calibration.py).

Method: iterative multi-channel cross-correlation (MCCC)-style stack refinement.
1. Extract a padded window around each event's catalog S pick.
2. Build an initial reference stack (mean of all events' windows, aligned on the catalog pick).
3. For each event, slide its trace against the stack within +/-max-lag and find the lag of
   maximum SIGNED normalized cross-correlation (not absolute value -- a genuinely
   reverse-polarity event should score poorly and get filtered by --min-cc, not "matched" by
   flipping sign).
4. Rebuild the stack from lag-corrected traces and repeat --iterations times.
5. Report each event's final correlation coefficient (a genuine per-event quality metric,
   independent of and typically stricter than hypoDD's fixed 0.4 floor) and refined S-offset.

Usage:
    python full_catalog_pipeline/cc_refine_component.py --ids-file ... --station WICH \
        --component Z --min-cc 0.6 --tag cluster1_wich_Z
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

import os

import numpy as np
import pandas as pd

import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window
from lib.deej_waveform_common import (
    build_common_argparser, load_ids, load_events, load_reloc, load_dist,
    load_dist_geodetic, clean, predicted_sp,
)


def parse_args():
    p = build_common_argparser(__doc__)
    p.add_argument("--component", default="Z", choices=["Z", "N", "E"],
                   help="Single raw component to use (no envelope combination).")
    p.add_argument("--search-pre", type=float, default=0.15,
                   help="Seconds before the catalog S pick included in the correlated window.")
    p.add_argument("--search-post", type=float, default=0.45,
                   help="Seconds after the catalog S pick included in the correlated window.")
    p.add_argument("--max-lag", type=float, default=0.15, help="Max search lag, seconds.")
    p.add_argument("--iterations", type=int, default=3, help="Stack-rebuild iterations.")
    p.add_argument("--refine-p", action="store_true",
                   help="Also CC-refine the P arrival, so S-P is measured entirely from the "
                        "waveforms rather than half from the catalog pick table. Off by "
                        "default to preserve the original T1 behaviour (S refined, P as "
                        "picked).")
    p.add_argument("--min-cc", type=float, default=0.6,
                   help="Minimum final correlation coefficient for an event to count as "
                        "well-aligned in the reported depth-correlation statistics.")
    return p.parse_args()


def normxcorr_lag(trace, template, sr, max_lag_s):
    """Lag (seconds, +ve = trace delayed relative to template) of the SIGNED normalized
    cross-correlation peak within +/-max_lag_s. `trace` must be longer than `template` by at
    least 2*max_lag samples (padding on both sides)."""
    max_lag_n = int(round(max_lag_s * sr))
    n_t = len(template)
    template_c = template - template.mean()
    template_norm = np.linalg.norm(template_c)
    best_lag, best_cc = 0, -np.inf
    for lag in range(-max_lag_n, max_lag_n + 1):
        start = max_lag_n + lag
        seg = trace[start:start + n_t]
        seg_c = seg - seg.mean()
        denom = np.linalg.norm(seg_c) * template_norm
        cc = float(np.dot(seg_c, template_c) / denom) if denom > 0 else -np.inf
        if cc > best_cc:
            best_cc, best_lag = cc, lag
    return best_lag / sr, best_cc


def main():
    args = parse_args()
    ids = load_ids(args.ids_file)
    events = load_events(ids, args.phase_dat, args.station, network=args.network)
    reloc = load_reloc(args.reloc)
    df = events.merge(reloc, on="id")
    dist = None
    if args.src and os.path.exists(args.src):
        dist = load_dist(ids, args.src, args.station, network=args.network)
        if dist.empty or not dist["id"].isin(df["id"]).any():
            dist = None       # per-cluster src that does not cover these events
    if dist is None and args.station_sel:
        dist = load_dist_geodetic(args.reloc, args.station_sel, args.station,
                                  network=args.network)
    if dist is not None:
        df = df.merge(dist, on="id")
    print(f"[{args.tag}] {len(df)} events with P+S picks at {args.station}, component={args.component}")

    sr = config.SAMPLE_RATE_HZ
    cache = RollingDayCache(load_day_file_index_csv(config.DAY_FILE_INDEX_CSV), args.station)

    n_core = int(round((args.search_pre + args.search_post) * sr))
    n_pad = int(round(args.max_lag * sr))
    n_total = n_core + 2 * n_pad

    def mccc(pick_col, label):
        """Iterative stack-and-realign (MCCC) around `pick_col`. Returns (lags, ccs) keyed by
        event id, in seconds relative to that catalog pick."""
        padded = {}
        for row in df_all.itertuples():
            t_abs = row.origin + getattr(row, pick_col)
            w0 = t_abs - args.search_pre - args.max_lag
            w1 = t_abs + args.search_post + args.max_lag
            r = {"primary_date": w0.date.isoformat()}
            data, status = extract_window(cache, r, w0, w1)
            cache.evict_before(r["primary_date"])
            if data is None or len(data[args.component]) < n_total:
                continue
            tr = clean(data[args.component][:n_total])
            peak = np.max(np.abs(tr))
            if peak == 0:
                continue
            padded[row.id] = tr / peak
        ids_ok = [e for e in df_all["id"] if e in padded]
        print(f"[{args.tag}] {label}: {len(ids_ok)} events with a usable waveform window")
        lags = {e: 0.0 for e in ids_ok}
        ccs = {e: 0.0 for e in ids_ok}
        for it in range(args.iterations):
            stack = np.zeros(n_core)
            for e in ids_ok:
                start = n_pad + int(round(lags[e] * sr))
                stack += padded[e][start:start + n_core]
            stack /= max(len(ids_ok), 1)
            for e in ids_ok:
                # Always searched against the ORIGINAL pick position (padded[] is never
                # re-extracted), so each iteration yields an absolute lag, not an increment --
                # overwrite, never accumulate. Only the stack sharpens between iterations.
                lags[e], ccs[e] = normxcorr_lag(padded[e], stack, sr, args.max_lag)
            print(f"[{args.tag}] {label} iteration {it+1}/{args.iterations}: "
                  f"mean|lag|={np.mean([abs(v) for v in lags.values()])*1000:.1f}ms, "
                  f"mean CC={np.mean(list(ccs.values())):.3f}")
        return lags, ccs

    df_all = df
    s_lags, s_ccs = mccc("s_offset", "S")
    if args.refine_p:
        p_lags, p_ccs = mccc("p_offset", "P")
    else:
        p_lags, p_ccs = {}, {}

    keep = [e for e in df["id"] if e in s_lags and (not args.refine_p or e in p_lags)]
    df = df[df["id"].isin(keep)].reset_index(drop=True)
    lag_total = s_lags
    cc_final = s_ccs

    df["lag_s"] = df["id"].map(lag_total)
    df["cc"] = df["id"].map(cc_final)
    df["s_offset_refined"] = df["s_offset"] + df["lag_s"]
    df["p_lag_s"] = df["id"].map(p_lags) if args.refine_p else 0.0
    df["p_cc"] = df["id"].map(p_ccs) if args.refine_p else np.nan
    df["p_offset_refined"] = df["p_offset"] + df["p_lag_s"]
    # With --refine-p this S-P comes entirely from waveform cross-correlation; without it the
    # P side is still the catalog pick.
    df["sp_refined"] = df["s_offset_refined"] - df["p_offset_refined"]
    df["sp_original"] = df["s_offset"] - df["p_offset"]

    import scipy.stats as st
    for label, sub in [("all events", df), (f"CC>={args.min_cc} only", df[df["cc"] >= args.min_cc])]:
        if len(sub) < 5:
            print(f"[{args.tag}] {label}: n={len(sub)}, too few for stats")
            continue
        r_o, p_o = st.pearsonr(sub["depth"], sub["sp_original"])
        r_r, p_r = st.pearsonr(sub["depth"], sub["sp_refined"])
        print(f"[{args.tag}] {label} (n={len(sub)}): "
              f"depth vs original S-P r={r_o:+.3f}(p={p_o:.1g})  "
              f"depth vs REFINED S-P r={r_r:+.3f}(p={p_r:.1g})")
        # The slope, not just the correlation: r says whether S-P moves WITH depth, the slope
        # says whether it moves BY the amount the velocity model requires. A depth spread
        # invented by the depth/origin-time trade-off gives slope ~0 however tight r is.
        if "dist_km" in sub.columns and sub["depth"].std() > 0.01:
            obs = np.polyfit(sub["depth"], sub["sp_refined"] * 1000, 1)[0]
            pred_sp_ms = predicted_sp(sub["depth"].values, sub["dist_km"].values,
                                      args.vp, args.vs) * 1000
            pred = np.polyfit(sub["depth"], pred_sp_ms, 1)[0]
            resid = sub["sp_refined"].values * 1000 - np.polyval(
                np.polyfit(sub["depth"], sub["sp_refined"] * 1000, 1), sub["depth"].values)
            se = np.std(resid, ddof=2) / (np.std(sub["depth"]) * np.sqrt(len(sub)))
            print(f"[{args.tag}] {label}: dS-P/dz observed(CC) = {obs:+.0f} +/- {se:.0f} ms/km, "
                  f"model requires {pred:+.0f} ms/km  ->  ratio {obs/pred if pred else float('nan'):.2f}")

    out_csv = f"{args.out_dir}/{args.tag}_cc_refine_{args.component}.csv"
    cols = ["id", "depth", "sp_original", "sp_refined", "lag_s", "cc", "p_lag_s", "p_cc"]
    if "dist_km" in df.columns:
        cols.append("dist_km")
    df[cols].to_csv(out_csv, index=False)
    print(f"[{args.tag}] wrote {out_csv}")
    return df


if __name__ == "__main__":
    main()
