#!/usr/bin/env python3
"""Test 0 of PLAN_depth_resolvability.md: does MCCC recover known moveout, or erase it?

Every depth conclusion drawn from cc_refine_component.py rests on trusting that iterative
stack-and-realign cross-correlation MEASURES event-to-event timing differences rather than
destroying them. The failure mode to exclude is structural: MCCC aligns every trace to a
common stack, so a procedure that simply collapsed everything onto the stack would report
"no moveout" whatever the truth, and the flat dS-P/dz results would be meaningless.

Method -- injection recovery on REAL waveforms, not synthetics. For a chosen true slope
`m`, each event's window is re-extracted from the archive shifted by dt_i = m*(z_i - z̄),
so its S arrival genuinely sits dt_i away from where its catalog pick says it should. The
shift is applied by moving the EXTRACTION WINDOW by a whole number of samples, so no
interpolation, filtering or resampling touches the waveform. The identical MCCC pipeline
then runs on those traces, and the recovered dS-P/dz is regressed against depth.

  recovered ≈ injected  -> MCCC preserves moveout; a measured flat slope means flat data
  recovered ≈ 0         -> MCCC erases moveout; every CC-based depth conclusion is void

m = 0 is included as the null: it must return ~0, or the harness itself imprints a trend.
The sweep over --max-lag exists because a search half-width narrower than the true moveout
would clip it -- a real and separate way to under-recover.

Usage:
    python full_catalog_pipeline/test_mccc_moveout_recovery.py \
        --ids-file .../t2_cluster1_event_ids.txt --station DRSC --component Z \
        --reloc .../hypoDD.reloc --phase-dat .../phase.dat --station-sel .../station.sel \
        --out-dir ... --tag t2_cluster1_DRSC
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from cc_refine_component import normxcorr_lag
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import (
    build_common_argparser, load_ids, load_events, load_reloc, load_dist_geodetic, clean,
)
from lib.windowing import RollingDayCache, extract_window

INJECTED_SLOPES_MS_KM = [0.0, 65.0, 131.0, 262.0]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def parse_args():
    p = build_common_argparser(__doc__)
    p.add_argument("--component", default="Z", choices=["Z", "N", "E"])
    p.add_argument("--search-pre", type=float, default=0.15)
    p.add_argument("--search-post", type=float, default=0.45)
    p.add_argument("--max-lag", type=float, default=0.15)
    p.add_argument("--max-lag-sweep", type=float, nargs="*", default=[0.10, 0.15, 0.30])
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--min-cc", type=float, default=0.6)
    return p.parse_args()


def mccc(padded, ids_ok, n_core, n_pad, sr, max_lag, iterations):
    lags = {e: 0.0 for e in ids_ok}
    ccs = {e: 0.0 for e in ids_ok}
    for _ in range(iterations):
        stack = np.zeros(n_core)
        for e in ids_ok:
            start = n_pad + int(round(lags[e] * sr))
            stack += padded[e][start:start + n_core]
        stack /= max(len(ids_ok), 1)
        for e in ids_ok:
            lags[e], ccs[e] = normxcorr_lag(padded[e], stack, sr, max_lag)
    return lags, ccs


def main():
    args = parse_args()
    sr = config.SAMPLE_RATE_HZ
    ids = load_ids(args.ids_file)
    df = load_events(ids, args.phase_dat, args.station, network=args.network) \
        .merge(load_reloc(args.reloc), on="id")
    if args.station_sel:
        df = df.merge(load_dist_geodetic(args.reloc, args.station_sel, args.station,
                                         network=args.network), on="id")
    df = df.sort_values("depth").reset_index(drop=True)
    z_mean = df["depth"].mean()
    print(f"[{args.tag}] {len(df)} events, depth {df.depth.min():.2f}-{df.depth.max():.2f} km, "
          f"component {args.component}")

    cache = RollingDayCache(load_day_file_index_csv(config.DAY_FILE_INDEX_CSV), args.station)
    n_core = int(round((args.search_pre + args.search_post) * sr))

    results = []
    for max_lag in args.max_lag_sweep:
        n_pad = int(round(max_lag * sr))
        n_total = n_core + 2 * n_pad
        for m in INJECTED_SLOPES_MS_KM:
            padded, kept = {}, []
            for row in df.itertuples():
                # Whole-sample window shift: the arrival genuinely moves within the extracted
                # window, with no interpolation or filtering applied to the waveform itself.
                dt = (m / 1000.0) * (row.depth - z_mean)
                dt = round(dt * sr) / sr
                s_abs = row.origin + row.s_offset
                w0 = s_abs - args.search_pre - max_lag - dt
                w1 = w0 + (n_total + 4) / sr
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
                kept.append(row.id)
            if len(kept) < 20:
                print(f"[{args.tag}] max_lag={max_lag}s m={m}: only {len(kept)} usable, skipped")
                continue
            lags, ccs = mccc(padded, kept, n_core, n_pad, sr, max_lag, args.iterations)
            sub = df[df["id"].isin(kept)].copy()
            sub["lag"] = sub["id"].map(lags)
            sub["cc"] = sub["id"].map(ccs)
            # Recovered S-P uses the same construction cc_refine_component does, minus the P
            # refinement (P is untouched by the injection, so it cannot affect the slope).
            sub["sp"] = (sub["s_offset"] + sub["lag"]) - sub["p_offset"]
            good = sub[sub["cc"] >= args.min_cc]
            fit = good if len(good) >= 20 else sub
            rec = np.polyfit(fit["depth"], fit["sp"] * 1000, 1)[0]
            resid = fit["sp"].values * 1000 - np.polyval(
                np.polyfit(fit["depth"], fit["sp"] * 1000, 1), fit["depth"].values)
            se = np.std(resid, ddof=2) / (np.std(fit["depth"]) * np.sqrt(len(fit)))
            results.append(dict(max_lag=max_lag, injected=m, recovered=rec, se=se,
                                n=len(fit), mean_cc=float(fit["cc"].mean())))
            print(f"[{args.tag}] max_lag={max_lag:.2f}s  injected {m:+6.0f} ms/km  ->  "
                  f"recovered {rec:+7.1f} ± {se:.1f}  (n={len(fit)}, meanCC={fit['cc'].mean():.3f})")

    res = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, f"{args.tag}_mccc_recovery_{args.component}.csv")
    res.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    for color, (ml, g) in zip(SERIES, res.groupby("max_lag")):
        ax.errorbar(g["injected"], g["recovered"], yerr=g["se"], fmt="o-", color=color,
                    ms=7, lw=1.6, capsize=3, label=f"search half-width ±{ml*1000:.0f} ms")
    lim = [-20, max(INJECTED_SLOPES_MS_KM) * 1.1]
    ax.plot(lim, lim, ls="--", color=INK, lw=1.2, label="perfect recovery (1:1)")
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("injected dS-P/dz (ms/km)")
    ax.set_ylabel("recovered dS-P/dz (ms/km)")
    ax.set_title(f"Test 0 — does MCCC recover known moveout?  ({args.tag}, {args.component})\n"
                 "on the 1:1 line = the method measures moveout;  flat at 0 = it erases it",
                 fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"{args.tag}_mccc_recovery_{args.component}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{args.tag}] wrote {out_png}\n[{args.tag}] wrote {out_csv}")


if __name__ == "__main__":
    main()
