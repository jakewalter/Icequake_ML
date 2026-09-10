#!/usr/bin/env python3
"""Follow-up to plot_cluster_sp_cascade.py: stack all of a cluster's single-station waveforms
aligned on their P pick (no depth-based sorting/selection) to check whether the S arrival is
smeared out over time the way a genuine hypoDD depth spread should, or whether it stacks up
sharp and coherent (implying the population is more tightly clustered in depth than hypoDD's
DD-inversion reports). Each trace is linearly detrended and demeaned before normalizing to its
own peak. Originally built for T1 cluster3/DEEJ (see [[t1-sp-vs-depth-vpvs-check-result]]) but
generalized -- pass --ids-file/--station/etc. to run it on any other cluster/station/run.

Usage:
    python full_catalog_pipeline/plot_cluster_deej_stack_check.py [--ids-file ... --station ... --tag ...]
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window
from lib.deej_waveform_common import build_common_argparser, load_ids, load_merged, clean, horizontal_envelope


def main():
    args = build_common_argparser(__doc__).parse_args()
    ids = load_ids(args.ids_file)
    df = load_merged(ids, args.phase_dat, args.reloc, args.station, network=args.network)

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, args.station)
    n_samp = int(round((args.window_pre + args.window_post) * config.SAMPLE_RATE_HZ))

    z_stack = np.zeros(n_samp)
    h_stack = np.zeros(n_samp)
    n_ok = 0
    for row in df.itertuples():
        p_abs = row.origin + row.p_offset
        w0, w1 = p_abs - args.window_pre, p_abs + args.window_post
        r = {"primary_date": p_abs.date.isoformat()}
        data, status = extract_window(cache, r, w0, w1)
        if data is None or len(data["Z"]) < n_samp:
            continue
        z = clean(data["Z"][:n_samp])
        h = horizontal_envelope(data["N"][:n_samp], data["E"][:n_samp])
        zpk = np.max(np.abs(z)) or 1.0
        z_stack += np.abs(z) / zpk
        h_stack += h
        n_ok += 1
    z_stack /= n_ok
    h_stack /= n_ok
    print(f"n_ok = {n_ok} / {len(df)}")

    dt = 1.0 / config.SAMPLE_RATE_HZ
    xs = np.arange(n_samp) * dt - args.window_pre
    median_sp = (df["s_offset"] - df["p_offset"]).median()
    out_png = f"{args.out_dir}/{args.tag}_stack_check.png"

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(xs, z_stack, color="#0b0b0b", linewidth=1.2)
    axes[0].axvline(0, color="#2a78d6", linestyle="--", linewidth=1, label="P")
    axes[0].axvline(median_sp, color="crimson", linestyle="--", linewidth=1, label="median actual S pick")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("mean |Z| (stacked, per-trace normalized)")
    axes[0].set_title(f"{args.tag} {args.station}: stacked |Z| envelope, n={n_ok} events aligned on P "
                       f"(hypoDD depth range {df['depth'].min():.2f}-{df['depth'].max():.2f} km)")

    axes[1].plot(xs, h_stack, color="#0b0b0b", linewidth=1.2)
    axes[1].axvline(0, color="#2a78d6", linestyle="--", linewidth=1)
    axes[1].axvline(median_sp, color="crimson", linestyle="--", linewidth=1)
    axes[1].set_ylabel("mean horizontal envelope (stacked)")
    axes[1].set_xlabel("Time relative to P pick (s)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
