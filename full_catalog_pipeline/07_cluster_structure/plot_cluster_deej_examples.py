#!/usr/bin/env python3
"""Companion to plot_cluster_deej_stack_check.py: instead of stacking everything into one
trace, show ~10 individual cluster waveforms side by side (aligned on P, detrended + demeaned
+ peak-normalized, same cleaning as the cascade/stack scripts) so waveform shape and pick
consistency can be eyeballed directly rather than only in aggregate.

Events are sampled evenly across the depth-sorted list (not just the shallowest N) so any
depth-dependent change in waveform character would be visible across the panel. Originally
built for T1 cluster3/DEEJ (see [[t1-sp-vs-depth-vpvs-check-result]]) but generalized -- pass
--ids-file/--station/etc. to run it on any other cluster/station/run.

Usage:
    python full_catalog_pipeline/plot_cluster_deej_examples.py [--ids-file ... --station ... --tag ...]
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
from lib.deej_waveform_common import (
    build_common_argparser, load_ids, load_merged, norm_signed, horizontal_envelope, predicted_sp,
)

N_EXAMPLES = 10

GRID = "#e4e3de"
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    p = build_common_argparser(__doc__)
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES,
                   help="Number of events to sample evenly across the depth-sorted list.")
    args = p.parse_args()

    ids = load_ids(args.ids_file)
    df = load_merged(ids, args.phase_dat, args.reloc, args.station, network=args.network,
                      src_file=args.src, station_sel=args.station_sel)

    idx = np.linspace(0, len(df) - 1, args.n_examples).astype(int)
    sample = df.iloc[idx].reset_index(drop=True)

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, args.station)
    n_samp = int(round((args.window_pre + args.window_post) * config.SAMPLE_RATE_HZ))
    dt = 1.0 / config.SAMPLE_RATE_HZ
    xs = np.arange(n_samp) * dt - args.window_pre

    fig, axes = plt.subplots(len(sample), 2, figsize=(13, 1.5 * len(sample)), sharex=True)

    for i, row in enumerate(sample.itertuples()):
        p_abs = row.origin + row.p_offset
        w0, w1 = p_abs - args.window_pre, p_abs + args.window_post
        r = {"primary_date": p_abs.date.isoformat()}
        data, status = extract_window(cache, r, w0, w1)
        ax_z, ax_h = axes[i, 0], axes[i, 1]
        if data is None or len(data["Z"]) < n_samp:
            ax_z.text(0.5, 0.5, f"no data ({status})", transform=ax_z.transAxes, ha="center")
            continue
        z = norm_signed(data["Z"][:n_samp])
        h = horizontal_envelope(data["N"][:n_samp], data["E"][:n_samp])
        sp_obs = row.s_offset - row.p_offset
        sp_pred = predicted_sp(row.depth, row.dist_km, args.vp, args.vs)

        ax_z.plot(xs, z, color="#0b0b0b", linewidth=0.7)
        ax_z.axvline(0, color="#2a78d6", linewidth=1, linestyle="--")
        ax_z.set_ylabel(f"{row.depth*1000:.0f} m", fontsize=8, rotation=0, ha="right", va="center")
        ax_z.set_ylim(-1.05, 1.05)

        ax_h.plot(xs, h, color="#0b0b0b", linewidth=0.7)
        ax_h.axvline(0, color="#2a78d6", linewidth=1, linestyle="--")
        ax_h.axvline(sp_obs, color="crimson", linewidth=1, linestyle="--")
        ax_h.axvline(sp_pred, color="#1a9c5a", linewidth=1, linestyle=":")
        ax_h.set_ylim(-0.05, 1.05)

    axes[0, 0].set_title("Z (vertical) -- P only", fontsize=11)
    axes[0, 1].set_title("sqrt(HH1^2+HH2^2) (horizontal envelope) -- "
                          "actual S pick (red) vs. predicted from hypoDD depth (green)", fontsize=11)
    axes[-1, 0].set_xlabel("Time relative to P pick (s)")
    axes[-1, 1].set_xlabel("Time relative to P pick (s)")

    out_png = f"{args.out_dir}/{args.tag}_examples.png"
    fig.suptitle(f"{args.tag} {args.station} example waveforms (n={len(sample)}, "
                 f"sampled evenly across hypoDD depth {df['depth'].min():.2f}-{df['depth'].max():.2f} km)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
