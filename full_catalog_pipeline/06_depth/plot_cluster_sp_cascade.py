#!/usr/bin/env python3
"""Waveform cascade for a cluster's single-station P/S picks, to visually sanity-check
whether a hypoDD depth spread is corroborated by the actual waveforms. Originally built for
T1 cluster3/DEEJ (see [[t1-sp-vs-depth-vpvs-check-result]]) but generalized -- pass
--ids-file/--station/etc. to run it on any other cluster, station, or hypoDD run.

Each row is one event's waveform at --station, aligned on its P pick (t=0), rendered as a
grayscale raster (pixel intensity = normalized amplitude) rather than overlapping wiggle
traces so ~100+ events are visually tractable at once. Rows are sorted by hypoDD depth so any
real depth moveout in the actual waveforms/picks should appear as a visible top-to-bottom
trend. Each trace is linearly detrended and demeaned before normalizing (these traces commonly
carry a large slow drift that otherwise swamps the P/S wiggle amplitude) and normalized to its
own peak.

Left panel (Z) shows P only. Right panel (horizontal envelope) carries the S markers:
  - actual S pick at --station (from phase.dat)
  - predicted S pick from hypoDD depth + dist via the accepted layer velocity (--vp/--vs,
    not being tested here)
If the picks visually sit on real waveform features (onsets) but track the "actual pick"
marker rather than the "predicted from depth" marker, that argues the depth/velocity model
mismatch is real, not a picking error.

Usage:
    python full_catalog_pipeline/plot_cluster_sp_cascade.py [--ids-file ... --station ... --tag ...]
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

CHUNK_SIZE = 106  # ~100 waveforms/panel per the request

GRID = "#e4e3de"
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def build_panel(df, day_index, args, out_png, panel_label):
    cache = RollingDayCache(day_index, args.station)
    n = len(df)
    n_samp = int(round((args.window_pre + args.window_post) * config.SAMPLE_RATE_HZ))
    img_z = np.zeros((n, n_samp), dtype=np.float32)
    img_h = np.zeros((n, n_samp), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)

    for i, row in enumerate(df.itertuples()):
        p_abs = row.origin + row.p_offset
        w0, w1 = p_abs - args.window_pre, p_abs + args.window_post
        r = {"primary_date": p_abs.date.isoformat()}
        data, status = extract_window(cache, r, w0, w1)
        if data is None or len(data["Z"]) < n_samp:
            continue
        img_z[i, :] = norm_signed(data["Z"][:n_samp])
        img_h[i, :] = horizontal_envelope(data["N"][:n_samp], data["E"][:n_samp])
        ok[i] = True
    print(f"  {panel_label}: {ok.sum()} / {n} events with a complete waveform window")

    dt = 1.0 / config.SAMPLE_RATE_HZ
    xs = np.arange(n_samp) * dt - args.window_pre  # exact sample spacing, not linspace-across-endpoint
    extent = [xs[0], xs[-1], n - 0.5, -0.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(6, n * 0.055)), sharey=True)
    ax1.imshow(img_z, aspect="auto", cmap="gray_r", vmin=-1, vmax=1, extent=extent,
               interpolation="nearest")
    ax2.imshow(img_h, aspect="auto", cmap="gray_r", vmin=0, vmax=1, extent=extent,
               interpolation="nearest")

    sp_pred = predicted_sp(df["depth"].to_numpy(), df["dist_km"].to_numpy(), args.vp, args.vs)
    yy = np.arange(n)

    ax1.axvline(0, color="#2a78d6", linewidth=1, linestyle="--", label="P pick (aligned)")
    ax1.set_xlabel("Time relative to P pick (s)")
    ax1.set_title("Z (vertical) -- P only", fontsize=11)
    ax1.legend(fontsize=7.5, loc="upper right", frameon=True, facecolor="white", framealpha=0.85)
    ax1.set_ylabel("Event (sorted by hypoDD depth, shallow -> deep)")

    ax2.axvline(0, color="#2a78d6", linewidth=1, linestyle="--", label="P pick (aligned)")
    ax2.scatter(df["s_offset"] - df["p_offset"], yy, s=8, color="crimson", zorder=5,
                label="actual S pick")
    ax2.scatter(sp_pred, yy, s=14, marker="x", color="#1a9c5a", zorder=5,
                label=f"predicted S from hypoDD depth\n(accepted Vp/Vs={args.vp/args.vs:.2f})")
    ax2.set_xlabel("Time relative to P pick (s)")
    ax2.set_title("sqrt(HH1^2+HH2^2) (horizontal envelope)", fontsize=11)
    ax2.legend(fontsize=7.5, loc="upper right", frameon=True, facecolor="white", framealpha=0.85)

    depth_ticks_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    ax1.set_yticks(depth_ticks_idx)
    ax1.set_yticklabels([f"{df['depth'].iloc[j]*1000:.0f} m" for j in depth_ticks_idx], fontsize=8)

    fig.suptitle(f"{args.tag} {args.station} waveform cascade -- {panel_label} "
                 f"(n={n}, sorted by hypoDD depth)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    parser = build_common_argparser(__doc__)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                         help="Events per panel. Set to a number >= the cluster size "
                              "(e.g. 100000) to render the whole cascade as a single plot.")
    args = parser.parse_args()
    ids = load_ids(args.ids_file)
    df = load_merged(ids, args.phase_dat, args.reloc, args.station, network=args.network,
                      src_file=args.src, station_sel=args.station_sel)
    print(f"final merged n = {len(df)}, depth range {df['depth'].min():.2f}-{df['depth'].max():.2f} km")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    n_chunks = int(np.ceil(len(df) / args.chunk_size))
    for c in range(n_chunks):
        sub = df.iloc[c * args.chunk_size:(c + 1) * args.chunk_size].reset_index(drop=True)
        if n_chunks == 1:
            label = f"full cascade, shallowest -> deepest"
            out_png = f"{args.out_dir}/{args.tag}_cascade_full.png"
        else:
            label = f"panel {c+1}/{n_chunks} (shallowest)" if c == 0 else f"panel {c+1}/{n_chunks} (deeper)"
            out_png = f"{args.out_dir}/{args.tag}_cascade_panel{c+1}.png"
        build_panel(sub, day_index, args, out_png, label)


if __name__ == "__main__":
    main()
