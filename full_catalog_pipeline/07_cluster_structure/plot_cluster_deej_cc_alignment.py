#!/usr/bin/env python3
"""Does cross-correlation (dt.cc) refinement improve S-pick alignment relative to the raw
catalog picks? Solves a self-consistent, CC-informed S-offset per event from the pairwise
dt.cc graph at --station (see lib.deej_waveform_common.solve_cc_consistent_soffset), then
compares it against the original catalog picks two ways:

1. Cascade: two depth-sorted rasters of the SAME events, each re-windowed (not just
   marked) around a different alignment point -- "before" re-extracts each event's window
   centered on its own original catalog S pick, "after" re-extracts it centered on its
   CC-refined S pick. If CC refinement genuinely tightens alignment, the dominant burst
   should look visibly straighter/tighter (less row-to-row jitter) in the "after" raster.
2. Stack comparison: the same two alignments, but stacked (averaged) instead of kept as a
   raster -- quantifies the same effect as peak height / FWHM.

Usage:
    python full_catalog_pipeline/plot_cluster_deej_cc_alignment.py [--ids-file ... --station ... --tag ...]
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
    build_common_argparser, load_ids, load_merged, horizontal_envelope,
    load_dtcc_pairs, solve_cc_consistent_soffset,
)

CHUNK_SIZE = 106
STACK_HALF_WIDTH_S = 0.3  # window around the alignment point for the stack comparison

GRID = "#e4e3de"
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def build_before_after_cascade(df, day_index, args, chunk_size=CHUNK_SIZE):
    """Two depth-sorted rasters of the SAME events: each row is re-windowed (re-extracted
    from the archive, not just re-marked) centered on a different alignment time -- the
    event's original catalog S pick ("before") vs. its CC-refined S pick ("after"). If CC
    refinement tightens alignment, the dominant burst should sit visibly straighter (less
    row-to-row jitter) in the "after" panel."""
    cache = RollingDayCache(day_index, args.station)
    half = STACK_HALF_WIDTH_S
    n_samp = int(round(2 * half * config.SAMPLE_RATE_HZ))
    dt = 1.0 / config.SAMPLE_RATE_HZ
    xs = np.arange(n_samp) * dt - half
    extent_x = [xs[0], xs[-1]]

    n_chunks = int(np.ceil(len(df) / chunk_size))
    for c in range(n_chunks):
        sub = df.iloc[c * chunk_size:(c + 1) * chunk_size].reset_index(drop=True)
        n = len(sub)
        img_before = np.zeros((n, n_samp), dtype=np.float32)
        img_after = np.zeros((n, n_samp), dtype=np.float32)
        ok = np.zeros(n, dtype=bool)

        for i, row in enumerate(sub.itertuples()):
            align_before = row.origin + row.s_offset
            align_after = row.origin + row.s_offset_cc
            rb = {"primary_date": align_before.date.isoformat()}
            ra = {"primary_date": align_after.date.isoformat()}
            data_b, status_b = extract_window(cache, rb, align_before - half, align_before + half)
            data_a, status_a = extract_window(cache, ra, align_after - half, align_after + half)
            if data_b is None or data_a is None:
                continue
            if len(data_b["Z"]) < n_samp or len(data_a["Z"]) < n_samp:
                continue
            img_before[i, :] = horizontal_envelope(data_b["N"][:n_samp], data_b["E"][:n_samp])
            img_after[i, :] = horizontal_envelope(data_a["N"][:n_samp], data_a["E"][:n_samp])
            ok[i] = True
        if n_chunks == 1:
            label = "full cascade, shallowest -> deepest"
        else:
            label = f"panel {c+1}/{n_chunks} (shallowest)" if c == 0 else f"panel {c+1}/{n_chunks} (deeper)"
        print(f"  {label}: {ok.sum()} / {n} events with complete before+after windows")

        extent = extent_x + [n - 0.5, -0.5]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(6, n * 0.055)), sharey=True)
        ax1.imshow(img_before, aspect="auto", cmap="gray_r", vmin=0, vmax=1, extent=extent,
                   interpolation="nearest")
        ax2.imshow(img_after, aspect="auto", cmap="gray_r", vmin=0, vmax=1, extent=extent,
                   interpolation="nearest")
        for ax, title in ((ax1, "BEFORE: aligned on original S pick"),
                           (ax2, "AFTER: aligned on CC-refined S pick")):
            ax.axvline(0, color="#2a78d6", linewidth=1, linestyle="--")
            ax.set_xlabel("Time relative to alignment point (s)")
            ax.set_title(title, fontsize=11)
        ax1.set_ylabel("Event (sorted by hypoDD depth, shallow -> deep)")

        depth_ticks_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
        ax1.set_yticks(depth_ticks_idx)
        ax1.set_yticklabels([f"{sub['depth'].iloc[j]*1000:.0f} m" for j in depth_ticks_idx], fontsize=8)

        fig.suptitle(f"{args.tag} {args.station}: waveform cascade re-aligned before/after CC "
                     f"refinement -- {label} (n={n}, sorted by hypoDD depth)", fontsize=12, y=1.0)
        fig.tight_layout()
        out_png = (f"{args.out_dir}/{args.tag}_cc_alignment_cascade_full.png" if n_chunks == 1
                   else f"{args.out_dir}/{args.tag}_cc_alignment_cascade_panel{c+1}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")


def build_stack_comparison(df, day_index, args):
    cache = RollingDayCache(day_index, args.station)
    n_samp = int(round(2 * STACK_HALF_WIDTH_S * config.SAMPLE_RATE_HZ))
    dt = 1.0 / config.SAMPLE_RATE_HZ
    xs = np.arange(n_samp) * dt - STACK_HALF_WIDTH_S

    stack_before = np.zeros(n_samp)
    stack_after = np.zeros(n_samp)
    n_ok = 0
    for row in df.itertuples():
        align_before = row.origin + row.s_offset
        align_after = row.origin + row.s_offset_cc
        r_before = {"primary_date": align_before.date.isoformat()}
        r_after = {"primary_date": align_after.date.isoformat()}
        data_b, status_b = extract_window(cache, r_before, align_before - STACK_HALF_WIDTH_S,
                                           align_before + STACK_HALF_WIDTH_S)
        data_a, status_a = extract_window(cache, r_after, align_after - STACK_HALF_WIDTH_S,
                                           align_after + STACK_HALF_WIDTH_S)
        if data_b is None or data_a is None:
            continue
        if len(data_b["Z"]) < n_samp or len(data_a["Z"]) < n_samp:
            continue
        h_b = horizontal_envelope(data_b["N"][:n_samp], data_b["E"][:n_samp])
        h_a = horizontal_envelope(data_a["N"][:n_samp], data_a["E"][:n_samp])
        stack_before += h_b
        stack_after += h_a
        n_ok += 1
    stack_before /= n_ok
    stack_after /= n_ok
    print(f"stack comparison: n_ok = {n_ok} / {len(df)}")

    def peak_width_at_half_max(y):
        peak = y.max()
        above = y >= peak / 2
        idx = np.where(above)[0]
        return (idx.max() - idx.min()) * dt, peak

    fwhm_b, peak_b = peak_width_at_half_max(stack_before)
    fwhm_a, peak_a = peak_width_at_half_max(stack_after)
    print(f"  before-CC stack: peak={peak_b:.3f}, FWHM={fwhm_b*1000:.0f} ms")
    print(f"  after-CC  stack: peak={peak_a:.3f}, FWHM={fwhm_a*1000:.0f} ms")
    verdict = ("CC alignment SHARPENED the stack" if (peak_a > peak_b and fwhm_a < fwhm_b)
               else "CC alignment did NOT clearly sharpen the stack")
    print(f"  -> {verdict}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, sharey=True)
    axes[0].plot(xs, stack_before, color="crimson", linewidth=1.2)
    axes[0].axvline(0, color="#0b0b0b", linewidth=1, linestyle="--")
    axes[0].set_title(f"Aligned on ORIGINAL S pick (peak={peak_b:.2f}, FWHM={fwhm_b*1000:.0f} ms)")
    axes[0].set_ylabel("mean horizontal envelope")

    axes[1].plot(xs, stack_after, color="darkmagenta", linewidth=1.2)
    axes[1].axvline(0, color="#0b0b0b", linewidth=1, linestyle="--")
    axes[1].set_title(f"Aligned on CC-REFINED S pick (peak={peak_a:.2f}, FWHM={fwhm_a*1000:.0f} ms)")
    axes[1].set_ylabel("mean horizontal envelope")
    axes[1].set_xlabel("Time relative to alignment point (s)")

    fig.suptitle(f"{args.tag} {args.station}: does CC refinement sharpen the stack? "
                 f"({verdict}, n={n_ok})", fontsize=12, y=1.0)
    fig.tight_layout()
    out_png = f"{args.out_dir}/{args.tag}_cc_stack_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    p = build_common_argparser(__doc__)
    p.add_argument("--anchor-weight", type=float, default=0.05,
                   help="Per-event anchor weight pulling the CC solution back to the "
                        "original catalog pick (fixes the gauge; also means a bias shared "
                        "by ALL picks alike cannot be corrected by this method).")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help="Events per cascade panel. Set >= cluster size for one combined plot.")
    args = p.parse_args()

    ids = load_ids(args.ids_file)
    df = load_merged(ids, args.phase_dat, args.reloc, args.station, network=args.network,
                      src_file=args.src, station_sel=args.station_sel)

    pairs = load_dtcc_pairs(ids, args.dtcc, args.station, "S", network=args.network)
    print(f"dt.cc S-phase pairs at {args.station} within this cluster: {len(pairs)}")
    cc_s_offset = solve_cc_consistent_soffset(df[["id", "s_offset"]], pairs,
                                               anchor_weight=args.anchor_weight)
    df["s_offset_cc"] = df["id"].map(cc_s_offset)
    n_missing = df["s_offset_cc"].isna().sum()
    if n_missing:
        print(f"  {n_missing} events had no CC solution (no pairwise links); dropping them")
        df = df.dropna(subset=["s_offset_cc"]).reset_index(drop=True)

    orig_sp = df["s_offset"] - df["p_offset"]
    cc_sp = df["s_offset_cc"] - df["p_offset"]
    print(f"original S-P:    mean={orig_sp.mean():.4f} s, std={orig_sp.std()*1000:.1f} ms")
    print(f"CC-refined S-P:  mean={cc_sp.mean():.4f} s, std={cc_sp.std()*1000:.1f} ms")
    print(f"per-event shift (CC - original): mean={((cc_sp-orig_sp).mean()*1000):+.1f} ms, "
          f"std={(cc_sp-orig_sp).std()*1000:.1f} ms")

    import scipy.stats as st
    r_orig, p_orig = st.pearsonr(df["depth"], orig_sp)
    r_cc, p_cc = st.pearsonr(df["depth"], cc_sp)
    rho_orig, _ = st.spearmanr(df["depth"], orig_sp)
    rho_cc, _ = st.spearmanr(df["depth"], cc_sp)
    print(f"\ndepth vs S-P (catalog picks):     r={r_orig:+.3f} (p={p_orig:.2g}), rho={rho_orig:+.3f}")
    print(f"depth vs S-P (CC-refined, waveform-only): r={r_cc:+.3f} (p={p_cc:.2g}), rho={rho_cc:+.3f}")
    depth_range_m = (df["depth"].max() - df["depth"].min()) * 1000
    print(f"depth range: {depth_range_m:.0f} m; CC-refined S-P range: "
          f"{(cc_sp.max()-cc_sp.min())*1000:.0f} ms, std: {cc_sp.std()*1000:.1f} ms")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    build_before_after_cascade(df, day_index, args, chunk_size=args.chunk_size)
    build_stack_comparison(df, day_index, args)


if __name__ == "__main__":
    main()
