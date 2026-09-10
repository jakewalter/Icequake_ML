#!/usr/bin/env python3
"""Investigation 2 (see t1_composite_focal_mech_and_vpvs_consistency_plan memory):
real depth separation vs. location-error artifact check, for T1 pyocto's already-vetted
n=216 basal cluster ("cluster3"), using DEEJ (its single dominant near station, median
~0.5 km horizontal offset, next-closest ~4.3 km -- see memory).

This is NOT a Wadati-style test of whether Vp/Vs=1.73 holds -- that ratio is ACCEPTED as
given. The question is instead: are cluster3's events actually separated in depth by up to
~940 m (as hypoDD's double-difference inversion reports), or does that spread reflect
location error (e.g. a depth/origin-time trade-off in the DD inversion) with the events
really sitting at nearly the same depth?

Because these events are all nearly on top of one another horizontally (DEEJ is <1 km away
for all of them, vs. the ~940 m depth range in question), DEEJ's S-P time is driven almost
entirely by the vertical leg of the ray path. Converting that S-P time to a depth via the
accepted velocity model gives an INDEPENDENT depth estimate for each event, built only from
its two DEEJ pick times and its (velocity-independent, purely geometric) epicentral distance
to DEEJ -- not from hypoDD's own depth. If hypoDD's depth ordering is real, this independent
S-P-based depth should track it (ideally close to 1:1). If hypoDD's depth spread is an
artifact, the S-P-based depth should show little or no relationship to it.

Method:
1. Pull each event's DEEJ P/S catalog pick times from phase.dat; S-P is their difference.
2. Pull each event's hypoDD-relocated depth from the SVD rerun
   (t1_svd_resolvability_and_englacial_literature memory).
3. Pull each event's epicentral distance to DEEJ from hypoDD's own source-parameter file
   (cluster3_svd_hypoDD.src) -- this field is purely geodetic (lat/lon based), independent
   of any velocity assumption or of hypoDD's depth, so using it here is not circular.
4. Solve depth_sp = sqrt((S-P / (1/Vs - 1/Vp))^2 - dist^2) for each event, using the
   established VS_FIX layer velocities for the depth range this cluster occupies (accepted,
   not tested).
5. Compare depth_sp to hypoDD depth directly: correlation + agreement with the 1:1 line is
   evidence the depth separation is real; no relationship (or the wrong sign) is evidence
   it's a location-error artifact.

Usage:
    python full_catalog_pipeline/analyze_cluster3_sp_vpvs.py
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
import pandas as pd
from scipy import stats

import catalog_paths

IDS_FILE = catalog_paths.work_dir("T1") + "/cluster3_event_ids.txt"
PHASE_DAT = catalog_paths.phase_dat("T1")
DTCC_FILE = catalog_paths.dt_cc("T1")
SVD_RELOC = catalog_paths.work_dir("T1") + "/cluster3_svd_hypoDD.reloc"
SRC_FILE = catalog_paths.work_dir("T1") + "/cluster3_svd_hypoDD.src"
OUT_DIR = catalog_paths.work_dir("T1")
OUT_PNG = f"{OUT_DIR}/cluster3_sp_depth_realvsartifact_check.png"
OUT_CSV = f"{OUT_DIR}/cluster3_sp_depth_realvsartifact_data.csv"
STATION = "7U.DEEJ"
N_BOOT = 5000

# validated T1 velocity model (VS_FIX, hypodd_relocation_setup memory) -- ACCEPTED, not tested
LAYER_TOP_KM = [0.0, 0.1, 3.1, 3.8, 9.5, 14.0, 25.0, 52.0]
LAYER_VP = [2.50, 3.85, 5.1, 5.8, 6.10, 6.5, 7.5, 8.05]
LAYER_VS = [1.84, 2.22, 2.95, 3.35, 3.53, 3.76, 4.34, 4.65]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#8a8a86"
GRID = "#e4e3de"

# NB: this SVD rerun's hypoDD.reloc has 18 fields/row (no NCCP/NCCS/NCTP/NCTS/RCC/RCT --
# verified via `awk '{print NF}' ... | sort | uniq -c` -> all 216 rows have exactly 18),
# unlike the standard 24-field hypoDD.reloc format. depth (field 4) is unaffected either way.
RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag", "cid",
]

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def layer_vp_vs(depth_km):
    idx = 0
    for i, top in enumerate(LAYER_TOP_KM):
        if depth_km >= top:
            idx = i
    return LAYER_VP[idx], LAYER_VS[idx]


def load_sp_times(ids):
    events = {}
    cur_eid = None
    with open(PHASE_DAT) as f:
        for line in f:
            if line.startswith("#"):
                cur_eid = int(line.split()[-1])
                if cur_eid in ids:
                    events[cur_eid] = {}
            elif cur_eid in ids and cur_eid in events:
                parts = line.split()
                if len(parts) != 4:
                    continue
                sta, tt, wt, ph = parts
                events[cur_eid][(sta, ph)] = float(tt)

    rows = []
    for eid, picks in events.items():
        if (STATION, "P") in picks and (STATION, "S") in picks:
            rows.append({"id": eid, "sp_obs": picks[(STATION, "S")] - picks[(STATION, "P")]})
    return pd.DataFrame(rows)


def load_dist(ids):
    """Epicentral distance (km) from each event to DEEJ, from hypoDD's own source-parameter
    file. This field is purely geodetic (delaz2 on lat/lon) -- it does not depend on any
    velocity assumption or on hypoDD's depth, so using it to derive an independent depth
    from S-P is not circular (unlike using the file's take-off angle, which IS a function
    of hypoDD's assumed depth and would reintroduce the thing we're trying to test)."""
    rows = []
    with open(SRC_FILE) as f:
        for line in f:
            parts = line.split()
            if int(parts[0]) not in ids:
                continue
            if parts[3] != STATION:
                continue
            dist_km = float(parts[5])
            rows.append({"id": int(parts[0]), "dist_km": dist_km})
    return pd.DataFrame(rows).drop_duplicates("id")


def load_cc_pairs(ids):
    """Path 2 cross-check: for event pairs both linked to DEEJ in dt.cc with BOTH a P
    and an S entry, (S-P)_A - (S-P)_B = dt_cc,S(A,B) - dt_cc,P(A,B) directly."""
    pairs = {}
    cur = None
    with open(DTCC_FILE) as f:
        for line in f:
            if line.startswith("#"):
                parts = line.split()
                e1, e2 = int(parts[1]), int(parts[2])
                cur = (e1, e2) if (e1 in ids and e2 in ids) else None
                if cur is not None:
                    pairs.setdefault(cur, {})
            elif cur is not None:
                p = line.split()
                if len(p) < 4:
                    continue
                sta, dt, wt, ph = p[0], float(p[1]), float(p[2]), p[3]
                if sta in (STATION, STATION.split(".")[-1]):
                    pairs[cur][ph] = dt
    rows = []
    for (e1, e2), d in pairs.items():
        if "P" in d and "S" in d:
            rows.append({"id_a": e1, "id_b": e2, "dsp_cc": d["S"] - d["P"]})
    return pd.DataFrame(rows)


def main():
    ids = set(int(x) for x in open(IDS_FILE))
    sp = load_sp_times(ids)
    print(f"events with DEEJ P+S picks: {len(sp)} / {len(ids)}")

    reloc = pd.read_csv(SVD_RELOC, sep=r"\s+", header=None, names=RELOC_COLS)
    reloc = reloc[["id", "depth", "ez"]]

    dist = load_dist(ids)
    print(f"events with DEEJ epicentral distance: {len(dist)}")

    df = sp.merge(reloc, on="id").merge(dist, on="id")
    print(f"final merged n = {len(df)}")
    print(f"epicentral distance to DEEJ (km): median={df['dist_km'].median():.3f}, "
          f"max={df['dist_km'].max():.3f} (vs. hypoDD depth range "
          f"{df['depth'].min():.2f}-{df['depth'].max():.2f} km -- horizontal offset is "
          f"small relative to depth, so S-P is dominated by the vertical leg)")

    # Convert S-P to an independent depth estimate using the ACCEPTED velocity model
    # (Vp/Vs=1.73 is a given here, not what's being tested). Almost the whole cluster sits
    # in one layer, so use that layer's Vp/Vs throughout.
    depth_med = df["depth"].median()
    vp, vs = layer_vp_vs(depth_med)
    slope_vertical = 1 / vs - 1 / vp
    print(f"\nusing accepted layer velocity at median depth {depth_med:.3f} km: "
          f"Vp={vp}, Vs={vs} (Vp/Vs={vp/vs:.3f}); 1/Vs-1/Vp={slope_vertical:.4f} s/km")

    slant_range = df["sp_obs"] / slope_vertical
    valid = slant_range >= df["dist_km"]
    print(f"events where S-P implies a real (non-imaginary) depth solution: "
          f"{valid.sum()} / {len(df)}")
    df = df[valid].copy()
    df["depth_sp"] = np.sqrt(slant_range[valid] ** 2 - df["dist_km"] ** 2)

    # --- the actual test: does the independent S-P-based depth track hypoDD depth? ---
    r, p_val = stats.pearsonr(df["depth"], df["depth_sp"])
    rho, p_rho = stats.spearmanr(df["depth"], df["depth_sp"])
    print(f"\ndepth_sp vs. hypoDD depth: Pearson r={r:.3f} (p={p_val:.4f}), "
          f"Spearman rho={rho:.3f} (p={p_rho:.4f}), n={len(df)}")

    offset = (df["depth_sp"] - df["depth"])
    print(f"depth_sp - hypoDD depth: mean={offset.mean():.3f} km, std={offset.std():.3f} km "
          f"(a nonzero mean is a systematic bias -- e.g. near-DEEJ velocity structure not "
          f"captured by the layer model -- and is a separate issue from whether the RELATIVE "
          f"ordering/spread, i.e. the correlation above, is corroborated)")

    b, a = np.polyfit(df["depth"], df["depth_sp"], 1)
    print(f"OLS: depth_sp = {a:.4f} + {b:.4f} * hypoDD_depth  (1:1 slope would be 1.0 if the "
          f"depth spread were fully corroborated)")

    rng = np.random.default_rng(0)
    boot_r = np.empty(N_BOOT)
    idx_all = np.arange(len(df))
    depth_v, depth_sp_v = df["depth"].to_numpy(), df["depth_sp"].to_numpy()
    for i in range(N_BOOT):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        boot_r[i] = np.corrcoef(depth_v[idx], depth_sp_v[idx])[0, 1]
    r_lo, r_hi = np.percentile(boot_r, [2.5, 97.5])
    print(f"bootstrap 95% CI on Pearson r (n={N_BOOT} resamples): [{r_lo:.3f}, {r_hi:.3f}]")

    # Path 2 cross-check (see memory: dt.cc P-phase coverage is sparse) -- report but don't
    # over-interpret if n is too small.
    cc = load_cc_pairs(ids)
    print(f"\ndt.cc pairs with both P and S linked at DEEJ (cluster3-internal): {len(cc)}")
    if len(cc) >= 10:
        depth_by_id = df.set_index("id")["depth"]
        cc = cc[cc["id_a"].isin(depth_by_id.index) & cc["id_b"].isin(depth_by_id.index)]
        cc["ddepth"] = depth_by_id.loc[cc["id_a"]].to_numpy() - depth_by_id.loc[cc["id_b"]].to_numpy()
        r_cc, p_cc = stats.pearsonr(cc["ddepth"], cc["dsp_cc"])
        print(f"  CC-refined check: corr(d(depth), d(S-P))={r_cc:.3f} (p={p_cc:.4f}, n={len(cc)})")
    else:
        print(f"  -> too few pairs for a CC-refined check (plan's anticipated caveat): "
              f"{list(zip(cc['id_a'], cc['id_b'], cc['dsp_cc'].round(4)))}. "
              "not used quantitatively; the absolute-pick test above is primary.")

    stats_txt = (
        f"Pearson r={r:.3f} (p={p_val:.3g}), 95% CI [{r_lo:.3f}, {r_hi:.3f}]\n"
        f"Spearman rho={rho:.3f} (p={p_rho:.3g}), n={len(df)}\n"
        f"OLS slope vs. 1:1: {b:.3f}   mean offset: {offset.mean()*1000:+.0f} m"
    )
    plot(df, a, b, offset, stats_txt, OUT_PNG)
    print(f"\nwrote {OUT_PNG}")
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


def plot(df, a, b, offset, stats_txt, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    depth_m = df["depth"].values * 1000
    depth_sp_m = df["depth_sp"].values * 1000
    ax1.scatter(depth_m, depth_sp_m, s=22, color=BLUE, alpha=0.6, linewidth=0,
                label=f"events (n={len(df)})")
    lo, hi = min(depth_m.min(), depth_sp_m.min()), max(depth_m.max(), depth_sp_m.max())
    xs = np.linspace(lo, hi, 50)
    ax1.plot(xs, xs, color=MUTED, linewidth=1.5, linestyle=":", label="1:1 (fully corroborated)")
    ax1.plot(xs, (a + b * (xs / 1000)) * 1000, color=INK, linewidth=2,
              label=f"OLS fit: slope={b:.2f}")
    ax1.set_xlabel("hypoDD depth (m)")
    ax1.set_ylabel("Independent S-P-derived depth at DEEJ (m)\n(accepted Vp/Vs=1.73)")
    ax1.set_title("Cluster3 (n=216): is the hypoDD depth spread real,\n"
                   "or a location-error artifact?")
    ax1.legend(fontsize=9, loc="upper left", frameon=False)
    ax1.text(0.98, 0.02, stats_txt, transform=ax1.transAxes, fontsize=9,
              ha="right", va="bottom", color=INK,
              bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.4"))

    ax2.hist(offset * 1000, bins=30, color=MUTED, linewidth=0)
    ax2.axvline(0, color=INK, linewidth=1.5, linestyle="--")
    ax2.set_xlabel("depth_sp - hypoDD depth (m)")
    ax2.set_ylabel("Number of events")
    ax2.set_title(f"Offset distribution (mean={offset.mean()*1000:+.0f} m, "
                   f"std={offset.std()*1000:.0f} m)")

    fig.suptitle("Investigation 2: does DEEJ's S-P-derived depth (accepted Vp/Vs=1.73)\n"
                  "corroborate hypoDD's within-cluster depth separation?", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
