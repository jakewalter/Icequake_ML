#!/usr/bin/env python3
"""Investigation 1, step 2 (see t1_composite_focal_mech_and_vpvs_consistency_plan memory):
fit one composite double-couple focal mechanism to T1 cluster3's combined P first-motion
polarities, using each event x station's (azimuth, take-off angle) from hypoDD's own
partials.f source-parameter file (cluster3_svd_hypoDD.src) -- no new ray tracing needed.

Composite/stacked approach: individual events have too few clean polarities at only 5
stations to constrain a mechanism alone, so every event's polarities in the cluster are
pooled onto one shared focal sphere and a single strike/dip/rake is grid-searched to best
fit the combined set (standard technique for small/microseismic event groups assumed to
share one source mechanism).

Radiation-pattern formula (Aki & Richards convention; az = station azimuth from source,
ih = take-off angle from downward vertical, 0=down/180=up; phi = az - strike):
    R_P = cos(rake)*sin(dip)*sin(ih)^2*sin(2*phi)
          - cos(rake)*cos(dip)*sin(2*ih)*cos(phi)
          + sin(rake)*sin(2*dip)*(cos(ih)^2 - sin(ih)^2*sin(phi)^2)
          + sin(rake)*cos(2*dip)*sin(2*ih)*sin(phi)
predicted polarity = sign(R_P).

Usage:
    python full_catalog_pipeline/focal_mech_cluster3_fit.py
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

import catalog_paths

POLARITIES_CSV = catalog_paths.work_dir("T1") + "/cluster3_polarities.csv"
SRC_FILE = catalog_paths.work_dir("T1") + "/cluster3_svd_hypoDD.src"
OUT_DIR = catalog_paths.work_dir("T1")
OUT_CSV = f"{OUT_DIR}/cluster3_focal_mech_grid.csv"
OUT_PNG = f"{OUT_DIR}/cluster3_focal_mech_polarity_plot.png"

SRC_COLS = ["evid", "lat", "lon", "sta", "elv", "dist", "az", "ainp", "ains",
            "ttp", "tts", "xp", "yp", "zp", "xs", "ys", "zs"]

COARSE_STEP_DEG = 5
FINE_STEP_DEG = 1
FINE_HALF_RANGE_DEG = 8  # refine +/- this many degrees around the coarse best


def radiation_pattern_p(strike_deg, dip_deg, rake_deg, az_deg, ih_deg):
    strike, dip, rake = np.radians(strike_deg), np.radians(dip_deg), np.radians(rake_deg)
    phi = np.radians(az_deg - strike_deg)
    ih = np.radians(ih_deg)
    return (
        np.cos(rake) * np.sin(dip) * np.sin(ih) ** 2 * np.sin(2 * phi)
        - np.cos(rake) * np.cos(dip) * np.sin(2 * ih) * np.cos(phi)
        + np.sin(rake) * np.sin(2 * dip) * (np.cos(ih) ** 2 - np.sin(ih) ** 2 * np.sin(phi) ** 2)
        + np.sin(rake) * np.cos(2 * dip) * np.sin(2 * ih) * np.sin(phi)
    )


def load_data():
    pol = pd.read_csv(POLARITIES_CSV)
    pol = pol[pol["polarity"] != 0].copy()
    src = pd.read_csv(SRC_FILE, sep=r"\s+", header=None, names=SRC_COLS)
    src["sta"] = src["sta"].str.split(".").str[-1]
    src = src[["evid", "sta", "az", "ainp"]].rename(columns={"evid": "id"})
    df = pol.merge(src, left_on=["id", "station"], right_on=["id", "sta"], how="inner")
    return df.drop_duplicates(subset=["id", "station"])


def grid_search(df, strikes, dips, rakes):
    az = df["az"].to_numpy()
    ih = df["ainp"].to_numpy()
    obs = df["polarity"].to_numpy()
    n = len(df)

    best = None
    results = []
    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                pred = np.sign(radiation_pattern_p(strike, dip, rake, az, ih))
                n_mismatch = int(np.sum(pred != obs))
                results.append((strike, dip, rake, n_mismatch))
                if best is None or n_mismatch < best[3]:
                    best = (strike, dip, rake, n_mismatch)
    return best, pd.DataFrame(results, columns=["strike", "dip", "rake", "n_mismatch"]), n


def main():
    df = load_data()
    print(f"polarity observations usable for the composite mechanism: {len(df)}")
    print(df.groupby("station")["polarity"].agg(
        n=len, up=lambda s: (s == 1).sum(), down=lambda s: (s == -1).sum()))

    strikes = np.arange(0, 360, COARSE_STEP_DEG)
    dips = np.arange(5, 91, COARSE_STEP_DEG)
    rakes = np.arange(-180, 180, COARSE_STEP_DEG)
    print(f"\ncoarse grid: {len(strikes)} strikes x {len(dips)} dips x {len(rakes)} rakes "
          f"= {len(strikes)*len(dips)*len(rakes)} mechanisms")
    best, grid_df, n = grid_search(df, strikes, dips, rakes)
    print(f"coarse best: strike={best[0]}, dip={best[1]}, rake={best[2]}, "
          f"mismatches={best[3]}/{n} ({100*best[3]/n:.1f}%)")

    fs = np.arange(best[0] - FINE_HALF_RANGE_DEG, best[0] + FINE_HALF_RANGE_DEG + 1, FINE_STEP_DEG) % 360
    fd = np.clip(np.arange(best[1] - FINE_HALF_RANGE_DEG, best[1] + FINE_HALF_RANGE_DEG + 1, FINE_STEP_DEG), 0, 90)
    fr = np.arange(best[2] - FINE_HALF_RANGE_DEG, best[2] + FINE_HALF_RANGE_DEG + 1, FINE_STEP_DEG)
    fr = ((fr + 180) % 360) - 180
    best_fine, grid_fine_df, _ = grid_search(df, np.unique(fs), np.unique(fd), np.unique(fr))
    print(f"fine-refined best: strike={best_fine[0]:.0f}, dip={best_fine[1]:.0f}, "
          f"rake={best_fine[2]:.0f}, mismatches={best_fine[3]}/{n} ({100*best_fine[3]/n:.1f}%)")

    # acceptable-mechanism set (HASH-style uncertainty region): all coarse-grid mechanisms
    # within 1 extra misfit of the best, characterizing how tightly strike/dip/rake are
    # constrained by this polarity set rather than reporting a single point estimate.
    acceptable = grid_df[grid_df["n_mismatch"] <= best[3] + 1]
    print(f"\nmechanisms within +1 misfit of best ({best[3]+1}/{n}): {len(acceptable)} "
          f"of {len(grid_df)} grid points searched")
    print("acceptable strike range:", acceptable["strike"].min(), "-", acceptable["strike"].max())
    print("acceptable dip range:", acceptable["dip"].min(), "-", acceptable["dip"].max())
    print("acceptable rake range:", acceptable["rake"].min(), "-", acceptable["rake"].max())

    grid_df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    plot_polarities(df, best_fine, OUT_PNG)
    print(f"wrote {OUT_PNG}")

    return df, best_fine, acceptable


def plot_polarities(df, best, out_path):
    strike, dip, rake, n_mismatch = best
    az = df["az"].to_numpy()
    ih = df["ainp"].to_numpy()
    obs = df["polarity"].to_numpy()
    pred = np.sign(radiation_pattern_p(strike, dip, rake, az, ih))

    # equal-area (Schmidt) lower-hemisphere projection of the takeoff direction.
    # ih is measured 0=down,180=up; fold up-going rays (ih>90) to their lower-hemisphere
    # antipode (az+180, 180-ih) so every observation plots on one net, per standard
    # first-motion practice.
    ih_plot = ih.copy()
    az_plot = az.copy()
    up = ih_plot > 90
    ih_plot[up] = 180 - ih_plot[up]
    az_plot[up] = (az_plot[up] + 180) % 360

    r = np.sqrt(2) * np.sin(np.radians(ih_plot) / 2)
    theta = np.radians(90 - az_plot)  # azimuth clockwise from N -> standard map angle

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, np.sqrt(2))
    ax.set_yticklabels([])

    correct = pred == obs
    up_pol = obs == 1
    ax.scatter(np.radians(az_plot)[correct & up_pol], r[correct & up_pol],
               marker="o", facecolor="black", edgecolor="black", s=70, label="up, fit")
    ax.scatter(np.radians(az_plot)[correct & ~up_pol], r[correct & ~up_pol],
               marker="o", facecolor="white", edgecolor="black", s=70, label="down, fit")
    ax.scatter(np.radians(az_plot)[~correct & up_pol], r[~correct & up_pol],
               marker="^", facecolor="red", edgecolor="red", s=70, label="up, misfit")
    ax.scatter(np.radians(az_plot)[~correct & ~up_pol], r[~correct & ~up_pol],
               marker="v", facecolor="white", edgecolor="red", s=70, label="down, misfit")

    # nodal-plane trace: contour where R_P == 0 over a dense grid
    az_grid = np.linspace(0, 360, 361)
    ih_grid = np.linspace(0, 90, 181)
    AZ, IH = np.meshgrid(az_grid, ih_grid)
    RP = radiation_pattern_p(strike, dip, rake, AZ, IH)
    R_GRID = np.sqrt(2) * np.sin(np.radians(IH) / 2)
    THETA_GRID = np.radians(AZ)
    ax.contour(THETA_GRID, R_GRID, RP, levels=[0], colors="steelblue", linewidths=1.5)

    ax.set_title(f"Cluster3 composite focal mechanism (n={len(df)} obs)\n"
                 f"strike={strike:.0f}, dip={dip:.0f}, rake={rake:.0f}, "
                 f"misfits={n_mismatch}/{len(df)} ({100*n_mismatch/len(df):.1f}%)\n"
                 f"lower-hemisphere, equal-area", fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
