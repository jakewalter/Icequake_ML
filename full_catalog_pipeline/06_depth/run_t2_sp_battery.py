#!/usr/bin/env python3
"""Run the full S-P depth-reality battery over every major cluster of a T2 relocation.

Redoes, end to end, what [[t2-basal-event-nature-multi-analysis]] did by hand for two
clusters: waveform gathers, from-scratch single-component CC refinement (Z and N), and the
depth correlation -- but for every cluster large enough to test, against whichever relocation
is selected by ICEQUAKE_RELOC.

Two things this adds over invoking cc_refine_component.py by hand:

  * It correlates the CC-refined S-P against the cluster's OWN PCA long axis as well as
    against raw depth. That distinction previously produced a wrong verdict: a cluster
    plunging ~19 degrees from horizontal shows opposite-signed depth correlations at stations
    on opposite ends of its strike, which reads as "incoherent, therefore artifact" until you
    project onto the right axis. Clusters here range from 5 to 87 degrees of plunge, so the
    raw-depth test is the correct one for some and the wrong one for others.

  * It picks the comparison stations from geometry rather than by distance alone -- for a
    near-horizontal cluster the diagnostic pair sits at the two ENDS of the plunge axis.

Usage:
    ICEQUAKE_RELOC=hypodd_vels1d_ccstrong python full_catalog_pipeline/run_t2_sp_battery.py
    ... --clusters 0 1 2 --stations-per-cluster 3 --skip-gathers
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
import subprocess
import sys

import numpy as np
import pandas as pd
import pyproj
from obspy.geodetics import gps2dist_azimuth

import catalog_paths

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]

# vels1d reflection model, averaged over the 1.2-2.0 km interval the clusters occupy:
# Vp 3.850 km/s, Vp/Vs 2.008 -> Vs 1.917.
VP, VS = 3.85, 1.917
NETWORK = "7U"
MIN_CLUSTER = 35
MIN_PICKS = 40   # a station needs real coverage of the cluster to be worth correlating


def pick_counts(phase_dat, ids):
    """Phase picks per station for this event set.

    Station choice must be filtered by this, not by geometry alone. Cluster 1's two
    best-aligned stations by azimuth (BAUM, OKGS) turned out to have 0 and 1 picks for its
    events, so a purely geometric choice spent the run on empty selections and skipped DRSC
    and JULA, which have 150 each.
    """
    want, cur, cnt = set(ids), None, {}
    for line in open(phase_dat):
        if line.startswith("#"):
            cur = int(line.split()[-1])
            continue
        if cur in want:
            code = line.split()[0].split(".")[-1]
            cnt[code] = cnt.get(code, 0) + 1
    return cnt


def geometry(rel, ids, sta):
    d = rel.loc[[i for i in ids if i in rel.index]]
    to = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = to.transform(d["lon"].values, d["lat"].values)
    z = d["depth"].values * 1000.0
    pts = np.c_[x - x.mean(), y - y.mean(), z - z.mean()]
    w, v = np.linalg.eigh(np.cov(pts.T))
    axis = v[:, -1]
    plunge = float(np.degrees(np.arcsin(abs(axis[2]) / np.linalg.norm(axis))))
    axis_az = float(np.degrees(np.arctan2(axis[0], axis[1])) % 180)
    along = pd.Series(pts @ axis, index=d.index)

    clat, clon = d["lat"].mean(), d["lon"].mean()
    rows = []
    for _, s in sta.iterrows():
        dist, az, _ = gps2dist_azimuth(clat, clon, s["lat"], s["lon"])
        # for a near-horizontal cluster the informative stations lie along the plunge axis;
        # for a near-vertical one every azimuth is equivalent, so fall back to distance
        off = min(abs((az - axis_az) % 360), abs((az - axis_az - 180) % 360))
        off = min(off, 360 - off)
        rows.append({"code": s["code"], "dist_km": dist / 1000.0, "az": az, "axis_off": off})
    st = pd.DataFrame(rows).sort_values("dist_km")
    if plunge < 45:  # elongated near-horizontally: prefer stations near the axis ends
        st = st.sort_values(["axis_off", "dist_km"])
    return d, along, plunge, axis_az, st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clusters", nargs="*", type=int, default=None)
    ap.add_argument("--stations-per-cluster", type=int, default=3)
    ap.add_argument("--components", nargs="*", default=["Z", "N"])
    ap.add_argument("--skip-gathers", action="store_true")
    ap.add_argument("--min-cc", type=float, default=0.5)
    args = ap.parse_args()

    work = catalog_paths.work_dir("T2")
    reloc = catalog_paths.reloc("T2")
    print(f"relocation: {reloc}")
    rel = pd.read_csv(reloc, sep=r"\s+", header=None, names=RELOC_COLS).set_index("id")
    sta = pd.read_csv(catalog_paths.station_sel("T2"), sep=r"\s+", header=None,
                      names=["id", "lat", "lon", "elev"])
    sta["code"] = sta["id"].str.split(".").str[1]

    clusters = args.clusters
    if clusters is None:
        clusters = []
        for f in os.listdir(work):
            if f.startswith("t2_cluster") and f.endswith("_event_ids.txt"):
                n = sum(1 for _ in open(os.path.join(work, f)))
                if n >= MIN_CLUSTER:
                    clusters.append(int(f[len("t2_cluster"):-len("_event_ids.txt")]))
        clusters.sort(key=lambda c: -sum(
            1 for _ in open(os.path.join(work, f"t2_cluster{c}_event_ids.txt"))))

    out = []
    for c in clusters:
        ids_file = os.path.join(work, f"t2_cluster{c}_event_ids.txt")
        ids = [int(l) for l in open(ids_file)]
        d, along, plunge, axis_az, st = geometry(rel, ids, sta)
        cnt = pick_counts(catalog_paths.phase_dat("T2"), ids)
        st["n_picks"] = st["code"].map(lambda c: cnt.get(c, 0))
        dropped = st[st["n_picks"] < MIN_PICKS]["code"].tolist()
        st = st[st["n_picks"] >= MIN_PICKS]
        if dropped:
            print(f"    (no usable picks, skipped: {', '.join(dropped)})")
        if st.empty:
            print("    no station has enough picks for this cluster")
            continue
        spread = (d["depth"].max() - d["depth"].min()) * 1000.0
        print(f"\n=== cluster {c}: n={len(d)}, depth spread {spread:.0f} m, "
              f"plunge {plunge:.1f} deg, axis az {axis_az:.0f} ===")
        print("    " + ", ".join(
            f"{r.code}@{r.dist_km:.2f}km/az{r.az:.0f}/{r.n_picks}picks"
            for r in st.head(args.stations_per_cluster).itertuples()))

        for r in st.head(args.stations_per_cluster).itertuples():
            for comp in args.components:
                tag = f"t2sp_c{c}_{r.code}_{comp}"
                cmd = [sys.executable, "full_catalog_pipeline/cc_refine_component.py",
                       "--ids-file", ids_file,
                       "--phase-dat", catalog_paths.phase_dat("T2"),
                       "--dtcc", catalog_paths.dt_cc("T2"),
                       "--reloc", reloc,
                       "--station-sel", catalog_paths.station_sel("T2"),
                       "--out-dir", work, "--station", r.code, "--network", NETWORK,
                       "--component", comp, "--vp", str(VP), "--vs", str(VS),
                       "--min-cc", str(args.min_cc), "--tag", tag]
                p = subprocess.run(cmd, capture_output=True, text=True)
                csv = os.path.join(work, f"{tag}_cc_refine_{comp}.csv")
                if p.returncode != 0 or not os.path.exists(csv):
                    tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["(no output)"]
                    print(f"    {r.code}/{comp}: FAILED - {tail[0][:110]}")
                    continue

                t = pd.read_csv(csv).set_index("id")
                t = t[np.isfinite(t["sp_refined"])]
                if len(t) < 20:
                    print(f"    {r.code}/{comp}: only {len(t)} events, skipped")
                    continue
                a = along.reindex(t.index)
                good = t["cc"] >= args.min_cc
                rec = {
                    "cluster": c, "n_events": len(d), "station": r.code,
                    "dist_km": round(r.dist_km, 2), "az": round(r.az), "component": comp,
                    "plunge_deg": round(plunge, 1), "depth_spread_m": round(spread),
                    "n_scored": len(t), "n_good_cc": int(good.sum()),
                    "mean_cc": round(float(t["cc"].mean()), 3),
                }
                for lbl, sub in (("all", t.index), ("good", t.index[good])):
                    if len(sub) < 20:
                        continue
                    tt, aa = t.loc[sub], a.loc[sub]
                    rec[f"r_depth_{lbl}"] = round(float(
                        np.corrcoef(tt["depth"], tt["sp_refined"])[0, 1]), 3)
                    rec[f"r_axis_{lbl}"] = round(float(
                        np.corrcoef(aa, tt["sp_refined"])[0, 1]), 3)
                    slope = np.polyfit(tt["depth"], tt["sp_refined"] * 1000.0, 1)[0]
                    rec[f"slope_obs_{lbl}"] = round(float(slope))
                pred = (1.0 / VS - 1.0 / VP) * 1000.0 * (
                    d["depth"].median() / np.hypot(d["depth"].median(), r.dist_km))
                rec["slope_pred_msperkm"] = round(float(pred))
                out.append(rec)
                print(f"    {r.code}/{comp}: n={len(t)} meanCC={rec['mean_cc']} "
                      f"r_depth={rec.get('r_depth_all')} r_axis={rec.get('r_axis_all')} "
                      f"slope={rec.get('slope_obs_all')} vs pred {rec['slope_pred_msperkm']} ms/km")

        if not args.skip_gathers:
            for r in st.head(2).itertuples():
                g = subprocess.run(
                    [sys.executable, "full_catalog_pipeline/plot_cluster_sp_cascade.py",
                     "--ids-file", ids_file,
                     "--phase-dat", catalog_paths.phase_dat("T2"),
                     "--dtcc", catalog_paths.dt_cc("T2"), "--reloc", reloc,
                     "--station-sel", catalog_paths.station_sel("T2"),
                     "--out-dir", work, "--station", r.code, "--network", NETWORK,
                     "--vp", str(VP), "--vs", str(VS), "--chunk-size", "5000",
                     "--tag", f"t2sp_c{c}_{r.code}"],
                    capture_output=True, text=True)
                print(f"    gather {r.code}: {'ok' if g.returncode == 0 else 'FAILED'}")

    if out:
        df = pd.DataFrame(out)
        p = os.path.join(work, "t2_sp_battery_summary.csv")
        df.to_csv(p, index=False)
        print(f"\nwrote {p}")
        pd.set_option("display.width", 200, "display.max_columns", 30)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
