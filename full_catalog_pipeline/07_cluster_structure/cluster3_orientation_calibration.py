#!/usr/bin/env python3
"""Single-station horizontal-orientation calibration via P-wave particle motion.

This deployment's StationXML has no azimuth/dip for HH1/HH2 (verified: neither
`hypodd_relocate.py`'s `build_station_xml` nor `artifacts/hypodd_station_7U.xml`
carries orientation fields) -- so `deej_waveform_common`/`windowing.py`'s "N"="HH2",
"E"="HH1" labeling is a naming convention only, NOT a verified geographic
orientation. Any radial/transverse rotation (needed to test the firn-reverberation
vs. guided/converted-wave hypothesis from [[t1-sp-vs-depth-vpvs-check-result]] via
particle motion) requires first solving for each station's actual sensor-frame
rotation relative to true north.

Method: direct P-wave motion is, to good approximation, linearly polarized along
the source-to-station propagation direction (the "az" field in hypoDD's .src file,
independent of the orientation question). For each event, take a short window
starting at the P pick, compute the dominant polarization azimuth of the RAW
(sensor-frame) HH1/HH2 traces via 2D PCA (eigenvector of the largest eigenvalue),
and compare it to the known source azimuth -- both are only defined mod 180 deg
(a linear polarization axis has no direction sense), so the comparison is done on
doubled angles (standard circular-statistics trick for axial/undirected data).
The circular mean of (2*measured_phi - 2*known_az) over many high-rectilinearity
events gives 2x the station's constant sensor-frame rotation offset.

Usage:
    python full_catalog_pipeline/cluster3_orientation_calibration.py
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

import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import DEFAULTS, clean, load_merged
from lib.windowing import RollingDayCache, extract_window

import catalog_paths

STATIONS = ["DEEJ", "LILA", "TJTJ", "ELZA"]
WINDOW_PRE = 0.2
P_WINDOW_POST = 0.35  # raw P-window length used for the polarization PCA
RECTILINEARITY_MIN = 0.7  # QC gate: keep only clearly-linear P arrivals


def _origin_and_az_for_station(station):
    """Per-event origin time (from phase.dat) and source-to-station azimuth
    (from hypoDD.src, field index 6 -- see focal_mech_cluster3_fit.py's verified
    column layout: evid lat lon sta elv dist az ainp ains ...)."""
    ids = set(int(x) for x in open(DEFAULTS["ids_file"]))
    df = load_merged(ids, DEFAULTS["phase_dat"], DEFAULTS["reloc"], station, network="7U")

    origins = {}
    with open(DEFAULTS["phase_dat"]) as f:
        cur = None
        for line in f:
            if line.startswith("#"):
                p = line.split()
                cur = int(p[-1])
                yr, mo, dy, hr, mi, sc = p[1:7]
                from obspy import UTCDateTime
                origins[cur] = UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mi), float(sc))

    az = {}
    sta_tag = f"7U.{station}"
    with open(DEFAULTS["src"]) as f:
        for line in f:
            parts = line.split()
            if parts[3] != sta_tag:
                continue
            az[int(parts[0])] = float(parts[6])

    df = df[df["id"].isin(az)].copy()
    df["origin"] = df["id"].map(origins)
    df["az"] = df["id"].map(az)
    return df


def polarization_azimuth(n, e):
    """Dominant 2D polarization azimuth (deg, 0=N/+n axis, 90=E/+e axis, mod 180)
    and rectilinearity (1 - lambda2/lambda1) from the [n, e] covariance matrix."""
    n_c, e_c = n - n.mean(), e - e.mean()
    cov = np.cov(np.vstack([n_c, e_c]))
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    lam2, lam1 = eigvals[0], eigvals[1]
    principal = eigvecs[:, 1]  # [n_comp, e_comp] of the dominant eigenvector
    phi = np.degrees(np.arctan2(principal[1], principal[0])) % 180.0
    rect = 1.0 - (lam2 / lam1 if lam1 > 0 else 1.0)
    return phi, rect


def calibrate_station(station):
    df = _origin_and_az_for_station(station)
    print(f"\n=== {station} ===  {len(df)} events with P/S picks + az")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)

    rows = []
    for _, ev in df.sort_values("depth").iterrows():
        eid = int(ev["id"])
        p_pick = ev["origin"] + ev["p_offset"]
        row = {"primary_date": date(p_pick.year, p_pick.month, p_pick.day).isoformat()}
        window_start = p_pick - WINDOW_PRE
        window_end = p_pick + P_WINDOW_POST
        data, status = extract_window(cache, row, window_start, window_end)
        cache.evict_before(row["primary_date"])
        if data is None or not data["complete"]:
            continue

        n_clean, e_clean = clean(data["N"]), clean(data["E"])
        sr = config.SAMPLE_RATE_HZ
        t = np.arange(len(n_clean)) / sr - WINDOW_PRE
        # P-polarization window: pick to pick+0.15s (impulsive onset, before S arrives)
        mask = (t >= 0.0) & (t <= 0.15)
        if mask.sum() < 10:
            continue
        phi, rect = polarization_azimuth(n_clean[mask], e_clean[mask])
        rows.append({"id": eid, "az": ev["az"], "phi_raw": phi, "rect": rect})

    out = pd.DataFrame(rows)
    print(f"  usable P windows: {len(out)} / {len(df)}")
    if len(out) == 0:
        return station, None, 0, 0.0

    hq = out[out["rect"] >= RECTILINEARITY_MIN]
    print(f"  high-rectilinearity (>={RECTILINEARITY_MIN}) events: {len(hq)}")
    if len(hq) < 5:
        print("  too few high-quality events, skipping offset solve")
        return station, out, len(hq), 0.0

    resid_doubled = np.radians(2.0 * (hq["phi_raw"].values - hq["az"].values))
    mean_vec = np.mean(np.exp(1j * resid_doubled))
    offset = (np.degrees(np.angle(mean_vec)) / 2.0) % 180.0
    concentration = np.abs(mean_vec)  # 0=no consistent offset, 1=perfectly consistent
    print(f"  solved orientation offset (phi_raw - az, mod 180): {offset:.1f} deg, "
          f"circular concentration R={concentration:.2f}")
    return station, out, len(hq), offset, concentration


def main():
    results = {}
    for station in STATIONS:
        station, out, n_hq, *rest = calibrate_station(station)
        offset = rest[0] if rest else 0.0
        conc = rest[1] if len(rest) > 1 else 0.0
        results[station] = {"n_hq": n_hq, "offset_deg": offset, "concentration": conc}
        if out is not None:
            out_csv = (f"{catalog_paths.work_dir('T1')}/"
                       f"cluster3_orientation_{station.lower()}_raw.csv")
            out.to_csv(out_csv, index=False)

    print("\n=== Summary ===")
    summary = pd.DataFrame(results).T
    print(summary)
    summary_csv = catalog_paths.work_dir("T1") + "/cluster3_orientation_offsets.csv"
    summary.to_csv(summary_csv)
    print(f"\nwrote {summary_csv}")


if __name__ == "__main__":
    main()
