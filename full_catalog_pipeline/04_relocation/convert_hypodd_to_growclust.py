#!/usr/bin/env python3
"""Convert an existing hypoDD input set (event.sel/event.dat, station.sel,
dt.cc) into GrowClust's input file formats, so GrowClust can be run as an
independent cross-check against hypoDD's double-difference relocation on the
same underlying cross-correlation data.

Why: GrowClust (Trugman & Shearer 2017, hierarchical clustering + bootstrap)
uses a fundamentally different relocation algorithm than hypoDD's global LSQR
solve -- it never assembles one giant linear system for a whole cluster, so it
isn't vulnerable to the LSQR instability that forced hypoDD's T1 QuakeMigrate
run to fragment into many small MINOBS_CT-limited clusters (see
hypodd_relocation_setup / t1_optimized_hypodd_and_catalog_comparison_plan
memory). A second, independent algorithm agreeing (or disagreeing) with
hypoDD's basal-crack findings is a useful cross-check.

Important limitation, by design of GrowClust's Fortran90 version: it uses ONLY
cross-correlation (dt.cc) differential times, not catalog-pick (dt.ct) ones --
unlike hypoDD, which combines both. For the QuakeMigrate catalog, dt.cc is only
~1.8% of the total differential-time volume (dt.ct dominates), so GrowClust
will relocate fewer events than hypoDD did -- expected, not a bug.

Format specs below were confirmed directly from GrowClust's Fortran source
(SRC/input_subs.f90 READ_EVFILE/READ_STLIST/READ_XCORDATA), not guessed from
the bundled EXAMPLE alone:
- evlist.txt (evlist_fmt=1, "phase" format): free-format
  YR MON DY HR MN SEC LAT LON DEP MAG EH EZ RMS ID per line.
- stlist.txt (stlist_fmt=1, "station name" format): STA LAT LON per line --
  GrowClust matches names literally, so the network-code prefix hypoDD's
  station.sel carries (e.g. "2E.TJTJ") must be stripped to match "TJTJ".
- xcordata.txt (xcordat_fmt=1, "dt.cc format"): READ_XCORDATA's irxform==1
  branch is explicitly commented "dt.cc format" and reads station names from
  a fixed first-10-columns field -- so hypoDD's dt.cc is reusable almost
  as-is, EXCEPT the same network-code-prefix stripping is required (the
  fixed-width station-name reader would otherwise truncate "2E.TJTJ" to
  "2E.TJ", matching no real station).
- vzmodel.txt: (depth, Vp, Vs) triples read by vel_subs.f90's VZFILLIN; Vs=0.0
  falls back to a single global vpvs_factor, but GrowClust accepts explicit
  per-layer Vs directly, so this converter writes the actual validated T1
  Vp/Vs profile per layer (avoiding the single-ratio approximation hypoDDpy's
  Python API forced for the hypoDD.inp velocity model).

Usage:
    python full_catalog_pipeline/convert_hypodd_to_growclust.py --catalog pyocto
    python full_catalog_pipeline/convert_hypodd_to_growclust.py --catalog qm
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

import numpy as np
import pandas as pd
import pyproj

import catalog_paths

SOURCES = {
    "pyocto": {
        "event_sel": catalog_paths.event_sel("T1"),
        "station_sel": catalog_paths.station_sel("T1"),
        "dt_cc": catalog_paths.dt_cc("T1"),
        "out_dir": "full_catalog_pipeline/artifacts/full_run/T1_v5/growclust_pyocto",
        "radius_km": None,  # already confined to ~8.5 km of centroid, no restriction needed
    },
    "qm": {
        "event_sel": "/scratch2/qm/t1/output/hypodd_optimized/input_files/event.sel",
        "station_sel": "/scratch2/qm/t1/output/hypodd_optimized/input_files/station.sel",
        "dt_cc": "/scratch2/qm/t1/output/hypodd_optimized/input_files/dt.cc",
        "out_dir": "full_catalog_pipeline/artifacts/full_run/T1_v5/growclust_qm",
        # same restriction used in plot_hypodd_t1_basal_3d.py / plot_hypodd_t1_basal_clusters.py --
        # QM's original catalog spans the whole ~70 km region, not confined to a small
        # association box like pyocto; far-field events also blow past GrowClust's
        # travel-time table distance range (tt_del1) otherwise.
        "radius_km": 15.0,
    },
}

# GrowClust's grow_params.f90 hardcodes tdifmax=30s as a sanity cutoff and stops the whole
# run on the first violation. A handful of cross-correlation pairs (measured: 1/114370 for
# pyocto, 1/11773 for QM exceed 30s; ~6 exceed 10s in each) have non-physical differential
# times for local microseismicity (expected differential times are ms-scale) -- clearly bad
# CC measurements, not real signal. Filtered well below GrowClust's own cutoff so a future
# outlier doesn't silently reintroduce this crash.
MAX_ABS_DT_SEC = 10.0

# Same validated T1 layered model used in the finalized hypoDD.inp (ccscale_0.33
# config, see t1_optimized_hypodd_and_catalog_comparison_plan / hypodd_relocation_setup
# memory) -- layer TOP depths, km.
VEL_TOP_KM = [0.0, 0.1, 3.1, 3.8, 9.5, 14.0, 25.0, 52.0]
VEL_VP = [2.50, 3.85, 5.1, 5.8, 6.10, 6.5, 7.5, 8.05]
VEL_VS = [1.84, 2.22, 2.95, 3.35, 3.53, 3.76, 4.34, 4.65]
VEL_BOTTOM_KM = 16.0  # must stay a bit beyond tt_dep1 (15 km -- relocation search can probe
                      # slightly past the observed depth range): vel_subs.f90 interpolates
                      # the ENTIRE input vzmodel.txt at tt_ddep spacing regardless of
                      # tt_dep1, then a separate downstream reader caps at 1000 fine-grid
                      # points -- a deep "unconstrained" tail (e.g. 100 km) at fine spacing
                      # silently overflows that limit


def strip_network(sta):
    return sta.split(".")[-1] if "." in sta else sta


def convert_evlist(event_sel_path, station_sel_path, out_path, radius_km=None):
    cols = ["date", "time", "lat", "lon", "depth", "mag", "eh", "ez", "rms", "id"]
    ev = pd.read_csv(event_sel_path, sep=r"\s+", header=None, names=cols)
    n_total = len(ev)

    if radius_km is not None:
        st = pd.read_csv(station_sel_path, sep=r"\s+", header=None,
                          names=["id", "lat", "lon", "elev"])
        to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        sx, sy = to_ps.transform(st["lon"].values, st["lat"].values)
        cx, cy = sx.mean(), sy.mean()
        ex, ey = to_ps.transform(ev["lon"].values, ev["lat"].values)
        r_km = np.hypot(ex - cx, ey - cy) / 1000.0
        ev = ev[r_km <= radius_km].reset_index(drop=True)

    date_s = ev["date"].astype(str)
    time_s = ev["time"].astype(str).str.zfill(8)  # HHMMSSss, zero-padded
    yr = date_s.str[0:4].astype(int)
    mon = date_s.str[4:6].astype(int)
    dy = date_s.str[6:8].astype(int)
    hr = time_s.str[0:2].astype(int)
    mn = time_s.str[2:4].astype(int)
    sec = time_s.str[4:6].astype(int) + time_s.str[6:8].astype(int) / 100.0
    with open(out_path, "w") as f:
        for i in range(len(ev)):
            f.write(f"{yr[i]} {mon[i]:02d} {dy[i]:02d} {hr[i]:02d} {mn[i]:02d} "
                    f"{sec[i]:6.3f} {ev['lat'][i]:.5f} {ev['lon'][i]:.5f} "
                    f"{ev['depth'][i]:.3f} {ev['mag'][i]:.2f} {ev['eh'][i]:.3f} "
                    f"{ev['ez'][i]:.3f} {ev['rms'][i]:.3f} {ev['id'][i]}\n")
    if radius_km is not None:
        print(f"  radius restriction ({radius_km} km): kept {len(ev)}/{n_total} events")
    return set(ev["id"].tolist())


def convert_stlist(station_sel_path, out_path):
    st = pd.read_csv(station_sel_path, sep=r"\s+", header=None,
                      names=["id", "lat", "lon", "elev"])
    st["code"] = st["id"].apply(strip_network)
    with open(out_path, "w") as f:
        for _, row in st.iterrows():
            f.write(f"{row['code']:<6s} {row['lat']:.5f} {row['lon']:.5f}\n")
    return len(st)


def convert_xcordata(dt_cc_path, out_path, keep_ids=None):
    n_pairs = 0
    n_obs = 0
    n_pairs_dropped_radius = 0
    n_obs_dropped_dt = 0
    write_pair = True  # whether the current pair's observations should be written
    with open(dt_cc_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split()
                id1, id2 = int(parts[0]), int(parts[1])
                if keep_ids is not None and (id1 not in keep_ids or id2 not in keep_ids):
                    write_pair = False
                    n_pairs_dropped_radius += 1
                    continue
                write_pair = True
                fout.write(line + "\n")
                n_pairs += 1
            else:
                if not write_pair:
                    continue
                parts = line.split()
                sta, dt, xcor, phase = parts[0], parts[1], parts[2], parts[3]
                if abs(float(dt)) > MAX_ABS_DT_SEC:
                    n_obs_dropped_dt += 1
                    continue
                sta = strip_network(sta)
                # GrowClust's Fortran reader takes a fixed 10-char station-name field,
                # then scans forward for the first blank -- a name packed flush to
                # column 10 (e.g. a 4-letter code right-justified in exactly 10 cols)
                # leaves no in-bounds blank to stop on and reads past the buffer
                # (undefined behavior, silently corrupts the parsed name). Left-justify
                # in 9 columns + an explicit space guarantees a blank within bounds.
                fout.write(f"{sta:<9s} {dt} {xcor} {phase}\n")
                n_obs += 1
    if keep_ids is not None and n_pairs_dropped_radius:
        print(f"  dropped {n_pairs_dropped_radius} pairs referencing a radius-excluded event")
    if n_obs_dropped_dt:
        print(f"  dropped {n_obs_dropped_dt} observations with |dt| > {MAX_ABS_DT_SEC}s (bad CC outliers)")
    return n_pairs, n_obs


def write_vzmodel(out_path):
    rows = []
    n = len(VEL_TOP_KM)
    for i in range(n):
        top = VEL_TOP_KM[i]
        if top >= VEL_BOTTOM_KM:
            break
        bottom = min(VEL_TOP_KM[i + 1], VEL_BOTTOM_KM) if i + 1 < n else VEL_BOTTOM_KM
        rows.append((top, VEL_VP[i], VEL_VS[i]))
        rows.append((bottom, VEL_VP[i], VEL_VS[i]))
        if bottom >= VEL_BOTTOM_KM:
            break
    with open(out_path, "w") as f:
        for depth, vp, vs in rows:
            f.write(f"{depth:7.3f} {vp:6.3f} {vs:6.3f}\n")


def write_control_file(out_dir, catalog_label, n_events):
    # Distance/depth table sized generously for T1's array + basal-event depth
    # range (observed max ~9-10 km; array footprint ~15-20 km after the QM
    # far-field restriction used elsewhere in this pipeline).
    ctl = f"""****     GrowClust Control File       *****
******      T1, {catalog_label} catalog        *******
*******************************************
*
*******************************************
*************  Event list  ****************
*******************************************
* evlist_fmt (0 = evlist, 1 = phase, 2 = GrowClust, 3 = HypoInverse)
1
* fin_evlist (event list file name)
IN/evlist.txt
*
*******************************************
************   Station list   *************
*******************************************
* stlist_fmt (0 = SEED channel, 1 = station name)
1
* fin_stlist (station list file name)
IN/stlist.txt
*
*******************************************
*************   XCOR data   ***************
*******************************************
* xcordat_fmt (0 = binary, 1 = text), tdif_fmt (21 = tt2-tt1, 12 = tt1-tt2)
1 12
* fin_xcordat
IN/xcordata.txt
*
*******************************************
*** Velocity Model / Travel Time Tables ***
*******************************************
* fin_vzmdl (input vz model file)
IN/vzmodel.txt
* fout_vzfine (output, interpolated vz model file)
TT/vzfine.txt
* fout_pTT (output travel time table, P phase)
TT/tt.pg
* fout_sTT (output travel time table, S phase)
TT/tt.sg
*
******************************************
***** Travel Time Table Parameters  ******
******************************************
* vpvs_factor  rayparam_min (-1 = default)
   1.73          0.0
* tt_dep0  tt_dep1  tt_ddep
   0.       15.      0.08
* tt_del0  tt_del1  tt_ddel
   0.       50.      0.1
*
******************************************
***** GrowClust Algorithm Parameters *****
******************************************
* rmin  delmax rmsmax
   0     20     0.2
* rpsavgmin, rmincut  ngoodmin   iponly
   0          0        0          0
*
******************************************
************ Output files ****************
******************************************
* nboot  nbranch_min
   100    1
* fout_cat (relocated catalog)
OUT/out.growclust_cat
* fout_clust (relocated cluster file)
OUT/out.growclust_clust
* fout_log (program log)
OUT/out.growclust_log
* fout_boot (bootstrap distribution)
OUT/out.growclust_boot
******************************************
******************************************
"""
    with open(f"{out_dir}/growclust.inp", "w") as f:
        f.write(ctl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", choices=["pyocto", "qm"], required=True)
    args = ap.parse_args()
    src = SOURCES[args.catalog]

    for d in ("IN", "OUT", "TT"):
        os.makedirs(f"{src['out_dir']}/{d}", exist_ok=True)

    keep_ids = convert_evlist(src["event_sel"], src["station_sel"],
                               f"{src['out_dir']}/IN/evlist.txt", radius_km=src["radius_km"])
    n_sta = convert_stlist(src["station_sel"], f"{src['out_dir']}/IN/stlist.txt")
    n_pairs, n_obs = convert_xcordata(src["dt_cc"], f"{src['out_dir']}/IN/xcordata.txt",
                                       keep_ids=keep_ids if src["radius_km"] is not None else None)
    write_vzmodel(f"{src['out_dir']}/IN/vzmodel.txt")
    write_control_file(src["out_dir"], args.catalog, len(keep_ids))

    print(f"[{args.catalog}] {len(keep_ids)} events, {n_sta} stations, "
          f"{n_pairs} event pairs / {n_obs} xcor observations")
    print(f"[{args.catalog}] wrote {src['out_dir']}/{{IN/evlist.txt,IN/stlist.txt,"
          f"IN/xcordata.txt,IN/vzmodel.txt,growclust.inp}}")


if __name__ == "__main__":
    main()
