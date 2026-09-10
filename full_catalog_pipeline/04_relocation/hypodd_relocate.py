#!/usr/bin/env python3
"""Double-difference relocation of a pyocto event catalog with hypoDD, via
hypoddpy's HypoDDRelocator (generates phase/event/station files and runs the
waveform cross-correlation), followed by the user's previously-validated T1
hypoDD parameter sweep (see ~/dr/time/input_files/sweep_d150_hcc.inp -- not
part of this repo) instead of hypoDDpy's own default weighting scheme.

hypoDDpy's public API can't reproduce the winning per-layer Vp/Vs profile
(setup_velocity_model only supports one constant ratio for all layers, and its
"layered_variable_vp_vs_ratio" mode has a bug that drops the ratio line
entirely), nor its 5-iteration reweighting scheme (vs hypoDDpy's hardcoded 2).
So this script runs HypoDDRelocator's private pipeline stages directly (the
same sequence start_relocation() calls) and overwrites input_files/hypoDD.inp
with the exact validated block right before invoking the hypoDD binary.

T2 has no equivalent prior sweep. Its velocity model reuses T2's own
established ice velocity (Vp=3.841 km/s, from
/scratch2/qm/t2/growclust/tests.inp) and ice-bed interface depth (~2.02 km,
BedMachine/Bedmap2 -- same source as pyocto's T2 velocity model), with the
corrected Vp/Vs=1.73 the T1 sweep validated (user's explicit choice over
keeping T2's original 1.949), and T1's bedrock Vp progression shape shifted to
start at T2's own interface depth (no T2-specific deep structure is known).
T2 reuses T1's winning ph2dt/reweighting scheme verbatim -- no basis to expect
otherwise and nothing to sweep against.

See METHODS.md (relocation section) for the full rationale and current results.

Usage:
    python full_catalog_pipeline/hypodd_relocate.py --array T1
    python full_catalog_pipeline/hypodd_relocate.py --array T2
"""
import argparse
import glob
import logging
import os
from datetime import timedelta

from obspy import read_events
from hypoddpy import HypoDDRelocator

# HypoDDRelocator.log() unconditionally print()s every message regardless of
# logging level -- logging.getLogger().setLevel(WARNING) in run_array() only
# suppresses stdlib `logging` calls (e.g. obspy's), not these. Its ~73
# level="debug" call sites fire once per trace/pick/pair in the cross-
# correlation hot loop, which is what exploded hypodd_relocate_driver.log to
# 1.5GB+ in a few hours and filled the disk on 2026-07-11. Patch it to drop
# debug-level messages (still routed through logging.debug so nothing is
# silently lost if a handler wants it) while leaving info/warning/error
# messages -- and progress reporting -- intact.
_original_relocator_log = HypoDDRelocator.log


def _quiet_relocator_log(self, string, level="info"):
    if level == "debug":
        logging.debug(string)
        return
    _original_relocator_log(self, string, level=level)


HypoDDRelocator.log = _quiet_relocator_log

DAY_VOLUMES_DIR = "/data/time/day_volumes"

ARRAY_STATIONS = {
    "T1": ["DEEJ", "ELZA", "LILA", "LOUS", "OTIS", "SQIG", "TJTJ"],
    "T2": ["BAUM", "DRSC", "EPJZ", "FRST", "JULA", "OKGS", "WICH"],
}
# /data/time/day_volumes/time.xml uses network code "2E", but the raw waveform archive's
# files (and their trace headers once read) use "7U" -- confirmed by a sanity run where
# cross-correlation found zero traces for every station ("available=[]") despite station.dat
# being built fine, because hypoDDpy keys its internal station/trace lookup by
# "network.station" straight from the StationXML. build_station_xml() below regenerates a
# corrected copy with network "7U" from the same lat/lon source associate_pyocto.py uses.
STATIONS_JSON = "/scratch2/qm/t1/output/working_files/stations.json"
STATION_XML = "full_catalog_pipeline/artifacts/hypodd_station_7U.xml"


def build_station_xml():
    import json
    from obspy.core.inventory import Inventory, Network, Station, Channel

    with open(STATIONS_JSON) as f:
        stations = json.load(f)

    net = Network(code="7U", stations=[])
    for key, meta in stations.items():
        code = key.split(".")[1]
        net.stations.append(Station(
            code=code, latitude=meta["latitude"], longitude=meta["longitude"],
            elevation=meta["elevation"], channels=[
                Channel(code=chan, location_code="", latitude=meta["latitude"],
                        longitude=meta["longitude"], elevation=meta["elevation"],
                        depth=0.0, sample_rate=100.0)
                for chan in ["HHZ", "HH1", "HH2"]
            ],
        ))
    Inventory(networks=[net], source="icequake_ml").write(STATION_XML, format="STATIONXML")
    print(f"wrote {len(net.stations)} stations to {STATION_XML}")

# Winning T1 ph2dt config (~/dr/time/input_files/ph2dt.inp), reused verbatim for T2 --
# EXCEPT MAXSEP, which is catalog-geometry-dependent (hypoDDpy itself derives it as the
# 5th-percentile inter-event distance) and does NOT transfer: the old value (3.03 km) was
# computed from the original 10,504-event QuakeMigrate catalog, spread across up to ~70 km.
# pyocto's catalogs are confined to ~10 km (a structural limit of its associator's search
# box -- see the map-comparison plots in [[pyocto-full-catalog-rebuild]]), so reusing 3.03 km
# verbatim swept in 525,684 event pairs for T1's 2400 events (vs 34,321 pairs for the original
# 10,504 events -- a ~68x higher pairs-per-event ratio), which would have taken many more
# hours of cross-correlation for a bloated, weakly-connected pair set. Recompute MAXSEP fresh
# per array with the same percentile methodology instead.
PH2DT_PARAMETERS_BASE = {
    "MINWGHT": 0.0,
    "MAXDIST": 528,
    "MAXNGH": 5,
    "MINLNK": 10,
    "MINOBS": 6,
    "MAXOBS": 50,
}


def compute_maxsep(quakeml_path, percentile=0.05, sample_size=800):
    """5th-percentile inter-event distance -- sampled for speed on large catalogs
    (O(n^2) pairs otherwise).

    Uses a haversine horizontal distance rather than hypoDDpy's own
    lat_deg*111.0 / lon_deg*111.0 approach (both its default MAXSEP/MAXDIST
    calculation in _write_ph2dt_inp_file and the original version of this
    function copied that same shortcut). Treating longitude degrees as if they
    were the same length as latitude degrees is only valid at the equator; at
    this array's latitude (~-77.3 deg) cos(77.3 deg) = 0.22, so 1 deg of
    longitude is ~24.6 km, not 111 km -- a ~4.5x overestimate of east-west
    separation. That inflated the computed T1 MAXSEP to 0.975 km when the
    geometrically correct 5th-percentile value is ~0.48 km, meaning ph2dt was
    linking event pairs roughly 2x farther apart than intended.
    """
    import math
    import random

    cat = read_events(quakeml_path)
    events = [e.preferred_origin() for e in cat if e.preferred_origin()]
    if len(events) > sample_size:
        random.seed(0)
        events = random.sample(events, sample_size)

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    distances = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            horiz_range = haversine_km(
                events[i].latitude, events[i].longitude,
                events[j].latitude, events[j].longitude,
            )
            depth_range = abs(events[i].depth - events[j].depth) / 1000.0
            distances.append(math.sqrt(horiz_range**2 + depth_range**2))
    distances.sort()
    return distances[int(len(distances) * percentile)]

# T1's official, held-out-CV-validated hypoDD.inp reweighting scheme ("ccscale_0.33", see
# hypodd_cc_heldout_cv_validation memory's FINAL DECISION) -- verbatim copy of the weights
# actually applied to T1_v5/hypodd/input_files/hypoDD.inp, reused for T2. NOTE: this used to
# hold the older, pre-CV-validation "hcc" (heavy-CC-weight) scheme from
# ~/dr/time/input_files/sweep_d150_hcc.inp, which was superseded by ccscale_0.33 on
# 2026-07-26 for T1 but never updated here -- T2's first relocation run (completed
# 2026-08-11) used that stale scheme and must be rerun with this corrected one.
HYPODD_DIST = 150
REWEIGHTING_SCHEME = [
    "6 0.99 0.99 -999 -999  0.60 0.60 10 100  40",
    "6 0.33 0.33 -999 -999  0.40 0.40 10 100  30",
    "6 0.10 0.10 0.06   50  0.20 0.20  3  20  15",
    "6 0.03 0.03 0.04   20  0.05 0.05 -999 -999   8",
    "8 0.02 0.02 0.02   10  0.05 0.05 -999 -999   5",
]

ARRAY_CONFIG = {
    "T1": {
        "quakeml": "full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.quakeml",
        "working_dir": "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd",
        # (depth_km, vp_km_s, vs_km_s) -- verbatim from sweep_d150_hcc.inp
        "velocity_layers": [
            (0.0, 2.50, 1.84),
            (0.1, 3.85, 2.22),
            (3.1, 5.1, 2.95),
            (3.8, 5.8, 3.35),
            (9.5, 6.10, 3.53),
            (14.0, 6.5, 3.76),
            (25.0, 7.5, 4.34),
            (52.0, 8.05, 4.65),
        ],
    },
    "T2": {
        "quakeml": "full_catalog_pipeline/artifacts/full_run/T2_v5/pyocto_events.quakeml",
        "working_dir": "full_catalog_pipeline/artifacts/full_run/T2_v5/hypodd",
        # Ice layer: T2's own Vp=3.841 (growclust), Vp/Vs=1.73 (T1's validated correction).
        # Below: T1's bedrock Vp progression, depths shifted by (2.02 - 3.1) = -1.08 km so
        # the ice-bed transition sits at T2's own ~2.02 km BedMachine/Bedmap2 thickness.
        "velocity_layers": [
            (0.0, 2.50, 1.84),
            (0.1, 3.841, round(3.841 / 1.73, 4)),
            (2.02, 5.1, round(5.1 / 1.73, 4)),
            (2.72, 5.8, round(5.8 / 1.73, 4)),
            (8.42, 6.10, round(6.10 / 1.73, 4)),
            (12.92, 6.5, round(6.5 / 1.73, 4)),
            (23.92, 7.5, round(7.5 / 1.73, 4)),
            (50.92, 8.05, round(8.05 / 1.73, 4)),
        ],
    },
}


def waveform_files_for_array(quakeml_path, stations, buffer_hours=6.0):
    """Pre-filter the raw archive to just this array's stations and the day-range
    the catalog actually spans (+/- buffer). hypoDDpy's own enable_smart_waveform_filtering
    assumes ObsPy-mass-downloader-style filenames (NET.STA.LOC.CHAN__START__END.mseed) and
    silently keeps everything for our archive's actual STA.NET.LOC.CHAN.YEAR.DOY naming --
    verified 0% reduction on a test run -- so we filter ourselves instead."""
    cat = read_events(quakeml_path)
    times = [e.preferred_origin().time for e in cat if e.preferred_origin()]
    min_time = min(times) - buffer_hours * 3600
    max_time = max(times) + buffer_hours * 3600

    files = []
    day = min_time.date
    end_day = max_time.date
    while day <= end_day:
        day_str = day.strftime("%Y%m%d")
        for sta in stations:
            files.extend(glob.glob(f"{DAY_VOLUMES_DIR}/{day_str}/{sta}.7U..HH?.*"))
        day += timedelta(days=1)
    return sorted(files)


def build_hypodd_inp(velocity_layers):
    """NOTE (found 2026-08-25, kept here as-is so the v5 runs stay reproducible): hypoDD's
    imod=1 model block is TOP / VP / RATIO -- the third line is the per-layer Vp/Vs RATIO,
    not Vs. The `vss` written below are Vs values, so every v5 relocation actually ran with
    Vs = Vp/vss, i.e. a near-constant ~1.73 km/s from the firn down through the bedrock
    (see the MOD_VS column in T1_v5/hypodd/hypoDD_log.txt). hypodd_relocate_vels1d.py, which
    supersedes this script, emits real ratios from the measured reflection profile."""
    depths = " ".join(str(d) for d, _, _ in velocity_layers)
    vps = " ".join(str(vp) for _, vp, _ in velocity_layers)
    vss = " ".join(str(vs) for _, _, vs in velocity_layers)
    lines = [
        "hypoDD_2",
        "dt.cc",
        "dt.ct",
        "event.sel",
        "station.sel",
        "",
        "",
        "hypoDD.sta",
        "hypoDD.res",
        "hypoDD.src",
        f"3 3 {HYPODD_DIST}",
        "3 4 1 -999 -999",
        "2 2 1 5",
        *REWEIGHTING_SCHEME,
        "1",
        depths,
        vps,
        vss,
        "0",
        "",
    ]
    return "\n".join(lines)


# Cross-correlation upsampling factor. See install_cc_upsampling_shim() for why this is
# needed at all; 2 is sufficient and 4/8 measurably add nothing.
CC_UPSAMPLE = 2


def install_cc_upsampling_shim(factor=CC_UPSAMPLE):
    """Interpolate both traces before hypoDDpy hands them to obspy's cross-correlation.

    Without this, T2's dt.cc comes out 97.8% S -- 17,015 P against 743,545 S -- even though
    the catalog picks (16.4k P / 15.5k S) and dt.ct (1.79M P / 1.78M S) are balanced. The
    cause is not correlation quality and not the picks. obspy's xcorr_pick_correction fits a
    parabola to the convex region around the cross-correlation peak to get sub-sample timing,
    and raises "Less than 3 samples selected for fit to cross correlation" when that region
    is narrower than three samples. At 200 Hz with a 20-90 Hz passband, content near the 90 Hz
    top of the band produces a CC function whose period is ~2.2 samples, so the region is 1-2
    samples wide and the fit is refused. Measured on real dt.ct pairs, the refusal rate is
    98.3% for P on Z -- but also 76% for S on Z, 73% on E and 65% on N. It hits EVERY phase.

    Two things then separate P from S, and both are measured (see
    plot_cc_upsampling_explainer.py, panel C):
      * on the same channel P is refused more often: 1.7% of P/Z attempts clear the
        coefficient threshold against 18.3% of S/Z, a factor of 10.8. P's coherent energy
        oscillates somewhat faster (~70 Hz against S's ~60 Hz median), so its lobe is the
        narrower of the two -- a real but modest effect, NOT the whole story.
      * S gets three attempts and P gets one: cc_s_phase_weighting offers Z, E and N and
        _perform_cross_correlation returns on the first channel clearing the threshold, while
        cc_p_phase_weighting offers only Z. Worth a further 2.3x (S 41.3% over three channels
        against 18.3% on Z alone).
    Together those account for the observed ~24x gap. It is emphatically NOT a data-quality
    problem: the P that does survive has a median coefficient of 0.79-0.85, HIGHER than S's
    0.53-0.71.

    Interpolating x2 before the correlation widens that convex region in samples without
    changing the passband and without inventing information. Measured pass rates at the
    0.4 coefficient threshold (full table in diagnose_cc_p_deficit.py --fix-sweep):

        200 Hz (as-is)  P   1.0%   S(Z) 19.7%   99.0% of P refused by the fit
        400 Hz (x2)     P  99.7%   S(Z) 56.3%    0.0% refused
        800 Hz (x4)     P  99.7%   S(Z) 55.7%    0.0% refused

    Lanczos is used rather than a cheaper interpolant because this is a TIMING measurement;
    a=20 costs ~1.6 ms per trace, negligible beside the ~3 ms correlation itself.

    Applied as a module-level shim because hypoDDpy binds the function at import
    (`from obspy.signal.cross_correlation import xcorr_pick_correction`, hypodd_relocator.py
    line 75) and exposes no hook. This driver already calls hypoDDpy's private stages
    directly, so patching one private name is consistent with how it works.
    """
    import hypoddpy.hypodd_relocator as hr

    if getattr(hr, "_icequake_upsampling_installed", False):
        return
    original = hr.xcorr_pick_correction

    def upsampled(pick1, trace1, pick2, trace2, **kwargs):
        t1, t2 = trace1.copy(), trace2.copy()
        for tr in (t1, t2):
            tr.interpolate(sampling_rate=tr.stats.sampling_rate * factor,
                           method="lanczos", a=20)
        return original(pick1, t1, pick2, t2, **kwargs)

    hr.xcorr_pick_correction = upsampled
    hr._icequake_upsampling_installed = True
    print(f"[cc] upsampling shim installed: traces interpolated x{factor} "
          f"(lanczos a=20) before cross-correlation")


def run_array(array, max_threads):
    cfg = ARRAY_CONFIG[array]
    os.makedirs(cfg["working_dir"], exist_ok=True)
    if not os.path.exists(STATION_XML):
        build_station_xml()

    maxsep = compute_maxsep(cfg["quakeml"])
    print(f"[{array}] MAXSEP (5th-pct inter-event distance) = {maxsep:.4f} km")
    ph2dt_parameters = {**PH2DT_PARAMETERS_BASE, "MAXSEP": maxsep}

    relocator = HypoDDRelocator(
        working_dir=cfg["working_dir"],
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=20.0,
        cc_filter_max_freq=90.0,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.4,
        ph2dt_parameters=ph2dt_parameters,
    )
    # HypoDDRelocator.__init__ calls logging.basicConfig(level=logging.DEBUG, ...), which
    # sets the ROOT logger to DEBUG -- this also enables DEBUG-level logging from every
    # imported library (obspy etc.), not just hypoDDpy's own ~240 self.log() calls. During
    # cross-correlation over hundreds of thousands of event pairs this exploded log.txt to
    # 1.5GB+ in a few hours and filled the disk (2026-07-11 OSError: No space left on
    # device). Raise it back down; hypoDDpy's own progress messages also go through
    # self.log()'s print() call, so they still reach stdout regardless of this level.
    logging.getLogger().setLevel(logging.WARNING)

    relocator.add_event_files([cfg["quakeml"]])
    stations = ARRAY_STATIONS.get(array, ARRAY_STATIONS.get(array.split("_")[0]))
    waveform_files = waveform_files_for_array(cfg["quakeml"], stations)
    print(f"[{array}] {len(waveform_files)} waveform files selected (stations={stations})")
    relocator.add_waveform_files(waveform_files)
    relocator.add_station_files([STATION_XML])

    # Placeholder velocity model: only used so hypoDDpy's internal bookkeeping has
    # something to reference before we overwrite the real hypoDD.inp below with the
    # exact validated per-layer Vp/Vs block (which the public API can't express).
    layer_tops = [(d, vp) for d, vp, _ in cfg["velocity_layers"]]
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        layer_tops=layer_tops,
        vp_vs_ratio=1.73,
    )

    output_event_file = os.path.join(cfg["working_dir"], "relocated_events.xml")
    relocator.output_event_file = output_event_file
    if os.path.exists(output_event_file):
        print(f"[{array}] {output_event_file} already exists, nothing to do.")
        return

    relocator._parse_station_files()
    relocator._write_station_input_file()
    relocator._read_event_information(max_threads)
    relocator._write_ph2dt_inp_file()
    relocator._create_event_id_map()
    relocator._write_catalog_input_file()
    relocator._run_ph2dt()
    relocator._parse_waveform_files()
    install_cc_upsampling_shim()
    relocator._cross_correlate_picks()
    relocator._write_hypoDD_inp_file()

    hypodd_inp_path = os.path.join(cfg["working_dir"], "input_files", "hypoDD.inp")
    with open(hypodd_inp_path, "w") as f:
        f.write(build_hypodd_inp(cfg["velocity_layers"]))
    print(f"[{array}] wrote validated hypoDD.inp to {hypodd_inp_path}")

    relocator._run_hypodd()

    # A handful of numerically unstable events ("air quakes") can come out of hypoDD with
    # NaN coordinates; Fortran prints their date fields as literal asterisks ("****") in
    # that case, which hypoDDpy's _create_output_event_file() chokes on (int("****")). Drop
    # those rows before it reads the file rather than losing the whole array's output to one
    # ValueError -- this is expected/acceptable event dropout, not a sign the run failed.
    reloc_path = os.path.join(cfg["working_dir"], "output_files", "hypoDD.reloc")
    with open(reloc_path) as f:
        lines = f.readlines()
    kept = [ln for ln in lines if "*" not in ln]
    n_dropped = len(lines) - len(kept)
    if n_dropped:
        print(f"[{array}] dropping {n_dropped} event(s) with invalid (NaN/'****') "
              f"hypoDD.reloc fields before writing final output")
        with open(reloc_path, "w") as f:
            f.writelines(kept)

    relocator._create_output_event_file()
    print(f"[{array}] relocation complete: {output_event_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", choices=sorted(ARRAY_CONFIG), required=True)
    ap.add_argument("--max-threads", type=int, default=8)
    args = ap.parse_args()
    run_array(args.array, args.max_threads)


if __name__ == "__main__":
    main()
