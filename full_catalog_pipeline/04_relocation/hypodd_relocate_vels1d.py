#!/usr/bin/env python3
"""Re-run the hypoDD relocation of T1 and T2 with the reflection-seismology 1D velocity
model (full_catalog_pipeline/vels1d/, see vels1d_model.py) in place of the hand-assembled
models in hypodd_relocate.py's ARRAY_CONFIG.

Only the velocity model changes. hypoDD's other two inputs -- dt.ct (catalog differential
times, from ph2dt) and dt.cc (waveform cross-correlation differential times) -- are both
derived purely from picks and waveforms and do not depend on the velocity model at all, so
this reuses the existing v5 run's input_files verbatim rather than spending another ~10 hours
re-cross-correlating. event.sel (the ph2dt-selected initial locations, from pyocto) is reused
for the same reason: the task is redoing the RELOCATION, not the association.

Reusing dt.ct/dt.cc means the numeric event ids in them must line up with the event map this
run builds. That map is built deterministically from working_files/events.json, which is
copied over with everything else, and verify_event_map_matches() below proves it by
regenerating phase.dat and diffing it against the v5 one before hypoDD is ever invoked.

The ph2dt/reweighting/damping configuration is the one validated for T1 in
[[hypodd-cc-heldout-cv-validation]] (ccscale_0.33, CC-inclusive -- see
[[hypodd-cc-must-be-included]]), imported unchanged from hypodd_relocate.py.

Usage:
    python full_catalog_pipeline/hypodd_relocate_vels1d.py --array T2
    python full_catalog_pipeline/hypodd_relocate_vels1d.py --array T1
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
import filecmp
import logging
import os
import shutil

from hypoddpy import HypoDDRelocator

from hypodd_relocate import (
    ARRAY_CONFIG,
    HYPODD_DIST,
    REWEIGHTING_SCHEME,
    STATION_XML,
    build_station_xml,
)
from vels1d_model import ICE_THICKNESS_KM, build_hypodd_model_lines, build_layers

# Files that carry over unchanged from the v5 run. dt.cc/dt.ct/event.sel/station.sel are what
# hypoDD actually reads; event.dat/phase.dat/ph2dt.inp come along so the new working dir is a
# self-describing record of what produced them.
REUSED_INPUT_FILES = [
    "dt.cc", "dt.ct", "event.sel", "station.sel",
    "event.dat", "phase.dat", "ph2dt.inp", "station.dat",
]
REUSED_WORKING_FILES = ["events.json", "stations.json"]

OUTPUT_SUBDIR = "hypodd_vels1d"  # sits beside the v5 run's "hypodd" dir, does not touch it


def source_dir(array):
    return ARRAY_CONFIG[array]["working_dir"]


def target_dir(array):
    return os.path.join(os.path.dirname(source_dir(array)), OUTPUT_SUBDIR)


def stage_inputs(array):
    """Copy the v5 run's velocity-model-independent inputs into the new working dir."""
    src, dst = source_dir(array), target_dir(array)
    for sub, names in [("input_files", REUSED_INPUT_FILES), ("working_files", REUSED_WORKING_FILES)]:
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        for name in names:
            src_path = os.path.join(src, sub, name)
            dst_path = os.path.join(dst, sub, name)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"{src_path} missing -- run hypodd_relocate.py first")
            if os.path.exists(dst_path):
                continue
            if sub == "working_files":
                # events.json is REWRITTEN in place by _read_event_information (it re-saves
                # the cache after renaming any duplicate event ids), so this one must be a
                # real copy -- a hardlink would edit the v5 run's file too.
                shutil.copy2(src_path, dst_path)
                continue
            # input_files are read-only here, so hardlink: dt.ct alone is 128 MB for T2.
            try:
                os.link(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)
    os.makedirs(os.path.join(dst, "output_files"), exist_ok=True)
    print(f"[{array}] staged v5 inputs into {dst}")


def verify_event_map_matches(relocator, array):
    """Regenerate phase.dat from this run's event map and diff it against the v5 one.

    phase.dat carries the mapped numeric event ids, so an identical file proves this run
    numbers events exactly as the run that produced the dt.ct/dt.cc we are reusing. A
    mismatch would silently relocate events against other events' differential times.
    """
    phase_dat = os.path.join(target_dir(array), "input_files", "phase.dat")
    reference = phase_dat + ".v5_reference"
    os.rename(phase_dat, reference)
    try:
        relocator._write_catalog_input_file()
        if not filecmp.cmp(phase_dat, reference, shallow=False):
            raise RuntimeError(
                f"[{array}] regenerated phase.dat differs from the v5 one -- the event id map "
                f"does not match the dt.ct/dt.cc being reused; aborting.")
        print(f"[{array}] event id map verified identical to the v5 run ({len(relocator.events)} events)")
    finally:
        os.remove(reference)


def build_hypodd_inp(layers):
    tops, vps, ratios = build_hypodd_model_lines(layers)
    return "\n".join([
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
        tops,
        vps,
        # imod=1's third line is the per-layer Vp/Vs RATIO, not Vs -- see vels1d_model.py's
        # module docstring for how the previous configs got this wrong.
        ratios,
        "0",
        "",
    ])


def run_array(array, max_threads):
    dst = target_dir(array)
    stage_inputs(array)
    if not os.path.exists(STATION_XML):
        build_station_xml()

    layers = build_layers(array)
    print(f"[{array}] vels1d model: {len(layers)} layers, "
          f"ice thickness {ICE_THICKNESS_KM[array]} km, "
          f"Vp {layers[0][1]:.3f} km/s at the surface -> {layers[-1][1]:.3f} km/s in bedrock")

    relocator = HypoDDRelocator(
        working_dir=dst,
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=20.0,
        cc_filter_max_freq=90.0,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.4,
    )
    # See hypodd_relocate.py: HypoDDRelocator.__init__ turns the ROOT logger up to DEBUG,
    # which floods the log with every imported library's debug output.
    logging.getLogger().setLevel(logging.WARNING)

    cfg = ARRAY_CONFIG[array]
    relocator.add_event_files([cfg["quakeml"]])
    relocator.add_station_files([STATION_XML])

    output_event_file = os.path.join(dst, "relocated_events.xml")
    relocator.output_event_file = output_event_file
    if os.path.exists(output_event_file):
        print(f"[{array}] {output_event_file} already exists, nothing to do.")
        return

    relocator._parse_station_files()
    relocator._read_event_information(max_threads)
    relocator._create_event_id_map()
    verify_event_map_matches(relocator, array)

    hypodd_inp_path = os.path.join(dst, "input_files", "hypoDD.inp")
    with open(hypodd_inp_path, "w") as f:
        f.write(build_hypodd_inp(layers))
    print(f"[{array}] wrote vels1d hypoDD.inp to {hypodd_inp_path}")

    relocator._run_hypodd()

    # Numerically unstable events can come out with NaN coordinates, which Fortran writes as
    # "****"; hypoDDpy's reader chokes on those. Same handling as hypodd_relocate.py.
    reloc_path = os.path.join(dst, "output_files", "hypoDD.reloc")
    with open(reloc_path) as f:
        lines = f.readlines()
    kept = [ln for ln in lines if "*" not in ln]
    if len(kept) != len(lines):
        print(f"[{array}] dropping {len(lines) - len(kept)} event(s) with invalid "
              f"(NaN/'****') hypoDD.reloc fields")
        with open(reloc_path, "w") as f:
            f.writelines(kept)

    relocator._create_output_event_file()
    print(f"[{array}] relocation complete: {output_event_file} ({len(kept)} events)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", required=True, choices=sorted(ARRAY_CONFIG))
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()
    run_array(args.array, args.max_threads)


if __name__ == "__main__":
    main()
