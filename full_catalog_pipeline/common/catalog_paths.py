#!/usr/bin/env python3
"""Single source of truth for which hypoDD relocation is *authoritative* for each array.

Before this module every downstream script hard-coded
`.../{ARRAY}_v5/hypodd/output_files/hypoDD.reloc`, so switching to a new relocation meant
editing ~40 files and hoping none were missed. Scripts now import from here instead, and the
switch is one edit.

Two ideas keep the indirection honest:

  * Inputs vs. outputs are separate. `station_sel()`, `dt_cc()`, `event_sel()` and
    `phase_dat()` are ph2dt/cross-correlation products that are byte-identical across
    relocation runs (the vels1d run hardlinks them from the v5 run -- only the velocity model
    and the inversion settings differ). They are exposed here anyway so a script never has to
    name a run directory itself.

  * `work_dir()` is where DERIVED artifacts live -- cluster event-id lists, polarity CSVs,
    stacks, figures. It follows the authoritative relocation deliberately: cluster membership
    is a property of the relocation that produced it, so artifacts derived from one relocation
    must not be silently reused against another. Anything under a previous run's work_dir
    belongs to that run.

`scratch_dir()` is the third category: ephemeral run directories for scripts that shell out
to hypoDD. Nothing under it is an analysis product -- it is trimmed input files and hypoDD's
own output for one experimental rerun -- so it is untracked and safe to delete. It is named
here rather than per-script so a rerun lands somewhere reproducible instead of in whatever
temp directory the original run happened to use.

Set ICEQUAKE_RELOC to override the authoritative relocation at runtime (e.g.
ICEQUAKE_RELOC=hypodd to reproduce a figure against the superseded v5 relocation without
editing anything), or ICEQUAKE_SCRATCH to put scratch runs on a different disk.
"""
import os

# Directory name, under artifacts/full_run/{ARRAY}_v5/, of the authoritative relocation.
#   "hypodd"          -- the superseded v5 run (hand-assembled velocity model, wrong Vs)
#   "hypodd_vels1d"   -- reflection-seismology velocity model, tuned inversion settings
AUTHORITATIVE = os.environ.get("ICEQUAKE_RELOC", "hypodd_vels1d")

ARRAYS = ["T1", "T2"]
_ROOT = "full_catalog_pipeline/artifacts/full_run"
_SCRATCH = os.environ.get("ICEQUAKE_SCRATCH", "full_catalog_pipeline/scratch")


def run_dir(array, which=None):
    return os.path.join(_ROOT, f"{array}_v5", which or AUTHORITATIVE)


def work_dir(array, which=None):
    """Where derived artifacts for this relocation live (also the relocation's own dir)."""
    return run_dir(array, which)


def scratch_dir(name):
    """Ephemeral working directory for a script that shells out to hypoDD.

    Unlike work_dir(), this is not tied to a relocation: it holds the trimmed event.sel/dt.*
    files and hypoDD's raw output for one experimental rerun. Delete it freely.
    """
    return os.path.join(_SCRATCH, name)


def reloc(array, which=None):
    return os.path.join(run_dir(array, which), "output_files", "hypoDD.reloc")


def reloc_xml(array, which=None):
    return os.path.join(run_dir(array, which), "relocated_events.xml")


def src(array, which=None):
    return os.path.join(run_dir(array, which), "output_files", "hypoDD.src")


def input_dir(array, which=None):
    return os.path.join(run_dir(array, which), "input_files")


def station_sel(array, which=None):
    return os.path.join(input_dir(array, which), "station.sel")


def dt_cc(array, which=None):
    return os.path.join(input_dir(array, which), "dt.cc")


def event_sel(array, which=None):
    return os.path.join(input_dir(array, which), "event.sel")


def phase_dat(array, which=None):
    return os.path.join(input_dir(array, which), "phase.dat")


def describe():
    lines = [f"authoritative relocation: {AUTHORITATIVE}"]
    for a in ARRAYS:
        p = reloc(a)
        n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        lines.append(f"  {a}: {p} ({n} events)" if n else f"  {a}: {p} (MISSING)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
