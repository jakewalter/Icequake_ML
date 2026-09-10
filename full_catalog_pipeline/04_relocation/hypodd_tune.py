#!/usr/bin/env python3
"""Sweep hypoDD's iteration/inversion controls on top of the reflection-seismology velocity
model, and score each config the way [[hypodd-cc-heldout-cv-validation]] established: by
held-out cross-correlation cross-validation, NOT by how compact the resulting cloud looks.

Why re-tune at all, when the ccscale_0.33 weighting block was already validated? Because that
validation ran under a velocity model whose Vs was wrong (see vels1d_model.py) -- catalog
residuals at T1 were ~36 ms then and are ~7 ms now. Damping, iteration count and the ABSOLUTE
(seconds) cross-correlation residual cutoffs were all chosen against the old residual scale,
so they are the knobs most likely to be mistuned for the corrected model. The CC weight taper
itself is held fixed: it was validated (Task C), and CC weight must stay > 0
([[hypodd-cc-must-be-included]]).

Three numbers per config, deliberately measuring different things:

  accuracy   Held-out CV RMS. Split dt.cc 80/20 by observation line; relocate on dt.ct + the
             80% ("train"); then score those hypocentres against the untouched 20% with a
             second, single-iteration hypoDD run whose event.sel has been overwritten with the
             train locations, so hypoDD.res reports obs-calc at those fixed locations. This is
             the only metric in the previous battery that survived an adversarial artifact
             check, so it is the primary one.

  stability  Median horizontal+vertical displacement between the train run (80% of the CC
             data) and the production run (100%). Same config, same events, 20% of the
             cross-correlation constraint perturbed -- if the picture moves a lot, it is not
             stable. This is free: both runs are needed anyway.

  coverage   Events relocated, and the fraction of dt.ct/dt.cc surviving hypoDD's own outlier
             trimming at the last iteration. A config can post a flattering RMS purely by
             discarding more data, so accuracy is never read without this beside it.

Plus, straight from hypoDD's log: the final-iteration step sizes (DX/DY/DZ -- large steps at
the last iteration mean it never converged) and condition number CND.

Usage:
    python full_catalog_pipeline/hypodd_tune.py run   --array T1 --config base
    python full_catalog_pipeline/hypodd_tune.py sweep --array T1 --jobs 4
    python full_catalog_pipeline/hypodd_tune.py report --array T1
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
import concurrent.futures
import filecmp
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from hypodd_relocate import ARRAY_CONFIG, HYPODD_DIST
from vels1d_model import build_hypodd_model_lines, build_layers

HYPODD_BIN = "/home/jwalter/bin/hypoDD"
CV_SEED = 42
# Smallest cluster whose median-depth shift is trusted as a stability signal. Below this a
# handful of events can swing the median and the "worst cluster" becomes noise; T2's real
# offenders (cluster 1 at ~1700 events, cluster 2 at ~670) are far above it.
MIN_CLUSTER_FOR_SHIFT = 50
CV_TRAIN_FRAC = 0.8

# Which cross-correlation file in input_files/ the sweep reads, set by --dtcc. Anything
# other than the original "dt.cc" is a DIFFERENT dataset -- the upsampled regeneration of
# [[cc-p-deficit-root-cause]] has 4.4x the observations and 53% P against 2.2% -- so its
# results are not comparable with the original sweep's and must not land in the same
# directory. tune_root() therefore carries a suffix derived from the filename, and every
# subcommand (run/sweep/report/promote) needs the same --dtcc to address the same tree.
SRC_DTCC = "dt.cc"

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "year", "month", "day", "hour", "minute", "second", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]
EVENT_SEL_COLS = ["date", "time", "lat", "lon", "depth", "mag", "eh", "ez", "rms", "id"]

# The vels1d run's block: the ccscale_0.33 weighting validated for T1, five stages.
BASE = {
    "niter": [6, 6, 6, 6, 8],
    "wt_cc": [0.99, 0.33, 0.10, 0.03, 0.02],
    "wrcc": [-999, -999, 0.06, 0.04, 0.02],   # < 1 => ABSOLUTE cutoff in seconds
    "wdcc": [-999, -999, 50, 20, 10],
    "wt_ct": [0.60, 0.40, 0.20, 0.05, 0.05],
    "wrct": [10, 10, 3, -999, -999],          # >= 1 => multiple of the residual MAD
    "wdct": [100, 100, 20, -999, -999],
    "damp": [40, 30, 15, 8, 5],
}


# Scalar (non-per-stage) knobs a config may override. Kept separate from the per-stage
# weighting arrays so cfg() does not try to list()-ify them.
SCALARS = {
    "velocity": "vels1d",
    "dtcc_max_s": None,   # drop CC observations whose |dt| exceeds this (seconds)
    "dtcc_min_w": None,   # drop CC observations whose coefficient is below this
    "minobs_cc": 3,       # MINOBS_CC: CC links an event needs to join a cluster
    "minobs_ct": 4,       # MINOBS_CT
    "istart": 2,          # 1 = start from cluster centroid, 2 = from catalog locations
}


def cfg(**overrides):
    out = {k: list(v) for k, v in BASE.items()}
    out.update({k: list(v) for k, v in overrides.items() if k not in SCALARS})
    for k, default in SCALARS.items():
        out[k] = overrides.get(k, default)
    return out


# The evaluation set is held fixed across configs so held-out accuracy stays comparable even
# when configs train on differently filtered CC data. Observations beyond this cap are not
# predictable by ANY solution -- a 232 s differential time across a sub-kilometre pair is a
# cycle-skip, not a datum -- so scoring against them only adds noise to every config equally.
EVAL_MAX_DT_S = 0.5


# One factor at a time off the base, so a difference in score is attributable to one knob.
CONFIGS = {
    "base": cfg(),
    # Control: the superseded v5 velocity model under the identical weighting block and the
    # identical held-out split, so the model change is judged on the same primary metric as
    # every tuning knob rather than on in-sample residuals alone.
    "v5model": cfg(velocity="v5"),
    # --- damping: the main inversion-stability knob (hypoDD's CND climbs into the thousands
    #     by the last iteration, so how hard the late stages are regularized matters) ---
    "damp_flat40": cfg(damp=[40, 40, 40, 40, 40]),
    "damp_high": cfg(damp=[60, 50, 40, 30, 25]),
    "damp_low": cfg(damp=[20, 15, 10, 5, 3]),
    "damp_gentle": cfg(damp=[40, 35, 28, 20, 12]),
    # flat40 beating both the tapered schedules suggests it is the constancy, not the level,
    # that stabilizes -- so sweep the level with the schedule held flat.
    "damp_flat20": cfg(damp=[20, 20, 20, 20, 20]),
    "damp_flat25": cfg(damp=[25, 25, 25, 25, 25]),
    "damp_flat30": cfg(damp=[30, 30, 30, 30, 30]),
    "damp_flat35": cfg(damp=[35, 35, 35, 35, 35]),
    "damp_flat55": cfg(damp=[55, 55, 55, 55, 55]),
    # --- iteration count: does the solution keep moving, or is it converged well before the
    #     32nd iteration? ---
    "iter_long": cfg(niter=[10, 10, 10, 10, 12]),
    "iter_short": cfg(niter=[4, 4, 4, 4, 5]),
    # --- CC residual cutoffs: absolute, in seconds, so they do NOT rescale themselves to the
    #     corrected model's much smaller residuals the way the MAD-relative CT cutoffs do ---
    "wrcc_tight": cfg(wrcc=[-999, -999, 0.03, 0.02, 0.01]),
    "wrcc_loose": cfg(wrcc=[-999, -999, 0.12, 0.08, 0.04]),
    # --- CT trimming: keep rejecting catalog outliers in the late stages instead of switching
    #     the cutoff off entirely ---
    "wrct_all": cfg(wrct=[10, 10, 3, 3, 3]),
    "wrct_tight": cfg(wrct=[6, 6, 2, -999, -999]),
    # --- more cross-correlation weight carried into the late stages ---
    "cc_stronger": cfg(wt_cc=[0.99, 0.50, 0.25, 0.10, 0.05]),

    # ---------------------------------------------------------------------------------
    # dt.cc quality filtering. All of these keep the BASE damping schedule, so if one of
    # them stops T2's 2.4 km cluster flip, the fix is attributable to the data, not to the
    # inversion settings being tuned around the problem.
    #
    # The motivating measurement: T2's |dt| is median 0.018 s and p99.9 0.19 s, but its
    # maximum is 232 s -- and of the 241 observations above 0.5 s, the MEDIAN CC coefficient
    # is 1.000. Those are spurious perfect correlations (cycle-skips onto a near-identical
    # waveform), and they enter the inversion at full weight. A coefficient threshold alone
    # cannot touch them; only a cap on the time shift can. Hence both axes, separately, so
    # their effects can be told apart.
    # ---------------------------------------------------------------------------------
    "dtmax_0.5": cfg(dtcc_max_s=0.5),     # drops only the physically impossible tail (0.03%)
    "dtmax_0.2": cfg(dtcc_max_s=0.2),     # ~p99.95 for T2
    "dtmax_0.1": cfg(dtcc_max_s=0.1),     # ~p99.9
    "dtmax_0.05": cfg(dtcc_max_s=0.05),   # aggressive: drops 5.3% of T2's CC data
    "ccmin_0.5": cfg(dtcc_min_w=0.5),     # keeps 78% of T2
    "ccmin_0.6": cfg(dtcc_min_w=0.6),     # keeps 53%
    "ccmin_0.7": cfg(dtcc_min_w=0.7),     # keeps 29%
    "clean_0.2_0.5": cfg(dtcc_max_s=0.2, dtcc_min_w=0.5),   # both caps together
    "clean_0.1_0.6": cfg(dtcc_max_s=0.1, dtcc_min_w=0.6),

    # T1-calibrated caps. The PRINCIPLE transfers from T2 -- cut the physically impossible
    # time shifts, which are identifiable by carrying coefficient 1.000 -- but the threshold
    # does not: T1's events are more spread out, so its genuine |dt| runs an order of
    # magnitude larger (median 60 ms, p90 217 ms, against T2's 18/42 ms). At T1 the
    # coefficient-1.000 signature only appears above ~1.0 s, and T2's 0.2 s cap would discard
    # 37.5% of T1's cross-correlation data rather than 0.03%.
    "dtmax_1.0": cfg(dtcc_max_s=1.0),          # T1: 59 obs, 0.05%, all coeff 1.000
    "clean_1.0_0.5": cfg(dtcc_max_s=1.0, dtcc_min_w=0.5),
    "clean_0.5_0.5": cfg(dtcc_max_s=0.5, dtcc_min_w=0.5),
    "clean_t1_flat40": cfg(dtcc_max_s=0.5, dtcc_min_w=0.5, damp=[40, 40, 40, 40, 40]),
    "clean_t1b_flat40": cfg(dtcc_max_s=1.0, dtcc_min_w=0.5, damp=[40, 40, 40, 40, 40]),
    # T1's counterpart to T2's chosen clean_ccstrong: the SAME CC weight taper carried into
    # the late stages, on T1's own 0.5 s cap. That cap was validated on the regenerated
    # (upsampled) dt.cc by a physical-possibility test -- an observation's |dt| cannot exceed
    # its pair separation divided by Vs. At 0.5 s the discarded population is 70.7%
    # impossible while 99.88% of the data is kept; T2's 0.2 s cap would instead discard
    # 107,134 T1 observations that are 80.2% LEGITIMATE, because T1's pairs sit a median
    # 0.75 km apart against T2's 0.23 km. Caveat: ~12% of the KEPT data also fails that test
    # at every cap, so the coefficient floor is still doing real work here.
    "t1_ccstrong": cfg(dtcc_max_s=0.5, dtcc_min_w=0.5,
                       wt_cc=[0.99, 0.50, 0.25, 0.10, 0.05]),

    # ---------------------------------------------------------------------------------
    # hypoDD controls that shape WHICH events are inverted together, rather than how the
    # data is weighted. T2 breaks into 118 clusters; requiring better-connected events, or
    # starting from the cluster centroid instead of the catalog locations, changes the
    # geometry the depth/origin-time trade-off has to work against.
    # ---------------------------------------------------------------------------------
    "minobs_strict": cfg(minobs_cc=8, minobs_ct=8),
    "minobs_loose": cfg(minobs_cc=1, minobs_ct=1),
    "istart1": cfg(istart=1),
    # The most promising combination, if the diagnosis is right: clean data + constant damping.
    "clean_flat40": cfg(dtcc_max_s=0.2, dtcc_min_w=0.5, damp=[40, 40, 40, 40, 40]),
    "clean_flat30": cfg(dtcc_max_s=0.2, dtcc_min_w=0.5, damp=[30, 30, 30, 30, 30]),

    # ---------------------------------------------------------------------------------
    # For the regenerated (upsampled) dt.cc only -- see --dtcc. That file carries 4.4x the
    # observations and 53% P where the original was 2.2%, so hypoDD's normal equations now
    # contain far more CC rows at the same per-observation wt_cc. Whether the base taper
    # still balances CC against CT is exactly the thing the original sweep could not have
    # tested, so re-open the CC weight axis on top of the winning filter.
    # ---------------------------------------------------------------------------------
    "clean_ccstrong": cfg(dtcc_max_s=0.2, dtcc_min_w=0.5,
                          wt_cc=[0.99, 0.50, 0.25, 0.10, 0.05]),
    "clean_ccweak": cfg(dtcc_max_s=0.2, dtcc_min_w=0.5,
                        wt_cc=[0.99, 0.20, 0.06, 0.02, 0.01]),
}


def tune_root(array):
    base = os.path.dirname(ARRAY_CONFIG[array]["working_dir"])
    suffix = "" if SRC_DTCC == "dt.cc" else "_" + SRC_DTCC.replace("dt.cc.", "").replace(".", "_")
    return os.path.join(base, "hypodd_vels1d", "tune" + suffix)


def src_dtcc(array):
    """The raw (unfiltered) cross-correlation file this sweep trains on."""
    return os.path.join(base_inputs(array), SRC_DTCC)


def base_inputs(array):
    return os.path.join(os.path.dirname(ARRAY_CONFIG[array]["working_dir"]),
                        "hypodd_vels1d", "input_files")


def fmt(v):
    if v == -999:
        return "-999"
    return f"{v:g}"


def model_lines(array, which="vels1d"):
    """The three hypoDD model lines for an array. `which="v5"` reproduces the superseded
    hand-assembled model EXACTLY as hypoDD ran it -- including the third line being read as
    the Vp/Vs ratio -- so it can be scored head to head with the reflection model on the same
    held-out data rather than only on in-sample residuals."""
    if which == "v5":
        layers = ARRAY_CONFIG[array]["velocity_layers"]
        return (" ".join(f"{d:g}" for d, _, _ in layers),
                " ".join(f"{vp:g}" for _, vp, _ in layers),
                " ".join(f"{listed:g}" for _, _, listed in layers))
    return build_hypodd_model_lines(build_layers(array))


def build_inp(array, config, idat=3, nset=None, isolv=2, iaq=1, minobs=None):
    """hypoDD.inp text. idat=3 uses both dt.cc and dt.ct; idat=1 is cross-correlation only
    (used by the scoring run). minobs=(0, 0) disables hypoDD's own event clustering, which a
    sparse held-out CC subset would otherwise collapse to a tiny connected component."""
    tops, vps, ratios = model_lines(array, config.get("velocity") or "vels1d")
    stages = []
    n = len(config["niter"]) if nset is None else nset
    for i in range(n):
        stages.append(
            f"{config['niter'][i]} {config['wt_cc'][i]} {config['wt_cc'][i]} "
            f"{fmt(config['wrcc'][i])} {fmt(config['wdcc'][i])} "
            f"{config['wt_ct'][i]} {config['wt_ct'][i]} "
            f"{fmt(config['wrct'][i])} {fmt(config['wdct'][i])}  {config['damp'][i]}"
        )
    return "\n".join([
        "hypoDD_2", "dt.cc", "dt.ct", "event.sel", "station.sel", "", "",
        "hypoDD.sta", "hypoDD.res", "hypoDD.src",
        f"{idat} 3 {HYPODD_DIST}",
        f"{minobs[0] if minobs else config['minobs_cc']} "
        f"{minobs[1] if minobs else config['minobs_ct']} 1 -999 -999",
        f"{config.get('istart', 2)} {isolv} {iaq} {len(stages)}",
        *stages,
        "1", tops, vps, ratios, "0", "",
    ])


def run_hypodd(work_dir, timeout=6 * 3600, output="hypoDD.reloc"):
    """Run hypoDD in work_dir, or return its existing output. Resumability matters here: a
    crash in the (cheap) scoring step must never cost the hours the inversion already spent."""
    existing = os.path.join(work_dir, output)
    if os.path.exists(existing) and os.path.getsize(existing) > 0:
        return existing
    with open(os.path.join(work_dir, "run.log"), "w") as log:
        proc = subprocess.run([HYPODD_BIN, "hypoDD.inp"], cwd=work_dir, stdout=log,
                              stderr=subprocess.STDOUT, timeout=timeout)
    reloc = os.path.join(work_dir, output)
    if proc.returncode != 0 or not os.path.exists(reloc) or os.path.getsize(reloc) == 0:
        return None
    if output != "hypoDD.reloc":
        return reloc
    # Fortran writes "****" for NaN date fields on numerically unstable events.
    with open(reloc) as f:
        lines = f.readlines()
    kept = [ln for ln in lines if "*" not in ln]
    if len(kept) != len(lines):
        with open(reloc, "w") as f:
            f.writelines(kept)
    return reloc


def link_inputs(src, dst, names):
    os.makedirs(dst, exist_ok=True)
    for name in names:
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(d):
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)


def split_dtcc(src, train_path, heldout_path, seed=CV_SEED, train_frac=CV_TRAIN_FRAC):
    """Split dt.cc 80/20 at the OBSERVATION-line level (not the pair level), so both halves
    span the same event pairs and the held-out set is not a different geometry."""
    rng = random.Random(seed)
    n_train = n_held = 0
    with open(src) as fin, open(train_path, "w") as ftr, open(heldout_path, "w") as fhe:
        header = None
        buf_tr, buf_he = [], []
        def flush():
            if header is None:
                return
            if buf_tr:
                ftr.write(header); ftr.writelines(buf_tr)
            if buf_he:
                fhe.write(header); fhe.writelines(buf_he)
        for line in fin:
            if line.startswith("#"):
                flush()
                header, buf_tr, buf_he = line, [], []
            elif rng.random() < train_frac:
                buf_tr.append(line); n_train += 1
            else:
                buf_he.append(line); n_held += 1
        flush()
    return n_train, n_held


def filter_dtcc(src, dst, max_dt_s=None, min_w=None):
    """Copy a dt.cc, dropping observations that fail the caps. Pair headers with no surviving
    observations are dropped too, since hypoDD would otherwise read an empty pair block.

    dt.cc line format: STA  DT[s]  WEIGHT  PHASE
    """
    n_in = n_out = 0
    with open(src) as fin, open(dst, "w") as fout:
        header, buf = None, []
        def flush():
            if header is not None and buf:
                fout.write(header)
                fout.writelines(buf)
        for line in fin:
            if line.startswith("#"):
                flush()
                header, buf = line, []
                continue
            n_in += 1
            f = line.split()
            try:
                dt, w = float(f[1]), float(f[2])
            except (IndexError, ValueError):
                continue
            if max_dt_s is not None and abs(dt) > max_dt_s:
                continue
            if min_w is not None and w < min_w:
                continue
            buf.append(line)
            n_out += 1
        flush()
    return n_in, n_out


def rewrite_event_sel(base_sel, reloc_path, out_path):
    """event.sel with each event's coordinates replaced by its relocated ones, falling back to
    the original coordinates for events the train run did not relocate. NaN coordinates are
    also treated as unrelocated: a single NaN poisons hypoDD's station-distance reference and
    silently produces an empty hypoDD.res, rather than failing locally."""
    ev = pd.read_csv(base_sel, sep=r"\s+", header=None, names=EVENT_SEL_COLS)
    rel = pd.read_csv(reloc_path, sep=r"\s+", header=None, names=RELOC_COLS)
    rel = rel[np.isfinite(rel[["lat", "lon", "depth"]]).all(axis=1)]
    lut = rel.set_index("id")[["lat", "lon", "depth"]]
    joined = ev.set_index("id").join(lut, rsuffix="_new")
    for c in ["lat", "lon", "depth"]:
        joined[c] = joined[f"{c}_new"].combine_first(joined[c])
    joined = joined.drop(columns=[f"{c}_new" for c in ["lat", "lon", "depth"]]).reset_index()
    joined = joined[EVENT_SEL_COLS]
    joined.to_csv(out_path, sep=" ", header=False, index=False, float_format="%.6f")
    return int(lut.index.isin(ev["id"]).sum())


def score_heldout(res_path):
    """Score predicted vs observed held-out differential times.

    hypoDD.res is: STA  DT_obs[s]  C1  C2  IDX  QUAL  RES[ms]  WT  OFFS[m]; calc = obs - res/1000.
    Rows hypoDD down-weighted to zero carry no information and are excluded.

    Plain RMS is reported but is NOT the metric to rank on. The dt.cc population has a heavy
    tail -- median |dt| is ~60 ms while the RMS of the same column is ~150-900 ms -- so a few
    extreme differential times dominate the sum of squares. Which of those monsters happen to
    land in a given 20% held-out draw then swings the RMS far more than any config does:
    across two split seeds the identical config scored 56 ms and 121 ms. The robust statistics
    below (median |residual|, and RMS after trimming the worst 1%) are what survive that, and
    are what the ranking uses.
    """
    cols = ["sta", "dt", "c1", "c2", "idx", "qual", "res_ms", "wt", "offs"]
    res = pd.read_csv(res_path, sep=r"\s+", header=None, names=cols)
    # A residual too big for hypoDD's f12.6 field prints as "************" -- one such row
    # (an observed dt of -20.6 s across a 12 km pair) turns the whole column to strings and
    # takes the scoring down with it. Coerce and drop; those rows are garbage dt.cc entries.
    for c in ["dt", "res_ms", "wt"]:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    res = res[np.isfinite(res[["dt", "res_ms", "wt"]]).all(axis=1)]
    res = res[res["wt"] > 1e-6]
    if len(res) < 100:
        return None
    r = res["res_ms"].values
    keep = np.abs(r) <= np.percentile(np.abs(r), 99.0)
    calc = res["dt"] - res["res_ms"] / 1000.0
    return {
        "n_heldout_scored": int(len(res)),
        "heldout_medabs_ms": float(np.median(np.abs(r))),
        "heldout_rms_trim_ms": float(np.sqrt(np.mean(r[keep] ** 2))),
        "heldout_rms_ms": float(np.sqrt(np.mean(r ** 2))),
        "heldout_corr": float(np.corrcoef(res["dt"], calc)[0, 1]),
    }


def rescore(array):
    """Recompute held-out scores for every finished config from its saved hypoDD.res, without
    re-running hypoDD -- so a change to the scoring statistic costs nothing."""
    import glob as _glob
    for result_path in sorted(_glob.glob(os.path.join(tune_root(array), "*", "result.json"))):
        root = os.path.dirname(result_path)
        res_path = os.path.join(root, "eval", "hypoDD.res")
        if not (os.path.exists(res_path) and os.path.getsize(res_path) > 0):
            continue
        scored = score_heldout(res_path)
        if not scored:
            continue
        r = json.load(open(result_path))
        r.update(scored)
        prod_reloc = os.path.join(root, "prod", "hypoDD.reloc")
        train_reloc = os.path.join(root, "train", "hypoDD.reloc")
        if os.path.exists(prod_reloc) and os.path.exists(train_reloc):
            r.update(displacement(prod_reloc, train_reloc))
            r.update(motion_from_initial(os.path.join(root, "prod")))
        if r.get("n_cc_heldout"):
            r["heldout_recovery_pct"] = 100.0 * scored["n_heldout_scored"] / r["n_cc_heldout"]
        json.dump(r, open(result_path, "w"), indent=2)
    print(f"rescored {array}")


# hypoDD.f writes the per-iteration summary row as
#   i2, a3, 3(1x,i3), i5, f6.1, i5, f6.1, i6, 4i5, i5, i4  [, i5 CND]
# Splitting on whitespace is not safe: the leading IT/EV pair merges when the a3 field is
# blank, and AQ/CND merge once CND reaches 5 digits ("   013695"), each silently shifting
# every positional index. Slice by the declared widths instead.
_ROW_FIELDS = [
    ("iter", 2), ("set", 3), ("ev_pct", 4), ("ct_pct", 4), ("cc_pct", 4),
    ("rmsct_ms", 5), ("rmsct_dpct", 6), ("rmscc_ms", 5), ("rmscc_dpct", 6),
    ("rmsst_ms", 6), ("dx_m", 5), ("dy_m", 5), ("dz_m", 5), ("dt_ms", 5),
    ("os_m", 5), ("aq", 4), ("cnd", 5),
]


def parse_iteration_row(line):
    out, pos = {}, 0
    for name, width in _ROW_FIELDS:
        chunk = line[pos:pos + width].strip()
        pos += width
        if name == "set":
            continue
        try:
            out[name] = float(chunk) if chunk else float("nan")
        except ValueError:  # a field that overflowed its width
            out[name] = float("nan")
    return out


def parse_log(log_path, cluster=1):
    """Final-iteration diagnostics for hypoDD's largest cluster: retained-data percentages,
    RMS residuals, step sizes and condition number."""
    row, in_cluster, countdown = None, False, 0
    keep = ["ev_pct", "ct_pct", "cc_pct", "rmsct_ms", "rmscc_ms", "rmsst_ms",
            "dx_m", "dy_m", "dz_m", "cnd"]
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("RELOCATION OF CLUSTER:"):
                field = line.split(":")[1].split()[0]
                in_cluster = field.isdigit() and int(field) == cluster
                continue
            if not in_cluster:
                continue
            if line.lstrip().startswith("IT   EV"):
                countdown = 2
                continue
            if countdown:
                countdown -= 1
                if countdown == 0:
                    parsed = parse_iteration_row(line.rstrip("\n"))
                    if parsed.get("iter") == parsed.get("iter"):  # not NaN
                        row = {k: parsed[k] for k in keep if k in parsed}
    return row or {}


def displacement(reloc_a, reloc_b):
    """Displacement between two relocations of the same events (a = full data, b = 80% CC).

    The median alone is not enough. At T2 the median vertical shift for `base` is 58 m while
    HALF the catalog -- hypoDD's cluster 1 -- moves 2.4 km, from a median depth of 1.94 km to
    4.34 km. With just under 50% of events moving, the median lands in the stationary half and
    reports calm. So this also returns the 90th percentile, and the shift in the MEDIAN DEPTH
    of the largest cluster, which is what catches a whole population relocating en masse.

    Watching ONLY cluster 1 is not enough either, and that is the second lesson. On the
    upsampled dt.cc, `clean_ccstrong` scored cluster1_depth_shift = 0 m and led the held-out
    table -- while hypoDD's cluster 2 (682 events, 21% of the catalog) moved 501 m shallower,
    3-4x further than any other config moved the same events. A metric that looks at one
    cluster reports calm for a config that is visibly unstable elsewhere. So every cluster
    large enough to be meaningful is measured, and the ranking statistic is the WORST of them.
    """
    import pyproj
    a = pd.read_csv(reloc_a, sep=r"\s+", header=None, names=RELOC_COLS)
    b = pd.read_csv(reloc_b, sep=r"\s+", header=None, names=RELOC_COLS)
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if not len(m):
        return {}
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    xa, ya = to_ps.transform(m["lon_a"].values, m["lat_a"].values)
    xb, yb = to_ps.transform(m["lon_b"].values, m["lat_b"].values)
    dh = np.hypot(xb - xa, yb - ya)
    dz = np.abs(m["depth_b"].values - m["depth_a"].values) * 1000.0
    out = {
        "n_common": int(len(m)),
        "stability_dh_med_m": float(np.median(dh)),
        "stability_dz_med_m": float(np.median(dz)),
        "stability_dz_p90_m": float(np.percentile(dz, 90)),
    }
    # Per-cluster median-depth shift. Cluster ids are assigned independently by each run and
    # are NOT stable between them, so -- as the original cluster-1 test did -- a cluster is
    # the set of events that carry the same cid in BOTH runs. Events hypoDD reassigned are
    # simply not counted rather than being compared across two different populations.
    shifts = {}
    for cid in sorted(set(m["cid_a"]) & set(m["cid_b"])):
        c = m[(m["cid_a"] == cid) & (m["cid_b"] == cid)]
        if len(c) < MIN_CLUSTER_FOR_SHIFT:
            continue
        shifts[int(cid)] = {
            "n": int(len(c)),
            "shift_m": float(abs(np.median(c["depth_b"].values)
                                 - np.median(c["depth_a"].values)) * 1000.0),
        }

    if shifts:
        out["cluster_depth_shifts"] = shifts
        worst = max(shifts.items(), key=lambda kv: kv[1]["shift_m"])
        out["worst_cluster_id"] = worst[0]
        out["worst_cluster_n"] = worst[1]["n"]
        out["worst_cluster_depth_shift_m"] = worst[1]["shift_m"]
        out["n_clusters_measured"] = len(shifts)

    # kept under its original name so earlier result.json files stay comparable
    if 1 in shifts:
        out["n_cluster1"] = shifts[1]["n"]
        out["cluster1_depth_shift_m"] = shifts[1]["shift_m"]
    return out


def motion_from_initial(prod_dir):
    """How far the production run actually MOVED events from their input locations.

    This is the control on the stability metric. Damping harder trivially makes a solution
    more reproducible -- in the limit of infinite damping nothing moves at all and the two
    runs agree perfectly -- so "stable" only means something alongside evidence that the
    inversion is still doing real work. hypoDD.loc holds the initial locations of exactly the
    events that ended up in hypoDD.reloc, so the two are directly comparable.
    """
    import pyproj
    loc_path = os.path.join(prod_dir, "hypoDD.loc")
    reloc_path = os.path.join(prod_dir, "hypoDD.reloc")
    if not (os.path.exists(loc_path) and os.path.exists(reloc_path)):
        return {}
    loc = pd.read_csv(loc_path, sep=r"\s+", header=None, names=RELOC_COLS)
    rel = pd.read_csv(reloc_path, sep=r"\s+", header=None, names=RELOC_COLS)
    m = loc.merge(rel, on="id", suffixes=("_i", "_f"))
    if not len(m):
        return {}
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    xi, yi = to_ps.transform(m["lon_i"].values, m["lat_i"].values)
    xf, yf = to_ps.transform(m["lon_f"].values, m["lat_f"].values)
    return {
        "moved_dh_med_m": float(np.median(np.hypot(xf - xi, yf - yi))),
        "moved_dz_med_m": float(np.median(np.abs(m["depth_f"].values - m["depth_i"].values) * 1000.0)),
    }


def run_config(array, name, force=False, seed=CV_SEED):
    """A non-default `seed` reruns the whole config against a different 80/20 CC split, into
    its own directory -- the check that a stability or accuracy ranking is a property of the
    config and not of one lucky split."""
    config = CONFIGS[name]
    src = base_inputs(array)
    root = os.path.join(tune_root(array), name if seed == CV_SEED else f"{name}__seed{seed}")
    result_path = os.path.join(root, "result.json")
    if os.path.exists(result_path) and not force:
        print(f"[{array}/{os.path.basename(root)}] already done")
        return json.load(open(result_path))

    result = {"array": array, "config": name if seed == CV_SEED else f"{name}__seed{seed}", "seed": seed}

    # 1. Production run: the full config on all the data. This is the candidate relocation.
    prod = os.path.join(root, "prod")
    link_inputs(src, prod, ["dt.ct", "event.sel", "station.sel"])
    prod_cc = os.path.join(prod, "dt.cc")
    if not os.path.exists(prod_cc):
        n_in, n_out = filter_dtcc(src_dtcc(array), prod_cc,
                                  config["dtcc_max_s"], config["dtcc_min_w"])
        result["n_cc_total"], result["n_cc_kept"] = n_in, n_out
        result["cc_kept_pct"] = round(100.0 * n_out / max(n_in, 1), 2)
    # The production run uses ALL the CC data, so it does not depend on the CV split seed --
    # a seed variant reuses the default-seed run rather than spending another inversion on a
    # bit-identical result.
    if seed != CV_SEED:
        shared = os.path.join(tune_root(array), name, "prod")
        for fn in ["hypoDD.reloc", "hypoDD.loc", "hypoDD.log"]:
            s_path, d_path = os.path.join(shared, fn), os.path.join(prod, fn)
            if os.path.exists(s_path) and not os.path.exists(d_path):
                try:
                    os.link(s_path, d_path)
                except OSError:
                    shutil.copy2(s_path, d_path)
    with open(os.path.join(prod, "hypoDD.inp"), "w") as f:
        f.write(build_inp(array, config))
    prod_reloc = run_hypodd(prod)
    if prod_reloc is None:
        result["error"] = "production run produced no hypoDD.reloc"
        json.dump(result, open(result_path, "w"), indent=2)
        return result
    result["n_relocated"] = sum(1 for _ in open(prod_reloc))
    result.update(parse_log(os.path.join(prod, "hypoDD.log")))
    result.update(motion_from_initial(prod))

    # 2. Train run: same config, 80% of the cross-correlation data.
    train = os.path.join(root, "train")
    link_inputs(src, train, ["dt.ct", "event.sel", "station.sel"])
    # Split the FULL dt.cc, then apply this config's filter to the train half only. The
    # held-out half is filtered by one fixed, config-independent rule (EVAL_MAX_DT_S), so
    # every config is scored on exactly the same observations no matter what it trained on --
    # otherwise a config that discards data would be graded on an easier exam.
    raw_train = os.path.join(root, "train_raw.cc")
    raw_held = os.path.join(root, "heldout_raw.cc")
    if not (os.path.exists(raw_train) and os.path.exists(raw_held)):
        split_dtcc(src_dtcc(array), raw_train, raw_held, seed=seed)
    train_cc = os.path.join(train, "dt.cc")
    if not os.path.exists(train_cc):
        _, n_tr = filter_dtcc(raw_train, train_cc, config["dtcc_max_s"], config["dtcc_min_w"])
    else:
        n_tr = sum(1 for ln in open(train_cc) if not ln.startswith("#"))
    heldout_path = os.path.join(root, "heldout.cc")
    _, n_he = filter_dtcc(raw_held, heldout_path, EVAL_MAX_DT_S, None)
    result["n_cc_train"], result["n_cc_heldout"] = n_tr, n_he
    with open(os.path.join(train, "hypoDD.inp"), "w") as f:
        f.write(build_inp(array, config))
    train_reloc = run_hypodd(train)
    if train_reloc is None:
        result["error"] = "train run produced no hypoDD.reloc"
        json.dump(result, open(result_path, "w"), indent=2)
        return result

    # 3. Stability: how far the picture moved when 20% of the CC data was withheld.
    result.update(displacement(prod_reloc, train_reloc))

    # 4. Score the train locations against the held-out 20%.
    ev = os.path.join(root, "eval")
    link_inputs(src, ev, ["dt.ct", "station.sel"])
    eval_cc = os.path.join(ev, "dt.cc")
    stale = (os.path.exists(eval_cc)
             and not filecmp.cmp(heldout_path, eval_cc, shallow=False))
    shutil.copy2(heldout_path, eval_cc)
    if stale:
        # The evaluation set changed, so any saved scoring run answers the wrong question.
        for fn in ["hypoDD.res", "hypoDD.reloc"]:
            fp = os.path.join(ev, fn)
            if os.path.exists(fp):
                os.remove(fp)
    result["n_eval_events_from_train"] = rewrite_event_sel(
        os.path.join(src, "event.sel"), train_reloc, os.path.join(ev, "event.sel"))
    with open(os.path.join(ev, "hypoDD.inp"), "w") as f:
        # idat=1: cross-correlation only. minobs=(0,0): no clustering, keep every event --
        # with clustering on, a sparse 20% CC subset collapses to a handful of events.
        # One stage, one iteration, all weight, no trimming: report residuals as they are.
        f.write(build_inp(array, cfg(velocity=config.get("velocity", "vels1d"),
                                     niter=[1], wt_cc=[1.0], wrcc=[-999], wdcc=[-999],
                                     wt_ct=[0.0], wrct=[-999], wdct=[-999], damp=[BASE["damp"][0]]),
                          idat=1, isolv=2, iaq=0, minobs=(0, 0)))
    run_hypodd(ev, output="hypoDD.res")
    res_path = os.path.join(ev, "hypoDD.res")
    if os.path.exists(res_path) and os.path.getsize(res_path) > 0:
        scored = score_heldout(res_path)
        if scored:
            result.update(scored)
            result["heldout_recovery_pct"] = 100.0 * scored["n_heldout_scored"] / max(n_he, 1)
        else:
            result["error"] = "held-out scoring recovered too few observations"
    else:
        result["error"] = "eval run produced no hypoDD.res"

    json.dump(result, open(result_path, "w"), indent=2)
    print(f"[{array}/{result['config']}] done: {json.dumps({k: v for k, v in result.items() if k not in ('array', 'config')})}")
    return result


def promote(array, name):
    """Install one swept config's production relocation as the array's AUTHORITATIVE one.

    Everything downstream reads `<array>_v5/hypodd_vels1d/output_files/hypoDD.reloc` through
    catalog_paths.py, so promotion is a copy plus a provenance record -- not a re-run. The
    previous contents are moved to `output_files_prev/` rather than deleted, so figures already
    made against them stay reproducible.

    The config's FILTERED dt.cc is installed as `input_files/dt.cc.authoritative`. The
    unfiltered `input_files/dt.cc` is deliberately left alone: it is the raw cross-correlation
    product, and other analyses (CC refinement, template stacking) legitimately want every
    observation regardless of what the relocation filtered out. Because of that split, the
    promoted hypoDD.inp is rewritten to name `dt.cc.authoritative` rather than `dt.cc`, so the
    promoted input set actually reproduces the promoted output.
    """
    src_dir = os.path.join(tune_root(array), name, "prod")
    dst_root = os.path.dirname(tune_root(array))
    out_dir = os.path.join(dst_root, "output_files")
    if not os.path.exists(os.path.join(src_dir, "hypoDD.reloc")):
        raise SystemExit(f"{name} has no production relocation to promote")

    if os.path.exists(out_dir):
        prev = os.path.join(dst_root, "output_files_prev")
        if os.path.exists(prev):
            shutil.rmtree(prev)
        shutil.move(out_dir, prev)
        print(f"[{array}] previous relocation preserved at {prev}")
    os.makedirs(out_dir, exist_ok=True)
    for fn in ["hypoDD.reloc", "hypoDD.loc", "hypoDD.src", "hypoDD.sta", "hypoDD.res"]:
        fp = os.path.join(src_dir, fn)
        if os.path.exists(fp):
            shutil.copy2(fp, os.path.join(out_dir, fn))
    # hypoDD.inp names its input files RELATIVELY, and line 2 is the CC file. Inside
    # tune/<name>/prod/ the local "dt.cc" IS that config's filtered copy, so the name is
    # correct there -- but in the promoted input_files/ "dt.cc" is the unfiltered raw
    # product we deliberately keep (see this docstring). Copying the .inp verbatim therefore
    # leaves behind an input set that re-runs on UNFILTERED data and silently fails to
    # reproduce the very relocation it sits next to. Repoint line 2 at the filtered copy.
    inp_lines = open(os.path.join(src_dir, "hypoDD.inp")).read().split("\n")
    for i, line in enumerate(inp_lines[:6]):
        if line.strip() == "dt.cc":
            inp_lines[i] = "dt.cc.authoritative"
            break
    else:
        raise SystemExit("hypoDD.inp has no 'dt.cc' line to repoint -- format changed?")
    with open(os.path.join(dst_root, "input_files", "hypoDD.inp"), "w") as f:
        f.write("\n".join(inp_lines))
    log = os.path.join(src_dir, "hypoDD.log")
    if os.path.exists(log):
        shutil.copy2(log, os.path.join(dst_root, "hypoDD_log.txt"))
    cc = os.path.join(src_dir, "dt.cc")
    if os.path.exists(cc):
        shutil.copy2(cc, os.path.join(dst_root, "input_files", "dt.cc.authoritative"))

    cfg = CONFIGS[name]
    res = json.load(open(os.path.join(tune_root(array), name, "result.json")))
    n = sum(1 for _ in open(os.path.join(out_dir, "hypoDD.reloc")))
    lines = [
        f"# {array} authoritative relocation",
        "",
        f"Config `{name}`, promoted from `tune/{name}/prod/` by `hypodd_tune.py promote`.",
        "",
        f"- events relocated: **{n}**",
        f"- velocity model: {cfg['velocity']} (reflection-seismology profile)",
        f"- dt.cc filter: |dt| <= {cfg['dtcc_max_s']} s, coefficient >= {cfg['dtcc_min_w']} "
        f"(kept {res.get('cc_kept_pct')}% of observations)",
        f"- damping schedule: {cfg['damp']}",
        f"- worst per-cluster depth shift under a 20% CC withhold: "
        f"{res.get('worst_cluster_depth_shift_m')} m "
        f"(cluster {res.get('worst_cluster_id')}, n={res.get('worst_cluster_n')}, "
        f"of {res.get('n_clusters_measured')} clusters >= {MIN_CLUSTER_FOR_SHIFT} events, seed 42)",
        f"- cluster-1 depth shift: {res.get('cluster1_depth_shift_m')} m (seed 42)",
        f"- held-out CV median |residual|: {res.get('heldout_medabs_ms')} ms",
        "",
        f"- cross-correlation source: `input_files/{SRC_DTCC}`",
        "",
        "Filtered cross-correlation input: `input_files/dt.cc.authoritative`.",
        "`input_files/dt.cc` remains the unfiltered original.",
        "",
    ]
    with open(os.path.join(dst_root, "PROVENANCE.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[{array}] promoted {name}: {n} events -> {out_dir}")
    return out_dir


def cluster_consensus(array, configs=None):
    """Per-cluster disagreement BETWEEN configs -- the check the CV stability metric cannot make.

    `displacement()` asks whether one config reproduces itself when 20% of the CC data is
    withheld. A config can pass that and still be wrong: on the upsampled dt.cc,
    `clean_ccstrong` puts hypoDD's cluster 2 at 0.836 km in its production run and 0.857 km in
    its held-out run -- self-consistent to 21 m, a clean pass -- while every other config puts
    the SAME 667 events between 1.159 and 1.316 km. Internal reproducibility says nothing
    about agreement with the other solutions; a config that is confidently and consistently
    off is exactly what it cannot see.

    So: define cluster membership ONCE by event id (cluster ids are assigned per-run and are
    not comparable across configs), take each config's median depth for that fixed event set,
    and score every config by how far it sits from the across-config median. Returns
    {config: {"consensus_dev_max_m", "consensus_dev_cluster", "consensus_dev_by_cluster"}}.

    This is a DISPERSION statistic, not a truth test. It says a config is the odd one out, not
    that the majority is right -- if a real improvement moves one cluster, this flags it too.
    Read it as "explain this before promoting", never as an automatic disqualification.
    """
    import glob as _glob
    root = tune_root(array)
    if configs is None:
        configs = sorted(
            os.path.basename(os.path.dirname(os.path.dirname(p)))
            for p in _glob.glob(os.path.join(root, "*", "prod", "hypoDD.reloc"))
        )
    configs = [c for c in configs if "__seed" not in c]  # seed variants reuse prod
    if len(configs) < 3:
        return {}

    locs = {}
    for c in configs:
        d = pd.read_csv(os.path.join(root, c, "prod", "hypoDD.reloc"),
                        sep=r"\s+", header=None, names=RELOC_COLS).set_index("id")
        locs[c] = d[~d.index.duplicated()]

    common = set.intersection(*(set(d.index) for d in locs.values()))
    if len(common) < MIN_CLUSTER_FOR_SHIFT:
        return {}

    # membership from the config with the most relocated events, restricted to the common set
    ref = max(configs, key=lambda c: len(locs[c]))
    rd = locs[ref].loc[sorted(common)]
    groups = {int(cid): g.index for cid, g in rd.groupby("cid")
              if len(g) >= MIN_CLUSTER_FOR_SHIFT}
    if not groups:
        return {}

    med = {cid: {c: float(locs[c].loc[ids, "depth"].median()) for c in configs}
           for cid, ids in groups.items()}
    consensus = {cid: float(np.median(list(v.values()))) for cid, v in med.items()}

    out = {}
    for c in configs:
        devs = {cid: abs(med[cid][c] - consensus[cid]) * 1000.0 for cid in groups}
        worst = max(devs, key=devs.get)
        out[c] = {
            "consensus_dev_max_m": devs[worst],
            "consensus_dev_cluster": worst,
            "consensus_ref_config": ref,
            "consensus_dev_by_cluster": {str(k): round(v, 1) for k, v in devs.items()},
        }
    return out


def report(array):
    import glob as _glob
    rows = []
    for p in sorted(_glob.glob(os.path.join(tune_root(array), "*", "result.json"))):
        r = json.load(open(p))
        if "moved_dh_med_m" not in r:  # backfill without re-running hypoDD
            r.update(motion_from_initial(os.path.join(os.path.dirname(p), "prod")))
            json.dump(r, open(p, "w"), indent=2)
        rows.append(r)
    if not rows:
        print(f"no results yet for {array}")
        return

    # cross-config agreement, written back into each result.json so it is not report-only
    cons = cluster_consensus(array)
    for r in rows:
        if r["config"] in cons:
            r.update(cons[r["config"]])
            fp = os.path.join(tune_root(array), r["config"], "result.json")
            if os.path.exists(fp):
                json.dump(r, open(fp, "w"), indent=2)

    df = pd.DataFrame(rows).set_index("config")
    cols = ["heldout_medabs_ms", "heldout_rms_trim_ms", "heldout_rms_ms", "heldout_corr",
            "heldout_recovery_pct",
            "stability_dh_med_m", "stability_dz_med_m", "stability_dz_p90_m",
            "worst_cluster_depth_shift_m", "worst_cluster_id", "cluster1_depth_shift_m",
            "consensus_dev_max_m", "consensus_dev_cluster",
            "moved_dh_med_m", "moved_dz_med_m", "cc_kept_pct",
            "n_relocated", "ct_pct", "cc_pct", "rmsct_ms", "rmscc_ms", "rmsst_ms",
            "dx_m", "dy_m", "dz_m", "cnd"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("heldout_medabs_ms")
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(f"\n=== {array}: hypoDD iteration/inversion sweep on the vels1d model ===")
    print("accuracy = held-out CV median |residual| (lower better; plain RMS is shown too but")
    print("is dominated by a handful of heavy-tail differential times and is NOT rankable); stability = median shift when 20% of the")
    print("CC data is withheld (lower better); ct_pct/cc_pct = data surviving hypoDD's own")
    print("trimming (higher better -- a low RMS bought by discarding data is not an improvement);")
    print("moved_* = displacement from the INPUT locations, the control on stability: damping")
    print("harder makes any solution more reproducible, so 'stable' only counts if the")
    print("inversion is still moving events as far as the looser configs do;")
    print("consensus_dev_max_m = furthest any cluster's median depth sits from the")
    print("across-config median -- catches a config that reproduces itself but disagrees")
    print("with every other solution, which the withhold test passes. It is a dispersion")
    print("statistic: a large value means explain it before promoting, not automatic reject")
    print(df.to_string(float_format=lambda v: f"{v:.2f}"))
    out = os.path.join(tune_root(array), "sweep_summary.csv")
    df.to_csv(out)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ["run", "sweep", "report", "rescore", "promote"]:
        p = sub.add_parser(cmd)
        p.add_argument("--array", required=True, choices=sorted(ARRAY_CONFIG))
        if cmd in ("run", "promote"):
            p.add_argument("--config", required=True, choices=sorted(CONFIGS))
        if cmd == "run":
            p.add_argument("--force", action="store_true")
        if cmd == "sweep":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--configs", nargs="*", default=None)
            p.add_argument("--force", action="store_true",
                           help="recompute metrics even for configs that already have a "
                                "result.json; hypoDD stages whose output is on disk are "
                                "still reused, so this is cheap")
        if cmd in ("run", "sweep"):
            p.add_argument("--seed", type=int, default=CV_SEED)
        p.add_argument("--dtcc", default="dt.cc",
                       help="cross-correlation file in input_files/ to sweep on. Anything but "
                            "the default writes to (and reads from) tune_<name>/ instead of "
                            "tune/, since a different dt.cc is a different dataset.")
    args = ap.parse_args()

    global SRC_DTCC
    SRC_DTCC = args.dtcc
    if not os.path.exists(src_dtcc(args.array)):
        raise SystemExit(f"no such cross-correlation file: {src_dtcc(args.array)}")

    if args.cmd == "promote":
        promote(args.array, args.config)
    elif args.cmd == "run":
        run_config(args.array, args.config, force=args.force, seed=args.seed)
    elif args.cmd == "sweep":
        names = args.configs or list(CONFIGS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_config, args.array, n, force=args.force,
                                   seed=args.seed): n for n in names}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # keep the rest of the sweep alive
                    print(f"[{args.array}/{futures[fut]}] FAILED: {exc}", file=sys.stderr)
        report(args.array)
    elif args.cmd == "rescore":
        rescore(args.array)
        report(args.array)
    else:
        report(args.array)


if __name__ == "__main__":
    main()
