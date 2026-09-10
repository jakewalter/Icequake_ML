#!/usr/bin/env python3
"""Test 3 of PLAN_depth_resolvability.md, analysis half: migrate the stack semblance over
trial source depth using the ice-base reflection moveout.

Why this and not per-station peak picking. The first pass at this test picked the strongest
coherent arrival in a fixed delay window per station, and every "hit" landed 0.12-0.19 s after
that station's MEASURED composite S -- i.e. it found the S coda, which is coherent, large, and
moves out with distance exactly as the window boundaries did. Peak picking cannot separate a
depth phase from S coda at a single station. Moveout can: across the array the two depend on
distance in opposite ways, so the test has to use all stations at once.

Also, the plan's stated ghost -- an up-going ray reflecting off the free surface, delayed by
2z*cos(i)/Vp -- is the teleseismic pP geometry and does not reach a receiver sitting ON that
same free surface. The local observables are instead:

  * the ice-base reflection: down from the source, reflected at the bed, up to the receiver.
    Delay -> 0 as the source approaches the bed, and DECREASES with distance.
  * the surface-then-bed multiple: up from the source, reflected at the free surface, down to
    the bed, back up. Delay grows with depth instead.

Both are computed here by two-point ray tracing through the 5 m reflection profile itself
(down and up legs traced separately, so the firn gradient is honoured on every pass), never by
a homogeneous-halfspace image formula. The direct P is traced in the SAME model, so the
predicted delay is a difference of two consistently-computed traveltimes.

Migration. For each trial depth z, the predicted delay is evaluated at every station's own
distance and the stack semblance is sampled there; a real reflector shows up as a peak in the
summed semblance at the true depth, because only the correct z lines the stations' arrivals up
with each other. Stations whose predicted delay falls in the S coda band, or too close to P to
be separable, are dropped AT THAT z -- so the curve is renormalised by the stations actually
contributing, and coverage is reported alongside it.

The null matters as much as the curve. Semblance is non-negative and the traces are not white,
so ANY moveout curve returns a positive score; the question is whether the real one beats
arbitrary delays. The null draws random per-station delays from each station's own allowed
range, preserving its semblance distribution exactly, and reports the 95th/99th percentile of
the resulting peak score. A migration peak that does not clear that line is not evidence.

Usage:
    python full_catalog_pipeline/depth_phase_migration.py --array T2 --clusters 0 1 2 6 7
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
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import catalog_paths
import vels1d_model

# S exclusion: S itself plus the coda that follows it, in seconds around the measured S-P.
S_MASK_PRE, S_MASK_POST = 0.15, 0.50
MIN_STATIONS = 3
# Separability floor. A bed reflection arriving while the direct P is still ringing cannot be
# told apart from P's own coda -- and since semblance stays near its peak through that coda,
# a fixed small floor makes the migration climb toward the bed for free and invent a peak
# wherever the floor happens to cut it. The floor is therefore MEASURED per station from its
# own stack (see coda_floor) rather than assumed.
CODA_ABS, CODA_REL = 0.15, 3.0   # coda ends below max(0.15, 3x the late-record background)
CODA_HOLD_S = 0.05               # ...and must stay below for this long, not just dip
CODA_MARGIN_S = 0.03
Z_MIN, Z_MAX, Z_STEP = 0.20, 2.60, 0.02
N_NULL = 2000
RNG_SEED = 20260828

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


class Profile:
    """The 5 m reflection profile as a stack of thin homogeneous layers, in km and km/s."""

    def __init__(self, site):
        depth_m, vp_ms, _ = vels1d_model.shifted_profile(site)
        self.z = np.asarray(depth_m, float) / 1000.0
        self.v = np.asarray(vp_ms, float) / 1000.0

    def legs(self, za, zb):
        """Thicknesses and velocities of the layers spanned between depths za and zb (km)."""
        za, zb = min(za, zb), max(za, zb)
        edges = np.clip(self.z, za, zb)
        dz = np.diff(edges)
        keep = dz > 0
        return dz[keep], self.v[:-1][keep]

    def vmax(self, *segments):
        vs = [v for a, b in segments for v in (self.legs(a, b)[1],) if len(v)]
        return max(v.max() for v in vs) if vs else np.nan

    def travel(self, segments, r):
        """Two-point traveltime (s) of a ray made of `segments` [(za, zb), ...] whose
        horizontal offsets sum to `r` (km). Returns NaN if the geometry has no solution."""
        parts = [self.legs(a, b) for a, b in segments]
        parts = [(dz, v) for dz, v in parts if len(dz)]
        if not parts:
            return np.nan
        vmax = max(v.max() for _, v in parts)
        p_hi = (1.0 - 1e-9) / vmax

        def offset_time(p):
            x = t = 0.0
            for dz, v in parts:
                eta = np.sqrt(np.maximum(1.0 - (p * v) ** 2, 1e-12))
                x += float(np.sum(dz * p * v / eta))
                t += float(np.sum(dz / (v * eta)))
            return x, t

        if offset_time(p_hi)[0] < r:      # even a grazing ray cannot reach this far
            return np.nan
        lo, hi = 0.0, p_hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if offset_time(mid)[0] < r:
                lo = mid
            else:
                hi = mid
        return offset_time(0.5 * (lo + hi))[1]

    def direct_p(self, z, r):
        return self.travel([(0.0, z)], r)

    def bed_reflection(self, z, r, h):
        """Down from the source to the bed, then up to the surface."""
        return self.travel([(z, h), (0.0, h)], r)

    def surface_bed_multiple(self, z, r, h):
        """Up to the free surface, down to the bed, back up to the receiver."""
        return self.travel([(0.0, z), (0.0, h), (0.0, h)], r)


def load_stacks(work, array, cluster, component="Z"):
    """{station: dict(t, semb, n, dist_km, depth_hypodd)} from cluster_depth_phase_stack.py."""
    out = {}
    pat = os.path.join(work, "depth_phase",
                       f"{array.lower()}_c{cluster}_*_{component}_long.npz")
    for path in sorted(glob.glob(pat)):
        m = re.search(rf"_c{cluster}_(\w+)_{component}_long\.npz$", os.path.basename(path))
        if not m:
            continue
        d = np.load(path)
        out[m.group(1)] = dict(t=d["t"], semb=d["semb"], n=int(d["n"]),
                               dist_km=float(d["dist_km"]),
                               depth_hypodd=float(d["depth_hypodd"]))
    return out


def coda_floor(st):
    """Delay (s) beyond which this station's direct-P coda has died away, so an arrival there
    is genuinely separate energy. Measured on the station's own semblance trace: the first
    time after the P peak at which semblance drops below max(CODA_ABS, CODA_REL x the
    late-record background) and stays below for CODA_HOLD_S."""
    t, s = st["t"], st["semb"]
    near = (t >= -0.05) & (t <= 0.05)
    if not np.any(np.isfinite(s[near])):
        return np.nan
    t_pk = t[near][int(np.nanargmax(s[near]))]
    bg = np.nanmedian(s[t > t[-1] - 0.9])
    thr = max(CODA_ABS, CODA_REL * bg)
    hold = int(round(CODA_HOLD_S / (t[1] - t[0])))
    below = np.isfinite(s) & (s < thr) & (t > t_pk)
    run = np.convolve(below.astype(int), np.ones(hold, int), mode="valid")
    idx = np.flatnonzero(run == hold)
    if not len(idx):
        return np.nan
    return float(t[idx[0]] + CODA_MARGIN_S)


def sample(st, delay):
    """Semblance of station `st` at `delay` seconds after the direct P, NaN off the record."""
    t = st["t"]
    if not np.isfinite(delay) or delay < t[0] or delay > t[-1]:
        return np.nan
    return float(np.interp(delay, t, st["semb"]))


def phase_delay(prof, phase, z, r, h):
    """Predicted arrival delay of `phase` after the direct P, both traced in the same model."""
    tt, t0 = phase(prof, z, r, h), prof.direct_p(z, r)
    return tt - t0 if (np.isfinite(tt) and np.isfinite(t0)) else np.nan


def migrate(stacks, sp, prof, h, zs, phase, floors):
    """Weighted-mean semblance along `phase`'s moveout at each trial depth, plus the delays
    used and how many stations survived the S mask."""
    score = np.full(len(zs), np.nan)
    nsta = np.zeros(len(zs), int)
    delays = {code: np.full(len(zs), np.nan) for code in stacks}
    for i, z in enumerate(zs):
        vals, wts = [], []
        for code, st in stacks.items():
            tt = phase(prof, z, st["dist_km"], h)
            t0 = prof.direct_p(z, st["dist_km"])
            if not (np.isfinite(tt) and np.isfinite(t0)):
                continue
            d = tt - t0
            delays[code][i] = d
            s = sp.get(code)
            if not (d >= floors.get(code, np.inf)):
                continue
            if s is not None and (s - S_MASK_PRE) <= d <= (s + S_MASK_POST):
                continue
            v = sample(st, d)
            if np.isnan(v):
                continue
            vals.append(v)
            wts.append(np.sqrt(st["n"]))
        if len(vals) >= MIN_STATIONS:
            score[i] = float(np.average(vals, weights=wts))
            nsta[i] = len(vals)
    return score, nsta, delays


def null_distribution(stacks, sp, delays, nsta, rng, floors):
    """Peak score under random per-station delays drawn from that station's own allowed range,
    using the same station count the real migration achieved."""
    allowed = {}
    for code, st in stacks.items():
        t = st["t"]
        ok = t >= floors.get(code, np.inf)
        s = sp.get(code)
        if s is not None:
            ok &= ~((t >= s - S_MASK_PRE) & (t <= s + S_MASK_POST))
        ok &= np.isfinite(st["semb"])
        allowed[code] = st["semb"][ok]
    codes = [c for c in stacks if len(allowed[c])]
    wts = np.array([np.sqrt(stacks[c]["n"]) for c in codes])
    k = int(np.median(nsta[nsta > 0])) if np.any(nsta > 0) else len(codes)
    k = min(max(k, MIN_STATIONS), len(codes))
    n_z = int(np.sum(nsta > 0))
    peaks = np.empty(N_NULL)
    for j in range(N_NULL):
        best = -np.inf
        for _ in range(max(n_z, 1)):
            pick = rng.choice(len(codes), size=k, replace=False)
            vals = np.array([rng.choice(allowed[codes[i]]) for i in pick])
            best = max(best, float(np.average(vals, weights=wts[pick])))
        peaks[j] = best
    return peaks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, required=True)
    ap.add_argument("--component", default="Z")
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    prof = Profile(args.array)
    marks = vels1d_model.bed_markers(args.array)
    h = marks["ice_base"]
    arrivals = pd.read_csv(os.path.join(
        work, f"{args.array.lower()}_cluster_composite_arrivals.csv"))
    zs = np.arange(Z_MIN, Z_MAX + 1e-9, Z_STEP)
    rng = np.random.default_rng(RNG_SEED)

    phases = {"bed_reflection": Profile.bed_reflection,
              "surface_bed_multiple": Profile.surface_bed_multiple}
    rows = []
    for cl in args.clusters:
        stacks = load_stacks(work, args.array, cl, args.component)
        if len(stacks) < MIN_STATIONS:
            print(f"cluster {cl}: only {len(stacks)} stations, skipping")
            continue
        a = arrivals[(arrivals.cluster == cl) & (arrivals.component == args.component)]
        sp = dict(zip(a.station, a.sp_composite))
        z_dd = float(np.median([st["depth_hypodd"] for st in stacks.values()]))
        floors = {code: coda_floor(st) for code, st in stacks.items()}
        print(f"cluster {cl}: P-coda separability floor per station (s) — "
              + ", ".join(f"{c}={floors[c]:.2f}" for c in sorted(floors)
                          if np.isfinite(floors[c])))

        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
        for pi, (pname, fn) in enumerate(phases.items()):
            score, nsta, delays = migrate(stacks, sp, prof, h, zs, fn, floors)
            if not np.any(np.isfinite(score)):
                print(f"cluster {cl} {pname}: no depth has {MIN_STATIONS} clean stations")
                continue
            peaks = null_distribution(stacks, sp, delays, nsta, rng, floors)
            p95, p99 = np.percentile(peaks, [95, 99])
            j = int(np.nanargmax(score))
            cov = zs[nsta >= MIN_STATIONS]
            # What the test says AT hypoDD's depth, using only stations where the reflection
            # would be separable there -- the direct form of the question.
            at_dd, at_dd_n = [], 0
            for code, st in stacks.items():
                d = phase_delay(prof, fn, z_dd, st["dist_km"], h)
                if np.isfinite(d) and d >= floors.get(code, np.inf):
                    v = sample(st, d)
                    if not np.isnan(v):
                        at_dd.append(v)
                        at_dd_n += 1
            rows.append(dict(
                cluster=cl, phase=pname, component=args.component,
                n_stations_max=int(nsta.max()), z_best_km=float(zs[j]),
                score_best=float(score[j]), null_p95=float(p95), null_p99=float(p99),
                exceeds_null_p99=bool(score[j] > p99),
                z_testable_lo=float(cov.min()) if len(cov) else np.nan,
                z_testable_hi=float(cov.max()) if len(cov) else np.nan,
                depth_hypodd_median=z_dd,
                n_separable_at_hypodd=at_dd_n,
                semb_at_hypodd=float(np.mean(at_dd)) if at_dd else np.nan,
            ))
            axes[0].plot(zs, score, color=SERIES[pi], lw=1.6, label=pname.replace("_", " "))
            axes[0].axhline(p99, color=SERIES[pi], lw=1, ls="--")
            axes[1].plot(zs, nsta, color=SERIES[pi], lw=1.2)
            print(f"cluster {cl} {pname}: peak {score[j]:.3f} at z={zs[j]:.2f} km "
                  f"(null p95={p95:.3f} p99={p99:.3f}) "
                  f"{'ABOVE null' if score[j] > p99 else 'within null'}; "
                  f"testable z {cov.min():.2f}-{cov.max():.2f} km"
                  if len(cov) else
                  f"cluster {cl} {pname}: no depth has {MIN_STATIONS} separable stations")

        axes[0].axvline(h, color=INK, lw=1.2)
        axes[0].axvline(z_dd, color=MUTED, lw=1.2, ls=":")
        axes[0].annotate("ice base", (h, axes[0].get_ylim()[1]), fontsize=8,
                         ha="right", va="top", rotation=90, color=SECONDARY)
        axes[0].annotate("hypoDD median", (z_dd, axes[0].get_ylim()[1]), fontsize=8,
                         ha="right", va="top", rotation=90, color=SECONDARY)
        axes[0].set_ylabel("weighted mean semblance along moveout")
        axes[0].legend(fontsize=8, frameon=False)
        axes[1].set_ylabel("stations\nused")
        axes[1].set_xlabel("trial source depth (km)")
        fig.suptitle(f"{args.array} cluster {cl} — depth-phase migration ({args.component}); "
                     f"dashed lines are the 99th percentile of the random-delay null",
                     fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(work, f"{args.array.lower()}_cluster{cl}_depth_phase_migration.png")
        fig.savefig(out, dpi=145, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    if rows:
        df = pd.DataFrame(rows)
        csv = os.path.join(work, f"{args.array.lower()}_depth_phase_migration.csv")
        df.to_csv(csv, index=False)
        print(f"\nwrote {csv}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
